import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import pdb
from bisect import bisect_right


from models.oaclip_ov import OACLIPv3
from dataset import CompositionDataset
import evaluator as evaluator_ge
from tqdm import tqdm

from utils import utils
from config import cfg
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


list_stats_report = ['AUC','best_hm','sa_so_acc', 'sa_so_u_acc', 'sa_uo_acc', 'ua_so_acc', 'ua_uo_acc', 's_a_acc', 'u_a_acc', 's_o_acc', 'u_o_acc']
def freeze(m):
    """Freezes module m.
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None

def decay_learning_rate(optimizer, cfg):
    """Decays learning rate using the decay factor in cfg.
    """
    print('# of param groups in optimizer: %d' % len(optimizer.param_groups))
    param_groups = optimizer.param_groups
    for i, p in enumerate(param_groups):
        current_lr = p['lr']
        new_lr = current_lr * cfg.TRAIN.decay_factor
        print(f'Group {i}: current lr = {current_lr:.8f}, decay to lr = {new_lr:.8f}')
        p['lr'] = new_lr


def decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg):
    """Decays learning rate following milestones in cfg.
    """
    milestones = cfg.TRAIN.lr_decay_milestones
    it = bisect_right(milestones, epoch)
    gamma = cfg.TRAIN.decay_factor ** it
    
    gammas = [gamma] * len(group_lrs)
    assert len(optimizer.param_groups) == len(group_lrs)
    i = 0
    for param_group, lr, gamma_group in zip(optimizer.param_groups, group_lrs, gammas):
        param_group["lr"] = lr * gamma_group
        i += 1
        print(f"Group {i}, lr = {lr * gamma_group}")


def save_checkpoint(model_or_optim, name, cfg):
    """Saves checkpoint.
    """
    if isinstance(model_or_optim, nn.parallel.DistributedDataParallel):
        state_dict = model_or_optim.module.state_dict()
    else:
        state_dict = model_or_optim.state_dict()
    path = os.path.join(
        f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.seed}/{name}.pth')
    torch.save(state_dict, path)


def train(epoch, model, optimizer, trainloader, logger, device, cfg):
    model.train()
    if not cfg.TRAIN.finetune_backbone and not cfg.TRAIN.use_precomputed_features:
        m = model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            m = model.module
        freeze(m.feat_extractor)

    if device == 'cuda:0':
        # Tracker.
        # Name of all losses.
        list_meters = [
            'loss_total'
        ]
        
        if cfg.MODEL.name == 'oaclipv3':
            if cfg.MODEL.use_obj_loss:
                list_meters.append('loss_aux_obj')
                list_meters.append('acc_aux_obj')
            if cfg.MODEL.use_attr_loss:
                list_meters.append('loss_aux_attr')
                list_meters.append('acc_aux_attr')
            if cfg.MODEL.use_emb_pair_loss:
                list_meters.append('emb_loss')
            if cfg.MODEL.use_composed_pair_loss:
                list_meters.append('composed_unseen_loss')
                list_meters.append('composed_seen_loss')

        dict_meters = { 
            k: utils.AverageMeter() for k in list_meters
        }

        acc_attr_meter = utils.AverageMeter()
        acc_obj_meter = utils.AverageMeter()
        acc_pair_meter = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        end_time = time.time()

    start_iter = (epoch - 1) * len(trainloader)

    for idx, batch in enumerate(tqdm(trainloader)):
        it = start_iter + idx + 1
        if device == 'cuda:0':
            data_time.update(time.time() - end_time)

        for k in batch:
            if isinstance(batch[k], list): 
                continue
            batch[k] = batch[k].to(device, non_blocking=True)
        out = model(batch)

        loss = out['loss_total']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device == 'cuda:0':
            if 'acc_attr' in out:
                acc_attr_meter.update(out['acc_attr'])
                acc_obj_meter.update(out['acc_obj'])
            acc_pair_meter.update(out['acc_pair'])
            for k in out:
                if k in dict_meters:
                    dict_meters[k].update(out[k].item())
            batch_time.update(time.time() - end_time)
            end_time = time.time()

        if (idx + 1) % cfg.TRAIN.disp_interval == 0 and device == 'cuda:0':
            print(
                f'Epoch: {epoch} Iter: {idx+1}/{len(trainloader)}, '
                f'Loss: {dict_meters["loss_total"].avg:.3f}, '
                f'Acc_Pair: {acc_pair_meter.avg:.2f}, '
                f'Batch_time: {batch_time.avg:.3f}, Data_time: {data_time.avg:.3f}',
                flush=True)

            for k in out:
                if k in dict_meters:
                    logger.add_scalar('train/%s' % k, dict_meters[k].avg, it)
                    
                logger.add_scalar('train/acc_attr', acc_attr_meter.avg, it)
                logger.add_scalar('train/acc_obj', acc_obj_meter.avg, it)
               
            logger.add_scalar('train/acc_pair', acc_pair_meter.avg, it)
            
            batch_time.reset()
            data_time.reset()
            acc_pair_meter.reset()
            if 'acc_attr' in out:
                acc_attr_meter.reset()
                acc_obj_meter.reset()
            for k in out:
                if k in dict_meters:
                    dict_meters[k].reset()

def validate_ge(epoch, model, testloader, evaluator, device, phase='val'):
    model.eval()

    # All pairs in the whole dataset, and their objs and attrs.
    pairs = testloader.dataset.pairs
    objs = testloader.dataset.objs
    attrs = testloader.dataset.attrs

    dset = testloader.dataset
    val_attrs, val_objs = zip(*dset.pairs)
    val_attrs = [dset.attr2idx[attr] for attr in val_attrs]
    val_objs = [dset.obj2idx[obj] for obj in val_objs]
    model.val_attrs = torch.LongTensor(val_attrs).cuda()
    model.val_objs = torch.LongTensor(val_objs).cuda()
    model.val_pairs = dset.pairs

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        for k in data:
            if isinstance(data[k], list): 
                continue
            data[k] = data[k].to(device, non_blocking=True)

        out = model(data)
        predictions = out['scores']

        attr_truth, obj_truth, pair_truth = data['attr'], data['obj'], data['pair']

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
        'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k].to('cpu') for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=1e3, topk=1)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=1)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    print(f'Test Epoch: {epoch}')
    print(result)

    del model.val_attrs
    del model.val_objs
    torch.cuda.empty_cache()
    return stats

def main_worker(gpu, cfg):
    """Main training code.
    """
    seed = cfg.TRAIN.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Use GPU {gpu} for training', flush=True)
    torch.cuda.set_device(gpu)
    device = f'cuda:{gpu}'

    # Setup distributed setting.
    if cfg.DISTRIBUTED.world_size > 1:
        dist.init_process_group(
            backend=cfg.DISTRIBUTED.backend,
            init_method='tcp://127.0.0.1:1426',
            world_size=cfg.DISTRIBUTED.world_size,
            rank=gpu
        )

    if gpu == 0:
        # Log directory for tensorboard.
        log_dir = f'{cfg.TRAIN.log_dir}/{cfg.config_name}_{cfg.TRAIN.seed}'
        logger = SummaryWriter(log_dir=log_dir)

        # Directory to save checkpoints.
        ckpt_dir = f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.seed}'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    else:
        logger = None

    # Distribute batch size evenly between GPUs.
    cfg.TRAIN.batch_size = cfg.TRAIN.batch_size // cfg.DISTRIBUTED.world_size
    print('Batch size on each gpu: %d' % cfg.TRAIN.batch_size)

    # Prepare dataset & dataloader.
    print('Prepare dataset')

    trainset = CompositionDataset(
        phase='train', split=cfg.DATASET.splitname, cfg=cfg)

    if cfg.DISTRIBUTED.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.TRAIN.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.TRAIN.num_workers // cfg.DISTRIBUTED.world_size,
        pin_memory=True, sampler=train_sampler, drop_last=False)

    if gpu == 0:
        valset = CompositionDataset(
            phase='val', split=cfg.DATASET.splitname, cfg=cfg)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
            num_workers=cfg.TRAIN.num_workers)

        testset = CompositionDataset(
            phase='test', split=cfg.DATASET.splitname, cfg=cfg)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
            num_workers=cfg.TRAIN.num_workers)

    if cfg.MODEL.name == 'oaclipv3':
        model = OACLIPv3(trainset, cfg)
    model.to(device)
    freeze(model.attr_embedder)
    freeze(model.obj_embedder)
    freeze(model.pair_embedder)

    if not cfg.TRAIN.finetune_backbone and not cfg.TRAIN.use_precomputed_features:
        freeze(model.feat_extractor)
    ## to print the number of parameters
    # total_params = utils.count_parameters(model)
    

    # Prepare distributed.
    if cfg.DISTRIBUTED.world_size > 1:
        print('Wrap model with DistributedDataParallel')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], broadcast_buffers=False)

    if gpu == 0:
        m = model
        if isinstance(m, nn.parallel.DistributedDataParallel):
            m = m.module
        evaluator_val_ge = evaluator_ge.Evaluator(valset, cfg)
        evaluator_test_ge = evaluator_ge.Evaluator(testset, cfg)
    
    torch.backends.cudnn.benchmark = True

    params_word_embedding = []
    params_encoder = []
    params = []
    ### for printing which layers are being trained
    # for name, p in model.named_parameters():
    #     if not p.requires_grad:
    #         continue

    #     if 'attr_embedder' in name or 'obj_embedder' in name:
    #         if cfg.TRAIN.lr_word_embedding > 0:
    #             params_word_embedding.append(p)
    #             if gpu == 0:
    #                 print('params_word_embedding: %s' % name)
    #     elif name.startswith('feat_extractor'):
    #         params_encoder.append(p)
    #         if gpu == 0:
    #             print('params_encoder: %s' % name)
    #     else:
    #         params.append(p)
    #         if gpu == 0:
    #             print('params_main: %s' % name)

    if cfg.TRAIN.lr_word_embedding > 0:
        optimizer = optim.Adam([
            {'params': params_encoder, 'lr': cfg.TRAIN.lr_encoder, 'weight_decay': cfg.TRAIN.wd_encoder},
            {'params': params_word_embedding, 'lr': cfg.TRAIN.lr_word_embedding},
            {'params': params, 'lr': cfg.TRAIN.lr},
        ], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)  
        group_lrs = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr_word_embedding, cfg.TRAIN.lr]
    else:
        optimizer = optim.Adam([
            {'params': params_encoder, 'lr': cfg.TRAIN.lr_encoder, 'weight_decay': cfg.TRAIN.wd_encoder},
            {'params': params, 'lr': cfg.TRAIN.lr},
        ], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)
        group_lrs = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr]

    start_epoch = cfg.TRAIN.start_epoch
    epoch = start_epoch



    best_records = {
        'val/best_auc': 0.0}
       

    for i in list_stats_report:
        name1 = 'val/'+i
        name2 = 'test/'+i
        best_records[name1] = 0.0
        best_records[name2] = 0.0


    best_auc = -1
    n_wait = 0
    n_patience = cfg.TRAIN.decay_patience
    last_time_eval_on_test = 0

    # for epoch in range(start_epoch, cfg.TRAIN.max_epoch + 1):
    while epoch <= cfg.TRAIN.max_epoch:
        epoch_time = time.time()
        if cfg.DISTRIBUTED.world_size > 1:
            train_sampler.set_epoch(epoch)

        train(epoch, model, optimizer, trainloader, logger, device, cfg)

        if gpu == 0:
            max_gpu_usage_mb = torch.cuda.max_memory_allocated(device=device) / 1048576.0
            print(f'Max GPU usage in MB till now: {max_gpu_usage_mb}')

        if cfg.TRAIN.decay_strategy == 'milestone':
            decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg)

        if epoch < cfg.TRAIN.start_epoch_validate:
            epoch += 1
            continue

        if gpu == 0 and epoch % cfg.TRAIN.eval_every_epoch == 0:
            # Validate.
            m = model
            if isinstance(m, nn.parallel.DistributedDataParallel):
                m = m.module

            print('Validation set ===>')
            stats_val = validate_ge(epoch, m, valloader, evaluator_val_ge, device)
            rep = {}
            for i in list_stats_report:
                name_ = 'val/'+i
                val = stats_val[i]
                if i == 'AUC':
                    auc = val
                if i == 'best_hm':
                    best_hm = val
                rep[name_] = val
                best_records[name_] = val
                logger.add_scalar(name_, val, epoch * len(trainloader))
               
            if  epoch == cfg.TRAIN.max_epoch and epoch+1 < cfg.TRAIN.final_max_epoch:
                cfg.TRAIN.max_epoch += 1

            if cfg.TRAIN.decay_strategy == 'plateau':
                if auc > best_auc:
                    best_auc = auc
                    n_wait = 0 # Reset waiting counter.
                else:
                    n_wait += 1
                    if n_wait >= n_patience:
                        decay_learning_rate(optimizer, cfg)
                        n_wait = 0 # Reset waiting counter.
                        n_patience += 1 # Increase patience.

            if auc > best_records['val/best_auc']:
                best_records['val/best_auc'] = auc
                best_records['val/best_hm'] = best_hm
                if gpu == 0 :
                    save_checkpoint(model, f'model_epoch{epoch}', cfg)
            # if epoch > 1:
                print('Evaluate on test set')
                # Test.
                stats_test = validate_ge(epoch, m, testloader, evaluator_test_ge, device,'test')
                last_time_eval_on_test = epoch
                for i in list_stats_report:
                    name_ = 'test/'+i
                    val = stats_test[i]
                    best_records[name_] = val
                    logger.add_scalar(name_, val, epoch * len(trainloader))

            # If have waited too long from the last time we evaluated on test set.
            if epoch % cfg.TRAIN.eval_every_epoch == 0 and last_time_eval_on_test !=  epoch:
                print("It's been a long time since we last evaluated on test set")
                # Test.
                stats_test = validate_ge(epoch, m, testloader, evaluator_test_ge, device,'test')
                last_time_eval_on_test = epoch
                for i in list_stats_report:
                    name_ = 'test/'+i
                    val = stats_test[i]
                    best_records[name_] = val
                    logger.add_scalar(name_, val, epoch * len(trainloader))
                   
                if gpu == 0 and epoch > 30 and epoch % 5 == 1:
                    save_checkpoint(model, f'model_epoch{epoch}', cfg)


        epoch += 1

    if gpu == 0:
        logger.close()
    
    if cfg.DISTRIBUTED.world_size > 1:
        dist.destroy_process_group()

    print('Done: %s' % cfg.config_name)
    print('New Best AUC:',best_records['test/auc_at_best_val'])
    print('New Best HM:',best_records['test/hm_at_best_val'])
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    print(cfg)

    seed = cfg.TRAIN.seed
    if seed == -1:
        seed = np.random.randint(1, 10000)
    print('Random seed:', seed)
    cfg.TRAIN.seed = seed

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    
    if cfg.DISTRIBUTED.world_size > 1:
        mp.spawn(main_worker, nprocs=cfg.DISTRIBUTED.world_size, args=(cfg,))
    else:
        main_worker(0, cfg)


if __name__ == "__main__":
    main()

    