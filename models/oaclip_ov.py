import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from .backbone import Backbone
from .basic_layers import MLP
from .word_embedding_utils import initialize_wordembedding_matrix
from .resnet_comb_feature import comb_resnet
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import pdb

## Label Smoothing using manual weights, seems to work for training labels only, not sure if we add neighbors
class LabelSmoothingCrossEntropy_pair(_WeightedLoss):
    def __init__(self,smoothing=0.0, weight=None, reduction='mean',):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets
    
    def k_one_hot_weighted_67(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            ## for smoothing = 0.9, 10% goes to lbl, 23% goes to neighbors, 67% goes to rest
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def k_one_hot_weighted_smoothing(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            ## for smoothing = 0.9, 10% goes to lbl+neighbors, 30% goes to neighbors, 90% goes to rest
            new1 = torch.empty(size=(targets.size(0), n_classes),device=targets.device).fill_(smoothing/(n_classes-6))
            new2 = new1.scatter_(1, targets.data.unsqueeze(1), (1-smoothing)*0.5)  # add 10% to lbl 
            n_weights = ((1-smoothing)*0.5)/5 
            orig_num_cls = n_classes - 5
            n1 = (torch.ones(targets.size(0))*orig_num_cls).to(torch.int64).cuda()
            n2 = (torch.ones(targets.size(0))*(orig_num_cls+1)).to(torch.int64).cuda()
            n3 = (torch.ones(targets.size(0))*(orig_num_cls+2)).to(torch.int64).cuda()
            n4 = (torch.ones(targets.size(0))*(orig_num_cls+3)).to(torch.int64).cuda()
            n5 = (torch.ones(targets.size(0))*(orig_num_cls+4)).to(torch.int64).cuda()
            
            tar = new2.scatter_(1, n1.data.unsqueeze(1), n_weights)
            tar = tar.scatter_(1, n2.data.unsqueeze(1), n_weights)
            tar = tar.scatter_(1, n3.data.unsqueeze(1), n_weights)
            tar = tar.scatter_(1, n4.data.unsqueeze(1), n_weights)
            targets = tar.scatter_(1, n5.data.unsqueeze(1), n_weights)
            
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1
        targets1 = self.k_one_hot_weighted_smoothing(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets1 * log_preds).sum(dim=-1))

class LabelSmoothingCrossEntropy(_WeightedLoss):
    def __init__(self,smoothing=0.0, weight=None, reduction='mean',):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets
    
    
    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1
        targets1 = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets1 * log_preds).sum(dim=-1))


class OACLIPv3(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """
    def __init__(self, dset, cfg):
        super(OACLIPv3, self).__init__()
        self.cfg = cfg

        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.pair2idx = dset.pair2idx

        # Set training pairs.
        train_attrs, train_objs = zip(*dset.train_pairs)
        train_attrs = [dset.attr2idx[attr] for attr in train_attrs]
        train_objs = [dset.obj2idx[obj] for obj in train_objs]
        train_pairs = [dset.pair2idx[pair] for pair in dset.train_pairs]
        self.train_attrs = torch.LongTensor(train_attrs).cuda()
        self.train_objs = torch.LongTensor(train_objs).cuda()
        self.train_pairs = torch.LongTensor(train_pairs).cuda()

        train_attrs1 = dset.train_attrs
        train_objs1 = dset.train_objs
        train_attrs1 = [dset.attr2idx[attr] for attr in train_attrs1]
        train_objs1 = [dset.obj2idx[obj] for obj in train_objs1]
        self.train_attrs1 = torch.LongTensor(train_attrs1).cuda()
        self.train_objs1 = torch.LongTensor(train_objs1).cuda()

        train_attrs2 = dset.train_attrs_extra
        train_objs2 = dset.train_objs_extra
        train_attrs2 = [dset.train_extra_attr2idx[attr] for attr in train_attrs2]
        train_objs2 = [dset.train_extra_obj2idx[obj] for obj in train_objs2]
        self.train_attrs2 = torch.LongTensor(train_attrs2).cuda()
        self.train_objs2 = torch.LongTensor(train_objs2).cuda()
    
        
        all_pairs = [dset.pair2idx[pair] for pair in dset.pairs]
        self.all_pairs = torch.LongTensor(all_pairs).cuda()
        self.all_pairs1 = dset.pairs

        all_attr = [dset.attr2idx[attr] for attr in dset.all_attrs]
        self.all_attrs = torch.LongTensor(all_attr).cuda()
        all_obj = [dset.obj2idx[obj] for obj in dset.all_objs]
        self.all_objs = torch.LongTensor(all_obj).cuda()

        test_pairs = [dset.pair2idx[pair] for pair in dset.test_pairs]
        self.test_pairs = torch.LongTensor(test_pairs).cuda()
        self.test_pairs1 = dset.test_pairs 

        unseen_pair_attrs, unseen_pair_objs = zip(*dset.unseen_pairs)
        unseen_pair_attrs = [dset.attr2idx[attr] for attr in unseen_pair_attrs]
        unseen_pair_objs = [dset.obj2idx[obj] for obj in unseen_pair_objs]
        self.unseen_pair_attrs = torch.LongTensor(unseen_pair_attrs).cuda()
        self.unseen_pair_objs = torch.LongTensor(unseen_pair_objs).cuda()
        unseen_pairs = [dset.pair2idx[pair] for pair in dset.unseen_pairs]
        self.unseen_pairs = torch.LongTensor(unseen_pairs).cuda()
        
        ## extra pairs
        self.extra_obj2idx = {obj: idx for idx, obj in enumerate(dset.extra_objs)}
        self.extra_attr2idx = {attr: idx for idx, attr in enumerate(dset.extra_attrs)}
        self.extra_pair2idx = {pair: idx for idx, pair in enumerate(dset.extra_pairs)}
        extra_objs = [self.extra_obj2idx[obj] for obj in dset.extra_objs]
        extra_attrs = [self.extra_attr2idx[attr] for attr in dset.extra_attrs]
        extra_pairs = [self.extra_pair2idx[pair] for pair in dset.extra_pairs]
        self.extra_attrs = torch.LongTensor(extra_attrs).cuda()
        self.extra_objs = torch.LongTensor(extra_objs).cuda()
        self.extra_pairs = torch.LongTensor(extra_pairs).cuda()
        
        train_extra_pairs = [dset.train_extra_pair2idx[pair] for pair in dset.train_pairs_extra]
        self.train_extra_pairs = torch.LongTensor(train_extra_pairs).cuda()

        # Dimension of the joint image-label embedding space.
        if '+' in cfg.MODEL.wordembs:
            self.emb_dim = cfg.MODEL.emb_dim*2
            self.attr_emb_dim = cfg.MODEL.attr_emb_dim*2
            self.obj_emb_dim = cfg.MODEL.obj_emb_dim*2
        else:
            self.emb_dim = cfg.MODEL.emb_dim
            self.attr_emb_dim = cfg.MODEL.attr_emb_dim
            self.obj_emb_dim = cfg.MODEL.obj_emb_dim

        # Setup layers for word embedding composer.
        self._setup_word_composer(dset, cfg)
        self.clip_type = cfg.TRAIN.clip_type
        if not cfg.TRAIN.use_precomputed_features and not cfg.TRAIN.comb_features:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.feat_extractor = Backbone('resnet18')
            feat_dim = 512
        

        self.drop_input = cfg.MODEL.drop_input > 0
        if self.drop_input:
            self.drop_inp = nn.Dropout2d(cfg.MODEL.drop_input)

        self.img_emb_method = cfg.MODEL.img_emb_method
        if self.clip_type == 'vit':
            img_emb_modules = [
                nn.Conv1d(feat_dim, cfg.MODEL.img_emb_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(cfg.MODEL.img_emb_dim),
                nn.ReLU()
            ]
        elif self.clip_type == 'resnet':
            img_emb_modules = [
                nn.Conv2d(cfg.MODEL.img_emb_dim, cfg.MODEL.img_emb_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(cfg.MODEL.img_emb_dim),
                nn.ReLU()
            ]
        else:
            img_emb_modules = [
                nn.Conv2d(feat_dim, cfg.MODEL.img_emb_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(cfg.MODEL.img_emb_dim),
                nn.ReLU()
            ]
        feat_dim = cfg.MODEL.img_emb_dim

        if cfg.MODEL.img_emb_drop > 0:
            img_emb_modules += [
                nn.Dropout2d(cfg.MODEL.img_emb_drop)]

        if self.img_emb_method == 'conv_end':
            img_emb_modules += [
                nn.Conv2d(cfg.MODEL.img_emb_dim, self.emb_dim, kernel_size=1)]
        
        self.img_embedder = nn.Sequential(*img_emb_modules)

        self.use_fg_estimator = cfg.MODEL.use_fg_estimator
        if self.use_fg_estimator:
            self.fg_estimator = nn.Linear(feat_dim, 1)
        else:
            self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.img_emb_method == 'conv_interm':
            self.img_final = nn.Linear(feat_dim, self.emb_dim, bias=cfg.MODEL.img_final_bias)
        if cfg.MODEL.img_drop_last > 0:
            self.drop_last = nn.Dropout2d(cfg.MODEL.img_drop_last)

        self.classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)
        self.label_smoothing = LabelSmoothingCrossEntropy_pair(cfg.MODEL.smoothing) # reduction='sum' #SmoothCrossEntropyLoss(smoothing=cfg.MODEL.smoothing)
        self.image_pair_comparison = ImagePairComparison(
            cfg, self.num_attrs, self.num_objs, self.train_attrs1, self.train_objs1,
            self.train_attrs2, self.train_objs2,self.attr_embedder, self.obj_embedder,
            img_dim=feat_dim,
            emb_dim=self.emb_dim, attr_emb_dim=self.attr_emb_dim, obj_emb_dim=self.obj_emb_dim,
            word_dim=self.word_dim,
            lambda_attn=cfg.MODEL.lambda_attn,
            attn_normalized=cfg.MODEL.attn_normalized,
            low_dim_cross_att=cfg.MODEL.low_dim_cross_att,
            cross_att_dim=cfg.MODEL.cross_att_dim,
            image_pair_multihead_attn=cfg.MODEL.image_pair_multihead_attn,
        )

        self.pair_final = nn.Linear(2*self.emb_dim, self.emb_dim)

    def _setup_word_composer(self, dset, cfg):
        attr_wordemb, self.word_dim = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs+'_attr', dset.unique_attrs, cfg)
        obj_wordemb, _ = \
            initialize_wordembedding_matrix(cfg.MODEL.wordembs+'_obj', dset.unique_objs, cfg)
        if 'bert' in cfg.MODEL.wordembs:
            self.word_dim = 768
        elif 'glv' in cfg.MODEL.wordembs:
            self.word_dim = 300
        else:
            self.word_dim = 512
        self.attr_embedder = nn.Embedding(len(dset.unique_attrs), self.word_dim)
        self.obj_embedder = nn.Embedding(len(dset.unique_objs), self.word_dim)
        self.attr_embedder.weight.data.copy_(attr_wordemb)
        self.obj_embedder.weight.data.copy_(obj_wordemb)

        self.pair_embedder = nn.Embedding(len(dset.unique_pairs), self.word_dim)
        pretrained_weight, _ = initialize_wordembedding_matrix(cfg.MODEL.wordembs+'_extra',dset.unique_pairs,cfg)
        self.pair_embedder.weight.data.copy_(pretrained_weight)

        # Dimension of the joint image-label embedding space.
        if '+' in cfg.MODEL.wordembs:
            emb_dim = cfg.MODEL.emb_dim*2
        else:
            emb_dim = cfg.MODEL.emb_dim

        self.wordemb_compose = cfg.MODEL.wordemb_compose
        if cfg.MODEL.wordemb_compose == 'linear':
            # Linear composer.
            self.compose = nn.Sequential(
                nn.Linear(self.word_dim, self.emb_dim)
            )
        elif cfg.MODEL.wordemb_compose == 'mlp':
            # Nonlinear composer.
            self.compose = nn.Sequential(
                nn.Linear(self.word_dim*2, self.word_dim*2),
                nn.ReLU(),
                nn.Linear(self.word_dim*2, emb_dim)
            )
        elif cfg.MODEL.wordemb_compose == 'obj-conditioned':
            # Composer conditioned on object.
            if cfg.MODEL.wordemb_object_code == 'mlp':
                self.object_code = MLP(
                    self.word_dim, 300, 600, 2, batchnorm=False
                )
            else:
                self.object_code = nn.Sequential(
                    nn.Linear(self.word_dim, 600),
                    nn.ReLU(True)
                )
            if cfg.MODEL.wordemb_attribute_code == 'mlp':
                self.attribute_code = MLP(
                    self.word_dim, 300, 600, 2, batchnorm=False
                )
            else:
                self.attribute_code = nn.Sequential(
                    nn.Linear(self.word_dim, 600),
                    nn.ReLU(True)
                )
            if cfg.MODEL.wordemb_attribute_code_fc == 'mlp':
                self.attribute_code_fc = MLP(
                    600, 600, 600, 2, batchnorm=False
                )
            else:
                self.attribute_code_fc = nn.Sequential(
                    nn.Linear(600, 600),
                    nn.ReLU(True),
                )
            if cfg.MODEL.wordemb_compose_final == 'mlp':
                self.compose = MLP(
                    self.word_dim + 600, 900, emb_dim, 2, batchnorm=False,
                    drop_input=cfg.MODEL.wordemb_compose_dropout,
                    final_linear_bias=cfg.MODEL.wordemb_final_bias
                )
            else:
                self.compose = MLP(
                    self.word_dim + 600, 600, emb_dim, 2, batchnorm=False,
                    drop_input=cfg.MODEL.wordemb_compose_dropout,
                    final_linear_bias=cfg.MODEL.wordemb_final_bias
                )

    def compose_word_embeddings(self, mode='train', pool_of_pairs=None):
        if mode == 'train':
            attr_emb = self.attr_embedder(self.train_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.train_objs) # # [n_pairs, word_dim].
        elif mode == 'all':
            attr_emb = self.attr_embedder(self.all_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.all_objs)
        elif mode == 'unseen':
            attr_emb = self.attr_embedder(self.unseen_pair_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.unseen_pair_objs)
        elif mode == 'val':
            attr_emb = self.attr_embedder(self.val_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.val_objs)
        elif mode == 'train_extra':
            attr_emb = self.attr_embedder(self.train_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.train_objs)
        else:
            # Expect val_attrs and val_objs are already set (using _set_val_pairs()).
            attr_emb = self.attr_embedder(self.val_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.val_objs) # # [n_pairs, word_dim].

        if 'obj-conditioned' in self.cfg.MODEL.wordemb_compose:
            object_c = self.object_code(obj_emb) # [n_pairs, 1024].
            attribute_c = self.attribute_code(attr_emb) # [n_pairs, 1024].
            attribute_c = self.attribute_code_fc(object_c * attribute_c)
            concept_emb = torch.cat((obj_emb, attribute_c), dim=-1) # [n_pairs, word_dim + 1024].
        elif 'clip' in self.cfg.MODEL.wordembs and mode == 'train':
            concept_emb = self.pair_embedder(self.train_pairs)
        elif 'clip' in self.cfg.MODEL.wordembs and mode == 'extra':
            concept_emb = self.pair_embedder(self.extra_pairs)
        elif 'clip' in self.cfg.MODEL.wordembs and mode == 'train_extra':
            concept_emb = self.pair_embedder(self.train_extra_pairs)
        elif 'clip' in self.cfg.MODEL.wordembs and mode == 'all':
            concept_emb = self.pair_embedder(self.all_pairs)
        elif 'clip' in self.cfg.MODEL.wordembs and mode == 'unseen':
            concept_emb = self.pair_embedder(self.unseen_pairs)
        else:
            concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
        
        
        concept_emb = self.compose(concept_emb) # [n_pairs, emb_dim].

        if pool_of_pairs is not None:
            # If we have custom set of negative pairs, gather them.
            concept_emb = torch.gather(
                concept_emb.unsqueeze(0).repeat(pool_of_pairs.shape[0], 1, 1),
                1, pool_of_pairs.unsqueeze(2).repeat(1, 1, concept_emb.shape[-1]))

        return concept_emb

    def train_forward(self, batch):
        img1 = batch['img']
        img2_a = batch['img1_a'] # Image that shares the same attribute
        img2_o = batch['img1_o'] # Image that shares the same object

        # Labels of 1st image.
        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']

        attr2_labels_a = batch['attr1_a'] # attr labels of 2nd image
        obj2_labels_a = batch['obj1_a'] # obj labels of 2nd image

        attr2_labels_o = batch['attr1_o'] # attr labels of 3rd image
        obj2_labels_o = batch['obj1_o'] # obj labels of 3rd image

        composed_unseen_pair = batch['composed_unseen_pair']
        composed_seen_pair = batch['composed_seen_pair']

        mask_task = batch['mask_task']
        bs = img1.shape[0]

        if self.cfg.TRAIN.sample_negative_pairs != -1:
            pool_of_pairs = batch['pool_of_pairs'] # [bs, n_pool].
            # We explicitly set positive label at index 0 (look at DataLoader code).
            pair_labels = torch.zeros(bs).to(img1.device).long()
        else:
            pool_of_pairs = None
        if self.cfg.MODEL.use_extra_pair_loss:
            concept = self.compose_word_embeddings(mode='train_extra', pool_of_pairs=pool_of_pairs)
            #[6148,300]
            concept_train_only = self.compose_word_embeddings(mode='train', pool_of_pairs=pool_of_pairs)
        else:
            concept = self.compose_word_embeddings(
                mode='train', pool_of_pairs=pool_of_pairs) # [501,512,300](n_pairs, emb_dim) or (bs, n_pairs, emb_dim)
            concept_train_only = concept

        if not self.cfg.TRAIN.use_precomputed_features and not self.cfg.TRAIN.comb_features:
            if self.clip_type  == 'vit':
                img1 = self.feat_extractor.encode_image(img1).float().permute(0,2,1) # Bx512x49
                img2_a = self.feat_extractor.encode_image(img2_a).float().permute(0,2,1)
                img2_o = self.feat_extractor.encode_image(img2_o).float().permute(0,2,1)

                img1 = self.img_embedder(img1) #B x 2048 x 49
                img2_a = self.img_embedder(img2_a)
                img2_o = self.img_embedder(img2_o)
            elif self.clip_type == 'resnet':
                img1 = self.feat_extractor.encode_image(img1).float() # Bx2048x7x7
                img2_a = self.feat_extractor.encode_image(img2_a).float()
                img2_o = self.feat_extractor.encode_image(img2_o).float()
                h, w = img1.shape[2:]
                img1 = self.img_embedder(img1).view(bs, -1, h*w) #B x 2048 x 49
                img2_a = self.img_embedder(img2_a).view(bs, -1, h*w)
                img2_o = self.img_embedder(img2_o).view(bs, -1, h*w)
            else:
                img1 = self.feat_extractor(img1)[0]   #B x 512 x 7 x 7
                img2_a = self.feat_extractor(img2_a)[0]
                img2_o = self.feat_extractor(img2_o)[0]
                h, w = img1.shape[2:]
                img1 = self.img_embedder(img1).view(bs, -1, h*w)  #B x 2048 x 49
                img2_a = self.img_embedder(img2_a).view(bs, -1, h*w)
                img2_o = self.img_embedder(img2_o).view(bs, -1, h*w)
        
        at_neigh = {'n1': batch['at1'], 'n2': batch['at2'], 'n3': batch['at3'], 'n4':batch['at4'], 'n5': batch['at5']}
        ob_neigh = {'n1':batch['ob1'], 'n2':batch['ob2'], 'n3': batch['ob3'], 'n4':batch['ob4'], 'n5': batch['ob5']}

        aux_loss = self.image_pair_comparison(
            img1, img2_a, img2_o, attr_labels, obj_labels, at_neigh, ob_neigh, mask_task)
        
        
        if self.use_fg_estimator:
            fg1 = F.softmax(self.fg_estimator(img1.transpose(1, 2)), dim=1) # (bs, L, 1)
            img1 = torch.matmul(img1, fg1).squeeze()
        else:
            h = w = 7
            img1 = self.img_avg_pool(img1.view(bs, -1, h, w)).squeeze()

        if self.img_emb_method == 'conv_interm':
            img1 = self.img_final(img1)
        
        if self.cfg.MODEL.img_drop_last > 0:
            img1 = self.drop_last(img1)
        
        
        pred = self.classifier(img1, concept_train_only)
        pred_extra = self.classifier(img1, concept_train_only) #self.classifier(img1, concept)
        pred_extra1 = self.classifier(img1, concept)
        
        if pool_of_pairs is None:
            pair_loss = F.cross_entropy(pred, pair_labels)
            loss1 = pair_loss * (1.0 - self.cfg.MODEL.extra_pair_loss_ratio)

            pred = torch.max(pred, dim=1)[1]
            attr_pred = self.train_attrs[pred]
            obj_pred = self.train_objs[pred]

            correct_attr = (attr_pred == attr_labels)
            correct_obj = (obj_pred == obj_labels)
            correct_pair = (pred == pair_labels)
        else:
            pair_loss = F.cross_entropy(pred, pair_labels)
            loss1 = pair_loss * (1.0 - self.cfg.MODEL.extra_pair_loss_ratio) #* self.cfg.MODEL.w_loss_main

            pred = torch.max(pred, dim=1)[1] # (bs)
            true_pair_labels = torch.gather(pool_of_pairs, 1, pred.unsqueeze(1)).squeeze(1) # (bs)
            attr_pred = self.train_attrs[true_pair_labels]
            obj_pred = self.train_objs[true_pair_labels]

            correct_attr = (attr_pred == attr_labels)
            correct_obj = (obj_pred == obj_labels)
            correct_pair = (pred == pair_labels)

        
        if self.cfg.MODEL.use_extra_pair_loss:
            sel_ind = pred_extra1.gather(1, pair_labels.long().view(-1,1)).squeeze()
            n1 = pred_extra1.gather(1, batch['lbl1'].long().view(-1,1)).squeeze()
            n2 = pred_extra1.gather(1, batch['lbl2'].long().view(-1,1)).squeeze()
            n3 = pred_extra1.gather(1, batch['lbl3'].long().view(-1,1)).squeeze()
            n4 = pred_extra1.gather(1, batch['lbl4'].long().view(-1,1)).squeeze()
            n5 = pred_extra1.gather(1, batch['lbl5'].long().view(-1,1)).squeeze()
           
            #5neigh
            new_pred = torch.cat([pred_extra,n1.unsqueeze(1),n2.unsqueeze(1),n3.unsqueeze(1),n4.unsqueeze(1),n5.unsqueeze(1)], axis=-1)
            
            #3neigh
            # new_pred = torch.cat([pred_extra,n1.unsqueeze(1),n2.unsqueeze(1),n3.unsqueeze(1)], axis=-1)
            # 1 neigh
            # new_pred = torch.cat([pred_extra,n1.unsqueeze(1)], axis=-1)
            
            pair_loss_ex = self.label_smoothing(new_pred, pair_labels)
            
            loss1 += pair_loss_ex * self.cfg.MODEL.extra_pair_loss_ratio
        
        loss = loss1 * self.cfg.MODEL.w_loss_main
           
        if self.cfg.MODEL.use_attr_loss:
            loss_attr = (1.0 - self.cfg.MODEL.extra_attr_loss_ratio) * aux_loss['loss_attr'] 
            if self.cfg.MODEL.use_extra_attr_loss and self.cfg.MODEL.extra_attr_loss_ratio > 0.0:
                loss_attr += self.cfg.MODEL.extra_attr_loss_ratio * aux_loss['loss_attr_ex']
            loss += loss_attr * self.cfg.MODEL.w_loss_attr
        
        if self.cfg.MODEL.use_obj_loss:
            loss_obj = (1.0 - self.cfg.MODEL.extra_obj_loss_ratio) * aux_loss['loss_obj'] 
            if self.cfg.MODEL.use_extra_obj_loss and self.cfg.MODEL.extra_obj_loss_ratio > 0.0:
                loss_obj += aux_loss['loss_obj_ex'] * self.cfg.MODEL.extra_obj_loss_ratio
            loss += loss_obj * self.cfg.MODEL.w_loss_obj
            
        if self.cfg.MODEL.use_emb_pair_loss:
            pair_emb = self.pair_final(torch.cat((aux_loss['attr_feat2'], aux_loss['obj_feat2']), 1))
            mask = aux_loss['mask']
            pred = self.classifier(pair_emb, concept_train_only)
            emb_pair_loss = F.cross_entropy(pred, pair_labels[mask])
            loss = loss + emb_pair_loss * self.cfg.MODEL.emb_loss_main
        
        ### hallucinating unseen pairs
        if self.cfg.MODEL.use_composed_pair_loss:
            unseen_concept = self.compose_word_embeddings(mode='unseen', pool_of_pairs=pool_of_pairs)

            mask_unseen = aux_loss['mask'] & (composed_unseen_pair != 2000)
            if mask_unseen.sum() > 0:
                attr_emb = aux_loss['diff_a'][mask_unseen]
                obj_emb = aux_loss['diff_o'][mask_unseen]
                pair_unseen_emb = self.pair_final(torch.cat((attr_emb, obj_emb), 1))
                pred_unseen = self.classifier(pair_unseen_emb, unseen_concept)
                composed_unseen_loss = F.cross_entropy(pred_unseen, composed_unseen_pair[mask_unseen])    
                loss = loss + composed_unseen_loss * self.cfg.MODEL.unseen_loss_ratio

            mask_seen = aux_loss['mask'] & (composed_seen_pair != 2000)
            if mask_seen.sum() > 0:
                attr_emb = aux_loss['diff_a'][mask_seen]
                obj_emb = aux_loss['diff_o'][mask_seen]
                pair_seen_emb = self.pair_final(torch.cat((attr_emb, obj_emb), 1))
                pred_seen = self.classifier(pair_seen_emb, concept_train_only)
                composed_seen_loss = F.cross_entropy(pred_seen, composed_seen_pair[mask_seen])    
                loss = loss + composed_seen_loss * self.cfg.MODEL.seen_loss_ratio
           
        out = {
            'loss_total': loss,
            'acc_attr': torch.div(correct_attr.sum(),float(bs)), 
            'acc_obj': torch.div(correct_obj.sum(),float(bs)), 
            'acc_pair': torch.div(correct_pair.sum(),float(bs)) 
        }

        if self.cfg.MODEL.use_attr_loss:
            out['loss_aux_attr'] = aux_loss['loss_attr']
            out['acc_aux_attr'] = aux_loss['acc_attr']

        if self.cfg.MODEL.use_obj_loss:
            out['loss_aux_obj'] = aux_loss['loss_obj']
            out['acc_aux_obj'] = aux_loss['acc_obj']

        if self.cfg.MODEL.use_emb_pair_loss:
            out['emb_loss'] = emb_pair_loss

        if self.cfg.MODEL.use_composed_pair_loss and mask_unseen.sum() > 0:
            out['composed_unseen_loss'] = composed_unseen_loss
        if self.cfg.MODEL.use_composed_pair_loss and mask_seen.sum() > 0:  
            out['composed_seen_loss'] = composed_seen_loss
        return out

    def val_forward(self, batch):
        img = batch['img']
        bs = img.shape[0]
        
        concept = self.compose_word_embeddings(mode='all') # [n_pairs, emb_dim].
        if self.clip_type == 'vit':
            img = self.feat_extractor.encode_image(img).float().permute(0,2,1) # Bx512x49
            img = self.img_embedder(img) #B x 2048 x 49
        elif self.clip_type == 'resnet':
            img = self.feat_extractor.encode_image(img).float() # Bx2048x7x7
            h, w = img.shape[2:]
            img = self.img_embedder(img).view(bs, -1, h*w) #B x 2048 x 49
        else:
            img = self.feat_extractor(img)[0]
            h, w = img.shape[2:]
            img = self.img_embedder(img).view(bs, -1, h*w)
        
        if self.drop_input:
            img = self.drop_inp(img)
        
       
        if self.use_fg_estimator:
            fg1 = F.softmax(self.fg_estimator(img.transpose(1, 2)), dim=1) # (bs, L)
            img = torch.matmul(img, fg1).squeeze()
        else:
            h = w = 7
            img = self.img_avg_pool(img.view(bs, -1, h, w)).squeeze()

        if self.img_emb_method == 'conv_interm':
            img = self.img_final(img)

        if self.cfg.MODEL.img_drop_last > 0:
            img = self.drop_last(img)
        
        pred = self.classifier(img, concept)
        pred = F.softmax(pred, dim=1)

        out = {}
        out['pred'] = pred
        
        out['scores'] = {}
        for _, pair in enumerate(self.all_pairs1):
            out['scores'][pair] = pred[:,self.pair2idx[pair]]

        return out

    def forward(self, x):
        if self.training:
            out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out


class ImagePairComparison(nn.Module):
    """Cross attention module to find difference/similarity between two images.
    """
    def __init__(
        self,
        cfg,
        num_attrs,
        num_objs,
        train_attrs,
        train_objs,
        extra_attrs,
        extra_objs,
        attr_embedder,
        obj_embedder,
        img_dim=300,
        emb_dim=300,
        attr_emb_dim=300,
        obj_emb_dim=300,
        word_dim=300,
        lambda_attn=10,
        attn_normalized=True,
        low_dim_cross_att=False,
        cross_att_dim=64,
        image_pair_multihead_attn=False,
    ):
        super(ImagePairComparison, self).__init__()

        self.num_attrs = num_attrs
        self.num_objs = num_objs

        self.train_attrs = train_attrs #torch.LongTensor(list(range(self.num_attrs))).cuda()
        self.train_objs = train_objs #torch.LongTensor(list(range(self.num_objs))).cuda()
        self.train_extra_attrs = extra_attrs
        self.train_extra_objs = extra_objs

        self.attr_embedder = attr_embedder
        self.obj_embedder = obj_embedder

        self.lambda_attn = lambda_attn
        self.attn_normalized = attn_normalized
        if cfg.MODEL.dropout_cross_attn > 0:
            self.dropout_cross_attn = nn.Dropout(cfg.MODEL.dropout_cross_attn)
        else:
            self.dropout_cross_attn = None
        
        self.low_dim_cross_att = low_dim_cross_att
        if low_dim_cross_att:
            self.img_proj = nn.Linear(img_dim, cross_att_dim)

        self.image_pair_multihead_attn = image_pair_multihead_attn
        if image_pair_multihead_attn:
            num_heads = cfg.MODEL.image_pair_multihead_num_heads
            self.multihead_attn = MultiheadAttention(
                inp_dim=img_dim, embed_dim=cfg.MODEL.image_pair_multihead_attn_dim,
                num_heads=num_heads,
                attn_normalized=attn_normalized, lambda_attn=lambda_attn
            )
            feat_dim = cfg.MODEL.image_pair_multihead_attn_dim * num_heads
        else:
            feat_dim = img_dim

        self.aux_loss_reweight = cfg.MODEL.aux_loss_reweight
        self.extra_attr_loss_ratio = cfg.MODEL.extra_attr_loss_ratio
        self.extra_obj_loss_ratio = cfg.MODEL.extra_obj_loss_ratio

        self.use_attr_loss = cfg.MODEL.use_attr_loss
        if self.use_attr_loss:
            self.sim_attr_embed = nn.Linear(feat_dim, attr_emb_dim)
            if cfg.MODEL.wordemb_compose_dropout > 0:
                self.attr_mlp = nn.Sequential(
                    nn.Dropout(cfg.MODEL.wordemb_compose_dropout),
                    nn.Linear(word_dim, attr_emb_dim)
                )
            else:
                self.attr_mlp = nn.Linear(word_dim, attr_emb_dim)
            self.classify_attr = CosineClassifier(cfg.MODEL.attr_cosine_cls_temp)

        self.use_obj_loss = cfg.MODEL.use_obj_loss
        if self.use_obj_loss:
            self.sim_obj_embed = nn.Linear(feat_dim, obj_emb_dim)
            if cfg.MODEL.wordemb_compose_dropout > 0:
                self.obj_mlp = nn.Sequential(
                    nn.Dropout(cfg.MODEL.wordemb_compose_dropout),
                    nn.Linear(word_dim, obj_emb_dim)
                )
            else:
                self.obj_mlp = nn.Linear(word_dim, obj_emb_dim)
            self.classify_obj = CosineClassifier(cfg.MODEL.obj_cosine_cls_temp)
            self.label_smoothing = LabelSmoothingCrossEntropy_pair(cfg.MODEL.smoothing) #SmoothCrossEntropyLoss(smoothing=cfg.MODEL.smoothing)
        
    def func_attention(self, img1, img2):
        """
        img1: (bs, d, L)
        img2: (bs, d, L)
        """
        # Get attention
        # --> (bs, L, d)
        img1T = torch.transpose(img1, 1, 2)

        # (bs, L, d)(bs, d, L)
        # --> (bs, L, L)
        if self.attn_normalized:
            relevance = torch.bmm(F.normalize(img1T, dim=2), F.normalize(img2, dim=1))
            non_relevance = -relevance
        else:
            relevance = torch.matmul(img1T, img2) / np.sqrt(2048)
        # relevance = self.relu(relevance)
        if self.dropout_cross_attn is not None:
            relevance = self.dropout_cross_attn(relevance)

        row_attn = F.softmax(relevance * self.lambda_attn, dim=2) # img1 -> img2 attention
        col_attn = F.softmax(relevance * self.lambda_attn, dim=1) # img2 -> img1 attention

        sim12 = row_attn.sum(1) # (bs, L) -> locations in img2 that are similar to many parts in img1
        sim21 = col_attn.sum(2) # (bs, L) -> locations in img1 that are similar to many parts in img2

        row_inv_attn = F.softmax(non_relevance * self.lambda_attn, dim=2)
        # row_inv_attn = 1 - row_attn
        diff12 = row_inv_attn.sum(1) # (bs, L) -> locations in img2 that differ from most parts in img1

        # Normalize to get sum = 1.
        sim12 = sim12 / (sim12.sum(1, keepdim=True) + 1e-8)
        sim21 = sim21 / (sim21.sum(1, keepdim=True) + 1e-8)
        diff12 = diff12 / (diff12.sum(1, keepdim=True) + 1e-8)
        
        return sim12, sim21, diff12

    def forward_attn(self, image1, image2, fg1=None, fg2=None):
        if self.low_dim_cross_att:
            img1 = self.img_proj(image1.transpose(1, 2)).transpose(1, 2)
            img2 = self.img_proj(image2.transpose(1, 2)).transpose(1, 2)
            sim12, sim21, diff12 = self.func_attention(img1, img2)
        else:
            sim12, sim21, diff12 = self.func_attention(image1, image2)

        # (bs, emb_dim, L) (bs, 1, L) -> (bs, emb_dim)
        sim_vec1 = (image1 * sim21.unsqueeze(1)).sum(2)
        sim_vec2 = (image2 * sim12.unsqueeze(1)).sum(2)

        # diff_vec2 = (image2 * (1.0 - sim12.unsqueeze(1))).sum(2)
        diff_vec2 = (image2 * diff12.unsqueeze(1)).sum(2)

        return sim_vec1, sim_vec2, sim21, sim12, diff_vec2

    def forward(self, img1, img2_a, img2_o, attr1, obj1, at_neigh, ob_neigh, mask_task):
        """
        """
        bs = img1.shape[0]
        
        if not self.image_pair_multihead_attn:
            sim_vec1_a, sim_vec2_a, sim21_a, sim12_a, diff_o = self.forward_attn(img1, img2_a)
            sim_vec1_o, sim_vec2_o, sim21_o, sim12_o, diff_a  = self.forward_attn(img1, img2_o)
        else:
            assert False

        mask = (mask_task == 1)

        out = {
            'mask': mask,
            'diff_a': self.sim_attr_embed(diff_a),
            'diff_o': self.sim_obj_embed(diff_o)
        }

        if self.use_attr_loss:

            attr_feat1 = self.sim_attr_embed(sim_vec1_a[mask])
            attr_emb = self.attr_embedder(self.train_attrs)
            attr_weight = self.attr_mlp(attr_emb)
            attr_pred1 = self.classify_attr(attr_feat1, attr_weight)
            
            attr_loss1 = F.cross_entropy(attr_pred1, attr1[mask])
            
            attr_pred1_ = torch.max(attr_pred1, dim=1)[1]
            attr_pred1_ = self.train_attrs[attr_pred1_]
            correct_attr1 = (attr_pred1_ == attr1[mask])


            attr_feat2 = self.sim_attr_embed(sim_vec2_a[mask])
            attr_pred2 = self.classify_attr(attr_feat2, attr_weight)
          
            attr_loss2 = F.cross_entropy(attr_pred2, attr1[mask])
            attr_pred2_ = torch.max(attr_pred2, dim=1)[1]
            attr_pred2_ = self.train_attrs[attr_pred2_]
            correct_attr2 = (attr_pred2_ == attr1[mask])

            if self.extra_attr_loss_ratio > 0.0:
                
                attr_emb1 = self.attr_embedder(self.train_extra_attrs)
                attr_weight1 = self.attr_mlp(attr_emb1)
                
                attr_pred11 = self.classify_attr(attr_feat1, attr_weight1)
              
                n11 = attr_pred11.gather(1, at_neigh['n1'].long().view(-1,1)).squeeze()
                n21 = attr_pred11.gather(1, at_neigh['n2'].long().view(-1,1)).squeeze()
                n31 = attr_pred11.gather(1, at_neigh['n3'].long().view(-1,1)).squeeze()
                n41 = attr_pred11.gather(1, at_neigh['n4'].long().view(-1,1)).squeeze()
                n51 = attr_pred11.gather(1, at_neigh['n5'].long().view(-1,1)).squeeze()
                at_pred11 = torch.cat([attr_pred1,n11.unsqueeze(1),n21.unsqueeze(1),n31.unsqueeze(1),n41.unsqueeze(1),n51.unsqueeze(1)], axis=-1)
                attr_loss_ex1 = self.label_smoothing(at_pred11, attr1[mask]) 
                
                attr_pred12 = self.classify_attr(attr_feat2, attr_weight1)
                n12 = attr_pred12.gather(1, at_neigh['n1'].long().view(-1,1)).squeeze()
                n22 = attr_pred12.gather(1, at_neigh['n2'].long().view(-1,1)).squeeze()
                n32 = attr_pred12.gather(1, at_neigh['n3'].long().view(-1,1)).squeeze()
                n42 = attr_pred12.gather(1, at_neigh['n4'].long().view(-1,1)).squeeze()
                n52 = attr_pred12.gather(1, at_neigh['n5'].long().view(-1,1)).squeeze()
                at_pred12 = torch.cat([attr_pred2,n12.unsqueeze(1),n22.unsqueeze(1),n32.unsqueeze(1),n42.unsqueeze(1),n52.unsqueeze(1)], axis=-1)
                attr_loss_ex2 = self.label_smoothing(at_pred12, attr1[mask]) 
                
                attr_loss_ex = (attr_loss_ex1 + attr_loss_ex2) / 2.0
                out['loss_attr_ex'] = attr_loss_ex

            out['loss_attr'] = (attr_loss1 + attr_loss2) / 2.0
            out['acc_attr'] = torch.div(torch.div(correct_attr1.sum().float(),mask.sum()) + \
                            torch.div(correct_attr2.sum().float(),mask.sum()), float(2))

            out['attr_feat1'] = attr_feat1
            out['attr_feat2'] = attr_feat2


        if self.use_obj_loss:
            obj_emb = self.obj_embedder(self.train_objs)
            obj_weight = self.obj_mlp(obj_emb)

            obj_feat1 = self.sim_obj_embed(sim_vec1_o[mask])
            obj_pred1 = self.classify_obj(obj_feat1, obj_weight)
           
            obj_loss1 = F.cross_entropy(obj_pred1, obj1[mask])
            obj_pred1_ = torch.max(obj_pred1, dim=1)[1]
            obj_pred1_ = self.train_objs[obj_pred1_]
            correct_obj1 = (obj_pred1_ == obj1[mask])

            obj_feat2 = self.sim_obj_embed(sim_vec2_o[mask])
            obj_pred2 = self.classify_obj(obj_feat2, obj_weight)
            
            obj_loss2 = F.cross_entropy(obj_pred2, obj1[mask])
            obj_pred2_ = torch.max(obj_pred2, dim=1)[1]
            obj_pred2_ = self.train_objs[obj_pred2_]
            correct_obj2 = (obj_pred2_ == obj1[mask])

            if self.extra_obj_loss_ratio > 0.0:
                obj_emb1 = self.obj_embedder(self.train_extra_objs)
                obj_weight1 = self.obj_mlp(obj_emb1)
                
                obj_pred11 = self.classify_obj(obj_feat1, obj_weight1)
                n11 = obj_pred11.gather(1, ob_neigh['n1'].long().view(-1,1)).squeeze()
                n12 = obj_pred11.gather(1, ob_neigh['n2'].long().view(-1,1)).squeeze()
                n13 = obj_pred11.gather(1, ob_neigh['n3'].long().view(-1,1)).squeeze()
                n14 = obj_pred11.gather(1, ob_neigh['n4'].long().view(-1,1)).squeeze()
                n15 = obj_pred11.gather(1, ob_neigh['n5'].long().view(-1,1)).squeeze()
                ob_pred11 = torch.cat([obj_pred1,n11.unsqueeze(1),n12.unsqueeze(1),n13.unsqueeze(1),n14.unsqueeze(1),n15.unsqueeze(1)], axis=-1)
                obj_loss_ex1 = self.label_smoothing(ob_pred11, obj1[mask]) 


                obj_pred12 = self.classify_obj(obj_feat2, obj_weight1)
                n11 = obj_pred12.gather(1, ob_neigh['n1'].long().view(-1,1)).squeeze()
                n12 = obj_pred12.gather(1, ob_neigh['n2'].long().view(-1,1)).squeeze()
                n13 = obj_pred12.gather(1, ob_neigh['n3'].long().view(-1,1)).squeeze()
                n14 = obj_pred12.gather(1, ob_neigh['n4'].long().view(-1,1)).squeeze()
                n15 = obj_pred12.gather(1, ob_neigh['n5'].long().view(-1,1)).squeeze()
                ob_pred12 = torch.cat([obj_pred2,n11.unsqueeze(1),n12.unsqueeze(1),n13.unsqueeze(1),n14.unsqueeze(1),n15.unsqueeze(1)], axis=-1)
                obj_loss_ex2 = self.label_smoothing(ob_pred12, obj1[mask]) 
                
                obj_loss_ex = (obj_loss_ex1 + obj_loss_ex2) / 2.0
                out['loss_obj_ex'] = obj_loss_ex

            out['loss_obj'] = (obj_loss1 + obj_loss2) / 2.0
            out['acc_obj'] = torch.div(torch.div(correct_obj1.sum().float(),mask.sum()) + \
                              torch.div(correct_obj2.sum().float(),mask.sum()),float(2))

            out['obj_feat1'] = obj_feat1
            out['obj_feat2'] = obj_feat2

        return out


class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = (img_norm.unsqueeze(1) * concept_norm).sum(-1) / self.temp # (bs, n_class)
        return pred


class MultiheadAttention(nn.Module):
    def __init__(self, inp_dim=2048, embed_dim=512, num_heads=1,
                 attn_normalized=True, lambda_attn=10):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_normalized = attn_normalized
        self.lambda_attn = lambda_attn

        f_query = []
        for _ in range(num_heads):
            f_query += [nn.Linear(inp_dim, embed_dim)]
        self.f_query = nn.ModuleList(f_query)

    def forward(self, img1, img2):
        img1 = img1.transpose(1, 2) # (bs, L, d)
        img2 = img2.transpose(1, 2) # (bs, L, d)

        out = {
            'sim_in_img1': [],
            'sim_in_img2': [],
            'img1': [],
            'img2': []
        }

        for i in range(self.num_heads):
            img1_query = self.f_query[i](img1)
            img2_query = self.f_query[i](img2)
            sim_in_img1, sim_in_img2 = self.func_attention(img1_query, img2_query)
            out['sim_in_img1'].append(sim_in_img1)
            out['sim_in_img2'].append(sim_in_img2)
            out['img1'].append(img1_query)
            out['img2'].append(img2_query)
        
        return out

    def func_attention(self, img1, img2):
        """
        img1: (bs, L, d)
        img2: (bs, L, d)
        """
        # Get attention
        # (bs, L, d)(bs, d, L)
        # --> (bs, L, L)
        if self.attn_normalized:
            relevance = torch.bmm(F.normalize(img1, dim=2), F.normalize(img2.transpose(1, 2), dim=1))
        else:
            relevance = torch.matmul(img1, img2.transpose(1, 2)) / np.sqrt(2048)

        row_attn = F.softmax(relevance * self.lambda_attn, dim=2)
        col_attn = F.softmax(relevance * self.lambda_attn, dim=1)

        sim12 = row_attn.sum(1) # (bs, L) -> locations in img2 that are similar to many parts in img1
        sim21 = col_attn.sum(2) # (bs, L) -> locations in img1 that are similar to many parts in img2

        sim12 = sim12 / (sim12.sum(1, keepdim=True) + 1e-8)
        sim21 = sim21 / (sim21.sum(1, keepdim=True) + 1e-8)
        
        return sim21, sim12