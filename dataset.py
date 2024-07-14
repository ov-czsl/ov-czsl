import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.models as tmodels
import torchvision.transforms as transforms
import tqdm
import pickle
from PIL import Image
import pdb
import random

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = f'{self.img_dir}/{img}'
        img = Image.open(file).convert('RGB')
        return img


def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
       
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


def imagenet_transform_zappos(phase, cfg):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


def process_neighbors_list(dict1):
    list1 = []
    for k in dict1.keys():
        l2 = list(dict1[k].keys())
        list1.extend(l2)
    list2 = list(set(list1))
    list2.sort()
    return list2

class CompositionDataset(tdata.Dataset):
    def __init__(
        self,
        phase,
        split='compositional-split',
        open_world=False,
        cfg=None
    ):
        self.phase = phase
        self.cfg = cfg
        self.split = split
        self.open_world = open_world
        self.split_files = cfg.DATASET.split_files_loc

        if 'ut-zap50k' in cfg.DATASET.name:
            self.transform = imagenet_transform_zappos(phase, cfg)
        else:
            self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(f'{cfg.DATASET.root_dir}/images')

        self.attrs, self.objs, self.pairs, \
            self.train_pairs, self.test_pairs, self.val_pairs, \
            self.train_attrs, self.train_objs, \
            self.test_attrs, self.test_objs,\
            self.val_attrs, self.val_objs = self.parse_split()

        self.train_data, self.test_data, self.val_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.train_pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}
 
        only_test = list(set(self.train_attrs) - set(self.train_attrs))
       
        self.all_attrs = self.train_attrs + self.val_attrs + self.test_attrs 
        self.all_objs = self.train_objs + self.val_objs + self.test_objs

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.all_objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.all_attrs)}


        print('# train pairs: %d | # val pairs: %d| # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d|# test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.sample_indices = list(range(len(self.data)))
        
        ### setup for new losses for clip feature loss and neighbor loss
        if self.cfg.DATASET.dset_name == 'mit':
            self.replace_attr = pickle.load(open(self.cfg.DATASET.split_files_loc+'/Replace_attr_mit_1.pkl','rb'))
            self.replace_obj = pickle.load(open(self.cfg.DATASET.split_files_loc+'/Replace_obj_mit_1.pkl','rb'))  
        elif self.cfg.DATASET.dset_name == 'vaw':
            self.replace_attr = {}
            self.replace_obj = {}
        elif self.cfg.DATASET.dset_name == 'cgqa': 
            self.replace_attr = pickle.load(open(self.cfg.DATASET.split_files_loc+'/Replace_attr_cgqa_1.pkl','rb'))
            self.replace_obj = pickle.load(open(self.cfg.DATASET.split_files_loc+'/Replace_obj_cgqa_1.pkl','rb'))
            self.replace_attr['blank'] = 'dark'
            self.replace_attr['rolled'] = 'raw'
            self.replace_attr['crouched'] = 'ugly'
        else:
            print('Enter a valid dataset name: some files are missing for: ',self.cfg.DATASET.dset_name)
        self.seen_pairs_neighbors = pickle.load(open(self.cfg.DATASET.split_files_loc+'/Train_seen_pairs_neighbors_'+self.cfg.DATASET.dset_name+'_'+str(self.cfg.DATASET.dset_split)+'.pkl','rb'))
        self.seen_attr_neighbors = pickle.load(open(self.cfg.DATASET.split_files_loc+'/Neighbor_attr_'+self.cfg.DATASET.dset_name+'_'+str(self.cfg.DATASET.dset_split)+'.pkl','rb'))
        self.seen_obj_neighbors = pickle.load(open(self.cfg.DATASET.split_files_loc+'/Neighbor_obj_'+self.cfg.DATASET.dset_name+'_'+str(self.cfg.DATASET.dset_split)+'.pkl','rb'))

        self.extra_pairs = process_neighbors_list(self.seen_pairs_neighbors) 
        self.extra_attrs = process_neighbors_list(self.seen_attr_neighbors) 
        self.extra_objs = process_neighbors_list(self.seen_obj_neighbors) 
        self.extra_objs = list(set(self.extra_objs) - set(self.train_objs))
        self.extra_attrs = list(set(self.extra_attrs) - set(self.train_attrs))
        self.extra_pairs = list(set(self.extra_pairs) - set(self.train_pairs))
        print('New_attrs:',len(self.extra_attrs),' New objs:',len(self.extra_objs),' New pairs:',len(self.extra_pairs))
       
        self.unique_pairs =  self.pairs + self.extra_pairs
        self.unique_pair2idx = {pair: idx for idx, pair in enumerate(self.unique_pairs)}
        
        self.unique_attrs =  self.attrs + self.extra_attrs
        self.unique_attr2idx = {attr: idx for idx, attr in enumerate(self.unique_attrs)}
        
        self.unique_objs =  self.objs + self.extra_objs 
        self.unique_obj2idx = {obj: idx for idx, obj in enumerate(self.unique_objs)}

        self.train_pairs_extra =  self.train_pairs + self.extra_pairs 
        self.train_extra_pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs_extra)}

        self.train_attrs_extra =  self.train_attrs + self.extra_attrs 
        self.train_extra_attr2idx = {attr: idx for idx, attr in enumerate(self.train_attrs_extra)}

        self.train_objs_extra =  self.train_objs + self.extra_objs 
        self.train_extra_obj2idx = {obj: idx for idx, obj in enumerate(self.train_objs_extra)}
        
        if cfg.TRAIN.use_precomputed_features:
            if cfg.MODEL.name not in ['oaclip', 'oaclipv2']:
                feat_file = f'{cfg.DATASET.root_dir}/features.t7'
                feat_avgpool = True
            else:
                feat_file = f'{cfg.DATASET.root_dir}/features_b4avgpool.t7'
                feat_avgpool = False
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file, feat_avgpool)

            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)

            print('%d activations loaded' % (len(self.activations)))

        # Affordance.
        self.obj_affordance = {} # -> contains attributes compatible with an object.
        self.attr_affordance = {} # -> contains objects compatible with an attribute.
        for _obj in self.objs:
            candidates = [
                attr
                for (_, attr, obj) in self.train_data
                if obj == _obj
            ]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))
            
        for _attr in self.attrs:
            candidates = [
                obj
                for (_, attr, obj) in self.train_data
                if attr == _attr
            ]
            self.attr_affordance[_attr] = sorted(list(set(candidates)))
            
        # Images that contain an object.
        self.image_with_obj = {}
        for i, instance in enumerate(self.train_data):
            obj = instance[2]
            if obj not in self.image_with_obj:
                self.image_with_obj[obj] = []
            self.image_with_obj[obj].append(i)
        
        # Images that contain an attribute.
        self.image_with_attr = {}
        for i, instance in enumerate(self.train_data):
            attr = instance[1]
            if attr not in self.image_with_attr:
                self.image_with_attr[attr] = []
            self.image_with_attr[attr].append(i)

        # Images that contain a pair.
        self.image_with_pair = {}
        for i, instance in enumerate(self.train_data):
            attr, obj = instance[1], instance[2]
            if (attr, obj) not in self.image_with_pair:
                self.image_with_pair[(attr, obj)] = []
            self.image_with_pair[(attr, obj)].append(i)

        if cfg.MODEL.use_composed_pair_loss:
            with open(self.cfg.DATASET.split_files_loc+'/unseen_pairs_ov_'+cfg.DATASET.dset_name+'_'+str(cfg.DATASET.dset_split)+'.txt', 'r') as f:
                self.unseen_pairs = [tuple(l.strip().split()) for l in f.readlines()]
            self.unseen_pair2idx = {pair: idx for idx, pair in enumerate(self.unseen_pairs)}

        self.if_set_seed = False

    def get_split_info(self):
        path_metadata = f'{self.cfg.DATASET.split_files_loc}/metadata_open_vocab_'+self.cfg.DATASET.dset_name+'.t7'
        data = torch.load(path_metadata)
        
        train_data, test_data, val_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                instance['obj'], instance['set']
            curr_data = [image, attr, obj]

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val': # and (attr,obj) in self.val_pairs:
                val_data.append(curr_data)
            elif settype == 'test': # and (attr,obj) in self.test_pairs:
                test_data.append(curr_data)

        return train_data, test_data, val_data
        

    def parse_split(self):
        def parse_pairs_pkl(split):
            split_path = self.cfg.DATASET.split_files_loc+'/'+(self.cfg.DATASET.dset_name).upper()+'_splits.pkl'
            data = pickle.load(open(split_path,'rb'))
            
            if split == 'train':
                l1 = data['sa_so_tr']
            elif split == 'val':
                l1 = data['ua_uo_va']
                l1.extend(data['sa_so_va'])
                l1.extend(data['sa_uo_va'])
                l1.extend(data['ua_so_va'])
                l1.extend(data['sa_so_unseen_comp_va'])
            elif split == 'test':
                l1 = data['ua_uo_te']
                l1.extend(data['sa_so_te'])
                l1.extend(data['sa_uo_te'])
                l1.extend(data['ua_so_te'])
                l1.extend(data['sa_so_unseen_comp_te'])
           
            pairs = list(map(tuple, l1))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs_pkl('train')
        ts_attrs, ts_objs, ts_pairs = parse_pairs_pkl('test')
        va_attrs, va_objs, va_pairs = parse_pairs_pkl('val')

       
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + va_attrs + ts_attrs))), sorted(
                list(set(tr_objs + va_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + va_pairs + ts_pairs)))

       
        va_pairs, va_attrs, va_objs = sorted(list(set(va_pairs))),  sorted(list(set(va_attrs))),sorted(list(set(va_objs)))
        tr_pairs, tr_attrs, tr_objs = sorted(list(set(tr_pairs))),  sorted(list(set(tr_attrs))), sorted(list(set(tr_objs)))
        te_pairs, te_attrs, te_objs = sorted(list(set(ts_pairs))),  sorted(list(set(ts_attrs))), sorted(list(set(ts_objs)))

        only_test_attrs = list(set(ts_attrs) - set(tr_attrs))
        only_test_objs = list(set(ts_objs) - set(tr_objs))

        only_val_attrs = list(set(va_attrs) - (set(tr_attrs + te_attrs)))
        only_val_objs = list(set(va_objs) - set(tr_objs + te_objs))
        # pdb.set_trace()
        return all_attrs, all_objs, all_pairs, tr_pairs, te_pairs, va_pairs, tr_attrs, tr_objs, only_test_attrs,\
            only_test_objs, only_val_attrs, only_val_objs
        
    def __getitem__(self, index):
        if hasattr(self, 'if_set_seed') and not self.if_set_seed:
            # Trick to set different numpy random seeds for each worker thread.
            np.random.seed(index)
            self.if_set_seed = True

        index = self.sample_indices[index]
        image, attr, obj = self.data[index]
        if self.cfg.TRAIN.use_precomputed_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)

        if self.phase == 'train':
            if hasattr(self.cfg.TRAIN, 'sample_negative_pairs') and \
                    self.cfg.TRAIN.sample_negative_pairs != -1:
                # If use custom set of negative pairs during training.
                num_negs = self.cfg.TRAIN.sample_negative_pairs - 1
                pair_idx = self.train_pair2idx[(attr, obj)]

                if self.cfg.TRAIN.n_sample_hard_negative > 0:
                    # If use hard negative.
                    # Hard negative pair: share either attr/obj with groundtruth.
                    afforded_attrs = self.obj_affordance[obj]
                    afforded_objs = self.attr_affordance[attr]
                    hard_negs = []
                    for a in afforded_attrs:
                        for o in afforded_objs:
                            if (a, o) not in self.train_pair2idx:
                                continue
                            neg_pair_idx = self.train_pair2idx[(a, o)]
                            hard_negs.append(neg_pair_idx)
                    hard_negs = np.random.choice(
                        hard_negs,
                        size=min(self.cfg.TRAIN.n_sample_hard_negative, len(hard_negs)),
                        replace=False)
                    n_remains = num_negs - len(hard_negs)
                else:
                    hard_negs = []
                    n_remains = num_negs

                if n_remains > 0:
                    # Randomly sample negative pairs.
                    neg_candidates = [
                        x
                        for x in list(range(pair_idx)) + list(range(pair_idx+1, len(self.train_pairs)))
                        if x not in hard_negs
                    ]
                    random_neg_candidates = np.random.choice(
                        neg_candidates,
                        size=n_remains,
                        replace=False
                    )
                else:
                    random_neg_candidates = []
                assert len(hard_negs) + len(random_neg_candidates) == num_negs, \
                    f"hard: {len(hard_negs)}, random: {len(random_neg_candidates)}"

                # Explicitly put groundtruth pair at index 0, the rest after.
                pool_of_pairs = np.concatenate(([pair_idx], hard_negs, random_neg_candidates))
                pool_of_pairs = pool_of_pairs.astype(np.int64)

            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.train_pair2idx[(attr, obj)],
                'img_name': self.data[index][0],
            }
            if self.cfg.MODEL.use_extra_pair_loss:
                L, At, Ob = self.sample_neighbor_features(attr,obj)
                data['lbl1'] = L[0]
                data['lbl2'] = L[1]
                data['lbl3'] = L[2]
                data['lbl4'] = L[3]
                data['lbl5'] = L[4]
                data['extra_lbl'] = L

                data['at1'] = At[0]
                data['at2'] = At[1]
                data['at3'] = At[2]
                data['at4'] = At[3]
                data['at5'] = At[4]

                data['ob1'] = Ob[0]
                data['ob2'] = Ob[1]
                data['ob3'] = Ob[2]
                data['ob4'] = Ob[3]
                data['ob5'] = Ob[4]

            if self.cfg.TRAIN.sample_negative_pairs != -1:
                data['pool_of_pairs'] = pool_of_pairs

            model_name = self.cfg.MODEL.name
           
            if model_name == 'oaclipv3':
                # Sample pair of images.
                data['mask_task'] = 1 # Attribute task
                
                attr_rep, i2 = self.sample_same_attribute(attr, obj, self.replace_attr,  with_different_obj=True)
                if i2 == -1:
                    data['mask_task'] = 0
                    print(attr, obj, attr_rep,i2)
                    pdb.set_trace()

                img1, attr1, obj1_a = self.data[i2]
                if attr_rep:
                    attr1 = attr_rep
            
                if self.cfg.TRAIN.use_precomputed_features:
                    img1 = self.activations[img1]
                else:
                    img1 = self.loader(img1)
                    img1 = self.transform(img1)

                data['img1_a'] = img1
                data['attr1_a'] = self.attr2idx[attr1]
                data['obj1_a'] = self.obj2idx[obj1_a]
                data['idx1_a'] = i2
                data['img1_name_a'] = self.data[i2][0]

                 # Object task.
                obj_rep, i2 = self.sample_same_object(attr, obj, self.replace_obj, with_different_attr=True)
                img1, attr1_o, obj1 = self.data[i2]
                if obj_rep:
                    obj1 = obj_rep
                # print('OBJ:',attr1_o, obj1, img1)

                if self.cfg.TRAIN.use_precomputed_features:
                    img1 = self.activations[img1]
                else:
                    img1 = self.loader(img1)
                    img1 = self.transform(img1)
                data['img1_o'] = img1
                data['attr1_o'] = self.attr2idx[attr1_o]
                data['obj1_o'] = self.obj2idx[obj1]
                data['idx1_o'] = i2
                data['img1_name_o'] = self.data[i2][0]

                if self.cfg.MODEL.use_composed_pair_loss:
                    if (attr1_o, obj1_a) in self.unseen_pair2idx:
                        data['composed_unseen_pair'] = self.unseen_pair2idx[(attr1_o, obj1_a)]
                        data['composed_seen_pair'] = 2000
                    elif (attr1_o, obj1_a) in self.train_pair2idx:
                        data['composed_seen_pair'] = self.train_pair2idx[(attr1_o, obj1_a)]
                        data['composed_unseen_pair'] = 2000
                    else:
                        data['composed_unseen_pair'] = 2000
                        data['composed_seen_pair'] = 2000
                
        else:
            # Testing mode.
            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.pair2idx[(attr, obj)],
                'img_name': self.data[index][0],
            }
        return data

    def __len__(self):
        return len(self.sample_indices)

    def sample_neighbor_features(self,attr,obj):
        list_neigh = list(self.seen_pairs_neighbors[(attr,obj)].keys())

        new = random.sample(list_neigh,5)
        L = [self.train_extra_pair2idx[k] for k in new]
        #### attrs neighbors
        neigh_attr = list(self.seen_attr_neighbors[attr].keys())
        new_attr = random.sample(neigh_attr,5)
        AT = [self.train_extra_attr2idx[k] for k in new_attr]

        #### objs neighbors
        neigh_obj = list(self.seen_obj_neighbors[obj].keys())
        new_obj = random.sample(neigh_obj,5)
        OB = [self.train_extra_obj2idx[k] for k in new_obj]

        return L, AT, OB
    
    def sample_same_attribute(self, attr, obj, replace_attr, with_different_obj=True):
        attr_rep = None
        if with_different_obj:
            if attr in replace_attr.keys():
                attr = replace_attr[attr]
                attr_rep = attr
            if len(self.attr_affordance[attr]) == 1:
                return attr_rep, -1

            pair = (attr, obj)
            
            if not self.cfg.TRAIN.sample_similar_object or \
                    pair not in self.oaclip_same_attr_images or \
                    len(self.oaclip_same_attr_images[pair]) == 0:
                i2 = np.random.choice(self.image_with_attr[attr])
                img1, attr1, obj1 = self.data[i2]
                while obj1 == obj:
                    i2 = np.random.choice(self.image_with_attr[attr])
                    img1, attr1, obj1 = self.data[i2]
            else:
                i2 = np.random.choice(self.oaclip_same_attr_images[pair])
                img1, attr1, obj1 = self.data[i2]
            assert obj1 != obj
        else:
            i2 = np.random.choice(self.image_with_attr[attr])
        return attr_rep, i2

    def sample_same_object(self, attr, obj, replace_objs, with_different_attr=True):
        obj_rep = None
        if obj in replace_objs.keys():
            obj = replace_objs[obj]
            obj_rep = obj
        i2 = np.random.choice(self.image_with_obj[obj])
        if with_different_attr:
            img1, attr1, obj1 = self.data[i2]
            while attr1 == attr:
                i2 = np.random.choice(self.image_with_obj[obj])
                img1, attr1, obj1 = self.data[i2]
        return obj_rep, i2

    def generate_features(self, out_file, feat_avgpool=True):
        data = self.train_data + self.val_data + self.test_data
        transform = imagenet_transform('test')
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(
                chunks(data, 512), total=len(data) // 512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            imgs = torch.stack(imgs, 0).cuda()
            if feat_avgpool:
                feats = feat_extractor(imgs)
            else:
                feats = feat_extractor.conv1(imgs)
                feats = feat_extractor.bn1(feats)
                feats = feat_extractor.relu(feats)
                feats = feat_extractor.maxpool(feats)
                feats = feat_extractor.layer1(feats)
                feats = feat_extractor.layer2(feats)
                feats = feat_extractor.layer3(feats)
                feats = feat_extractor.layer4(feats)
                assert feats.shape[-3:] == (512, 7, 7), feats.shape
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))
        torch.save({'features': image_feats, 'files': image_files}, out_file)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]