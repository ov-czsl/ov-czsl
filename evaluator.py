import torch
import numpy as np
from scipy.stats import hmean
import pdb
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Evaluator:

    def __init__(self, dset, cfg=None):

        self.dset = dset
        self.cfg = cfg
        path_split = cfg.DATASET.split_files_loc 
        if self.cfg.DATASET.dset_name == 'mit':
            d_splits = pickle.load(open(path_split+'/MIT_splits.pkl','rb'))
        elif self.cfg.DATASET.dset_name == 'vaw':
            d_splits = pickle.load(open(path_split+'/VAW_splits.pkl','rb'))
        elif self.cfg.DATASET.dset_name == 'cgqa':
            d_splits = pickle.load(open(path_split+'/CGQA_splits.pkl','rb'))
        
        
        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)
        if dset.phase == 'val':
            self.sa_so = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['sa_so_va']]
            self.ua_so = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['ua_so_va']]
            self.sa_uo = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['sa_uo_va']]
            self.ua_uo = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['ua_uo_va']]
            self.sa_so_u = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['sa_so_unseen_comp_va']]
        elif dset.phase == 'test':
            self.sa_so = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['sa_so_te']]
            self.ua_so = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['ua_so_te']]
            self.sa_uo = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['sa_uo_te']]
            self.ua_uo = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['ua_uo_te']]
            self.sa_so_u = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in d_splits['sa_so_unseen_comp_te']]
        self.s_a = [dset.attr2idx[a] for a in d_splits['s_a']]
        self.u_a = [dset.attr2idx[a] for a in d_splits['u_a']]
        self.s_o = [dset.obj2idx[a] for a in d_splits['s_o']]
        self.u_o = [dset.obj2idx[a] for a in d_splits['u_o']]
        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr,obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set  else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1  if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias = 0.0, topk = 5): # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''
        def get_pred_from_scores(_scores, topk):
            '''
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            '''
            _, pair_pred = _scores.topk(topk, dim = 1) #sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        scores[~mask] += bias # Add bias to test pairs

        # Unbiased setting
        
        # Open world setting --no mask, all pairs of the dataset
        results.update({'open': get_pred_from_scores(scores, topk)})
        results.update({'unbiased_open': get_pred_from_scores(orig_scores, topk)})
        # Closed world setting - set the score for all Non test pairs to -1e10, 
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10 
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({'closed': get_pred_from_scores(closed_scores, topk)})
        results.update({'unbiased_closed': get_pred_from_scores(closed_orig_scores, topk)})

        # Object_oracle setting - set the score to -1e10 for all pairs where the true object does Not participate, can also use the closed score
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        oracle_obj_scores[~mask] = -1e10
        oracle_obj_scores_unbiased = orig_scores.clone()
        oracle_obj_scores_unbiased[~mask] = -1e10
        results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores, 1)})
        results.update({'object_oracle_unbiased': get_pred_from_scores(oracle_obj_scores_unbiased, 1)})

        return results

    def score_clf_model(self, scores, obj_truth, topk = 5):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to('cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        attr_subset = attr_pred.index_select(1, self.pairs[:,0]) # Return only attributes that are in our pairs
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset) # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias = 0.0, topk = 5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr,obj)] for attr, obj in self.dset.pairs], 1
        ) # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias = 0.0, topk = 5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        
        results = {}
        mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
        scores[~mask] += bias # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10 

        _, pair_pred = closed_scores.topk(topk, dim = 1) #sort returns indices of k largest values

        pair_pred = pair_pred.contiguous().view(-1)

        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results
    
    def accuracy_attr_obj(self, output, target, type, topk=(1,)):
        """
        Computes the precision@k for the specified values of k
        """
        with torch.no_grad():
            maxk = max(topk)  # max number labels we will consider in the right choices for out model
            batch_size = target.size(0)

            _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
            y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
            target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
            # attr_pred, obj_pred = self.pairs[y_pred][:, 0].view(-1, topk), self.pairs[y_pred][:, 1].view(-1, topk)
            
            if type == 'attr':
                y_pred = self.pairs[y_pred.squeeze()][:, 0]
            else:
                y_pred = self.pairs[y_pred.squeeze()][:, 1]
            # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
            correct = (y_pred == target)  # [maxk, B] were for each example we know which topk prediction matched truth
            
            # -- get topk accuracy
            list_topk_accs = correct.sum()/correct.shape[0] #[]  # idx is topk1, topk2, ... etc
            # for k in topk:
            #     # get tensor of which topk answer was right
            #     ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            #     # flatten it to help compute if we got it correct for each example in batch
            #     flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            #     # get if we got it right for any of our top k prediction for each example in batch
            #     tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            #     # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            #     # pdb.set_trace()
            #     topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            #     list_topk_accs.append(topk_acc)
            return list_topk_accs 
    
    def accuracy(self, output, target, topk=(1,)):
        """
        Computes the precision@k for the specified values of k
        """
        with torch.no_grad():
            maxk = max(topk)  # max number labels we will consider in the right choices for out model
            batch_size = target.size(0)

            _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
            y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
            target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
            # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
            correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
            
            # -- get topk accuracy
            list_topk_accs = []  # idx is topk1, topk2, ... etc
            for k in topk:
                # get tensor of which topk answer was right
                ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
                # flatten it to help compute if we got it correct for each example in batch
                flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
                # get if we got it right for any of our top k prediction for each example in batch
                tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
                # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
                topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
                list_topk_accs.append(topk_acc)
            return list_topk_accs 

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk = 1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')

        pairs = list(
            zip(list(attr_truth.numpy()), list(obj_truth.numpy())))
        
        seen_ind, unseen_ind = [], []
        sa_so, ua_uo, sa_uo, ua_so, sa_so_u = [], [], [], [], []
        u_a, u_o, s_a, s_o = [], [], [], []
        for i in range(len(attr_truth)):
            if pairs[i][0] in self.s_a:
                s_a.append(i)
            else:
                u_a.append(i)
            if pairs[i][1] in self.s_o:
                s_o.append(i)
            else:
                u_o.append(i)

            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)
            if pairs[i] in self.sa_so:
                sa_so.append(i)
            elif pairs[i] in self.ua_uo:
                ua_uo.append(i)
            elif pairs[i] in self.ua_so:
                ua_so.append(i)
            elif pairs[i] in self.sa_uo:
                sa_uo.append(i)
            elif pairs[i] in self.sa_so_u:
                sa_so_u.append(i)

        
        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)
        sa_so = torch.LongTensor(sa_so)
        sa_so_u = torch.LongTensor(sa_so_u)
        ua_so = torch.LongTensor(ua_so)
        sa_uo = torch.LongTensor(sa_uo)
        ua_uo = torch.LongTensor(ua_uo)

        s_a = torch.LongTensor(s_a)
        u_a = torch.LongTensor(u_a)
        s_o = torch.LongTensor(s_o)
        u_o = torch.LongTensor(u_o)

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
            obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])
            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            seen_score, unseen_score = torch.ones(512,5), torch.ones(512,5)

            return attr_match, obj_match, match, seen_match, unseen_match, \
            torch.Tensor(seen_score+unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = ['_attr_match', '_obj_match', '_match', '_seen_match', '_unseen_match', '_ca', '_seen_ca', '_unseen_ca']
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        ##################### Match in places where corrent object
        obj_oracle_match = (attr_truth == predictions['object_oracle'][0][:, 0]).float()  #object is already conditioned
        obj_oracle_match_unbiased = (attr_truth == predictions['object_oracle_unbiased'][0][:, 0]).float()

        stats = dict(obj_oracle_match = obj_oracle_match, obj_oracle_match_unbiased = obj_oracle_match_unbiased)

        #################### Closed world
        closed_scores = _process(predictions['closed'])
        unbiased_closed = _process(predictions['unbiased_closed'])
        attr_match1 = (attr_truth[s_a].unsqueeze(1).repeat(1, topk) == predictions['closed'][0][s_a][:, :topk])
        attr_match1 = attr_match1.any(1).float()
        _add_to_dict(closed_scores, 'closed', stats)
        _add_to_dict(unbiased_closed, 'closed_ub', stats)

        #################### Calculating AUC
        scores = predictions['scores']
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][unseen_ind]
      
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats['closed_unseen_match'].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats['closed_seen_match'].mean())
        unseen_match_max = float(stats['closed_unseen_match'].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to('cpu') for k, v in allpred.items()}
        obj_truth = obj_truth.to('cpu')

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr,obj)] for attr, obj in self.dset.pairs], 1
        ) # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(scores, obj_truth, bias = bias, topk = topk)
            results = results['closed'] # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis = 0)
        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
            print('For final bias:', bias_term)
            scores = base_scores.clone()
            mask = self.seen_mask.repeat(scores.shape[0],1) # Repeat mask along pairs dimension
            scores[~mask] += bias_term # Add bias to test pairs
            mask = self.closed_mask.repeat(scores.shape[0], 1)
            closed_scores = scores.clone()
            closed_scores[~mask] = -1e10
            scores = closed_scores
            sa_so_acc = self.accuracy(scores[sa_so].detach().cpu(), pair_truth[sa_so],[1, 3, 5, 10]) 
            ua_so_acc = self.accuracy(scores[ua_so].detach().cpu(), pair_truth[ua_so],[1, 3, 5, 10])
            sa_uo_acc = self.accuracy(scores[sa_uo].detach().cpu(), pair_truth[sa_uo],[1, 3, 5, 10])
            ua_uo_acc = self.accuracy(scores[ua_uo].detach().cpu(), pair_truth[ua_uo],[1, 3, 5, 10])
            sa_so_u_acc = self.accuracy(scores[sa_so_u].detach().cpu(), pair_truth[sa_so_u],[1, 3, 5, 10])
            print('Topk [1,3,5]')
            print('Seen pairs acc:',sa_so_acc)
            print('Seen attr-obj, unseen comp acc:',sa_so_u_acc)
            print('Unseen attr, Seen obj pairs:',ua_so_acc)
            print('Seen attr, Unseen obj pairs:',sa_uo_acc)
            print('Unseen attr, Unseen obj pairs:',ua_uo_acc)
            s_a_acc = self.accuracy_attr_obj(scores[s_a], attr_truth[s_a],'attr',[1])
            u_a_acc = self.accuracy_attr_obj(scores[u_a], attr_truth[u_a],'attr',[1])
            s_o_acc = self.accuracy_attr_obj(scores[s_o], obj_truth[s_o],'obj',[1])
            u_o_acc = self.accuracy_attr_obj(scores[u_o], obj_truth[u_o],'obj',[1])
            print('Seen-unseen attr acc:', s_a_acc, u_a_acc)
            print('Seen-Unseen obj acc:',s_o_acc, u_o_acc)
            print('Seen_acc_at_bias:', seen_accuracy[idx], '| Unseen_acc_at_bias:', unseen_accuracy[idx])
        stats['biasterm'] = float(bias_term)
        stats['best_unseen'] = np.max(unseen_accuracy)
        stats['best_seen'] = np.max(seen_accuracy)
        stats['AUC'] = area
        stats['hm_unseen'] = unseen_accuracy[idx]
        stats['hm_seen'] = seen_accuracy[idx]
        stats['best_hm'] = max_hm
        stats['sa_so_acc'] = sa_so_acc[0].tolist()[0]
        stats['sa_so_u_acc'] = sa_so_u_acc[0].tolist()[0]
        stats['sa_uo_acc'] = sa_uo_acc[0].tolist()[0]
        stats['ua_so_acc'] = ua_so_acc[0].tolist()[0]
        stats['ua_uo_acc'] = ua_uo_acc[0].tolist()[0]
        stats['s_a_acc'] = s_a_acc.tolist()
        stats['u_a_acc'] = u_a_acc.tolist()
        stats['s_o_acc'] = s_o_acc.tolist()
        stats['u_o_acc'] = u_o_acc.tolist()

        return stats