import torch
import numpy as np
import fasttext.util

def load_bert_embeddings(wordemb,vocab,cfg):
    '''
    Inputs
        emb_file: Text file with word embedding pairs e.g. Glove, Processed in lower case.
        vocab: List of words
    Returns
        Embedding Matrix
    '''
   
    if 'bert' in wordemb:
        path_emb = cfg.DATASET.split_files_loc
        m = torch.load(path_emb+'/ALL_feats_'+cfg.DATASET.dset_name+'.pt')
        m1 = torch.load(path_emb+'/Neighbor_bert_text_features_prompt1_'+cfg.DATASET.dset_name+'.pt')
    else:
        print('enter valid dataset name')
    
    if '_obj' in wordemb:
        model = m['objs']
        model2 = m1['objs']
    elif '_attr' in wordemb:
        model = m['attrs']
        model2 = m1['attrs']
    else:
        model = m['pairs']
        model2 = m1['pairs']
    # Adding some vectors for UT Zappos
    custom_map = {
        'faux.fur': 'fake_fur',
        'faux.leather': 'fake_leather',
        'full.grain.leather': 'thick_leather',
        'hair.calf': 'hair_leather',
        'patent.leather': 'shiny_leather',
        'boots.ankle': 'ankle_boots',
        'boots.knee.high': 'knee_high_boots',
        'boots.mid-calf': 'midcalf_boots',
        'shoes.boat.shoes': 'boat_shoes',
        'shoes.clogs.and.mules': 'clogs_shoes',
        'shoes.flats': 'flats_shoes',
        'shoes.heels': 'heels',
        'shoes.loafers': 'loafers',
        'shoes.oxfords': 'oxford_shoes',
        'shoes.sneakers.and.athletic.shoes': 'sneakers',
        'traffic_light': 'traffic_light',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower',
        'nubuck' : 'grainy_leather',
    }
    # custom_map2 = {'abandoned':'left'}
    embeds = []
    for k in vocab:
        if k in model.keys():
            emb = model[k].cpu()
        else:
            emb = model2[k].cpu()
        embeds.append(emb)
    embeds = torch.stack(embeds)
    print('BERT Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds

def load_word_embeddings(emb_file, vocab):
    vocab = [v.lower() for v in vocab]
    embeds = {}
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        embeds[line[0]] = wvec
    
    custom_map = {
        'Faux.Fur':'fur', 'Faux.Leather':'leather', 'Full.grain.leather':'leather', 
        'Hair.Calf':'hair', 'Patent.Leather':'leather', 'Nubuck':'leather', 
        'Boots.Ankle':'boots', 'Boots.Knee.High':'knee-high', 'Boots.Mid-Calf':'midcalf', 
        'Shoes.Boat.Shoes':'shoes', 'Shoes.Clogs.and.Mules':'clogs', 'Shoes.Flats':'flats',
        'Shoes.Heels':'heels', 'Shoes.Loafers':'loafers', 'Shoes.Oxfords':'oxfords',
        'Shoes.Sneakers.and.Athletic.Shoes':'sneakers'}
    for k in custom_map:
        embeds[k.lower()] = embeds[custom_map[k]]

    custom_map2 = {'selfie':'photo'}

    embs = []
    for k in vocab:
        if '_' in k:
            ks = k.split('_')
            ks_new = []
            for it in ks:
                if it not in ['at','the','it','is','in','on','of','a','with','almost'] :
                    if it in embeds :
                        ks_new.append(it)
                    elif custom_map2[it] in embeds:
                        ks_new.append(custom_map2[it])
                       
            
            emb = torch.stack([embeds[it] for it in ks_new]).mean(dim=0)
        else:
            emb = embeds[k] 
        embs.append(emb)
    embs = torch.stack(embs)
    print ('Loaded embeddings from file %s' % emb_file, embs.size())
    return embs

def load_fasttext_embeddings(emb_file,vocab):
    custom_map = {
        'Faux.Fur': 'fake fur',
        'Faux.Leather': 'fake leather',
        'Full.grain.leather': 'thick leather',
        'Hair.Calf': 'hairy leather',
        'Patent.Leather': 'shiny leather',
        'Boots.Ankle': 'ankle boots',
        'Boots.Knee.High': 'kneehigh boots',
        'Boots.Mid-Calf': 'midcalf boots',
        'Shoes.Boat.Shoes': 'boatshoes',
        'Shoes.Clogs.and.Mules': 'clogs shoes',
        'Shoes.Flats': 'flats shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traficlight',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }
    vocab_lower = [v.lower() for v in vocab]
    vocab = []
    for current in vocab_lower:
        if current in custom_map:
            vocab.append(custom_map[current])
        else:
            vocab.append(current)

    
    ft = fasttext.load_model(emb_file) #DATA_FOLDER+'/fast/cc.en.300.bin')
    embeds = []
    for k in vocab:
        if '_' in k:
            ks = k.split('_')
            emb = np.stack([ft.get_word_vector(it) for it in ks]).mean(axis=0)
        else:
            emb = ft.get_word_vector(k)
        embeds.append(emb)

    embeds = torch.Tensor(np.stack(embeds))
    print('Fasttext Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds



def initialize_wordembedding_matrix(name, vocab, cfg):
    """
    Args:
    - name: hyphen separated word embedding names: 'glove-word2vec-conceptnet'.
    - vocab: list of attributes/objects.
    """
    wordembs = name.split('+')
    result = None

    for wordemb in wordembs:
        if wordemb == 'glove':
            wordemb_ = load_word_embeddings(
                f'{cfg.DATASET.root_dir}/../glove/glove.6B.300d.txt', vocab)
        elif wordemb == 'word2vec':
            wordemb_ = load_word2vec_embeddings(
                f'{cfg.DATASET.root_dir}/../w2v/GoogleNews-vectors-negative300.bin', vocab)
        elif 'bert' in wordemb:
            wordemb_ = load_bert_embeddings(wordemb,vocab,cfg)
        elif wordemb == 'fasttext':
            wordemb_ = load_fasttext_embeddings(
                f'{cfg.DATASET.root_dir}/../fast/cc.en.300.bin', vocab)
        elif wordemb == 'conceptnet':
            wordemb_ = load_word_embeddings(
                f'{cfg.DATASET.root_dir}/../conceptnet/mit-states.txt', vocab)
        if result is None:
            result = wordemb_
        else:
            result = torch.cat((result, wordemb_), dim=1)
    dim = 300 * len(wordembs)
    return result, dim
