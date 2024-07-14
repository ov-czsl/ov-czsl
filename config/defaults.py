from yacs.config import CfgNode as CN


_C = CN()
_C.config_name = 'attribute_probing'

# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
_C.DATASET = CN(new_allowed=True)
_C.DATASET.name = 'mitstates'
_C.DATASET.root_dir = '/'
_C.DATASET.splitname = 'compositional-split-natural'

# -----------------------------------------------------------------------
# Distributed settings
# -----------------------------------------------------------------------
_C.DISTRIBUTED = CN()
_C.DISTRIBUTED.backend = 'nccl'
_C.DISTRIBUTED.world_size = 1

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
_C.MODEL = CN(new_allowed=True)
_C.MODEL.name = 'attribute_probing'
_C.MODEL.load_checkpoint = False
_C.MODEL.weights = ''
_C.MODEL.optim_weights = ''
_C.MODEL.eval_topk = 1

# -----------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------
_C.TRAIN = CN(new_allowed=True)

_C.TRAIN.loss = 'bce' # or 'lsep'

_C.TRAIN.log_dir = '/'
_C.TRAIN.checkpoint_dir = '/'
_C.TRAIN.seed = 124
_C.TRAIN.num_workers = 4

_C.TRAIN.test_batch_size = 32
_C.TRAIN.batch_size = 256
_C.TRAIN.lr = 0.001

_C.TRAIN.disp_interval = 100
_C.TRAIN.save_every_epoch = 1
_C.TRAIN.eval_every_epoch = 1

# -----------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------
_C.EVAL = CN(new_allowed=True)
_C.EVAL.topk = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()