import os
from pathlib import Path

from colossalai.amp import AMP_TYPE
from colossalai.logging import get_dist_logger
from torchvision import transforms

import util.misc as misc
from util.crop import RandomResizedCrop

# ==== Variable Naming Convension ====
#
# 1. `THIS_WILL_BE_DERECTLY_ACCESSED_BY_MAIN`: All capital.
#   eg: VERBOSE, LEARNING_RATE
#
# 2. `_THIS_WILL_BE_USED_TO_GENERATE_(1)`: Begin with underscore.
#   eg: __BASE_LEARNING_RATE
#
# 3. `this_is_a_simple_helper`: Snake case.
#   eg: eff_batch_size

# ==== Static Configuration ====

# toggle more loggings
VERBOSE = True

NUM_EPOCHS = 2
# epochs to warmup LR
WARMUP_EPOCHS = 40 if NUM_EPOCHS > 40 else 0

# Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
BATCH_SIZE = 4
# Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
ACCUM_ITER = 1
# Masking ratio (percentage of removed patches).
MASK_RATIO = 0.75

# learning rate (absolute lr)
LEARNING_RATE = 0.01
# lower lr bound for cyclic schedulers that hit 0
MINIMUM_LEARNING_RATE = 0
# base learning rate: absolute_lr = base_lr * total_batch_size / 256
_BASE_LEARNING_RATE = 1e-3
try:
    LEARNING_RATE
except NameError:
    eff_batch_size = BATCH_SIZE * ACCUM_ITER * misc.get_world_size()
    LEARNING_RATE = _BASE_LEARNING_RATE * eff_batch_size / 256

WEIGHT_DECAY = 0.5

# Use (per-patch) normalized pixels as targets for computing loss
NORM_PIX_LOSS = True

# resume from checkpoint
RESUME = False
if RESUME:
    RESUME_ADDRESS = ""
    RESUME_START_EPOCH = 0

CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))

TRANSFORM_TRAIN = transforms.Compose(
    [
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TRANSFORM_VAL = transforms.Compose(
    [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ==== Dynamic Configuration ====

try:
    DATAPATH = Path(os.environ["DATA"])
except KeyError:
    DATAPATH = Path(__file__).parent.parent / "data"
