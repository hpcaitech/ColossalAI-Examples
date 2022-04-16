import os
from pathlib import Path

from colossalai.amp import AMP_TYPE
from torchvision import transforms

import util.misc as misc
from util.crop import RandomResizedCrop

# ==== Colossal-AI Configuration ====

gradient_accumulation = 1
fp16 = dict(mode=AMP_TYPE.TORCH)

# ==== Model Configuration ====
#
# Variable Naming Convension:
#
# 1. `THIS_WILL_BE_DERECTLY_ACCESSED_BY_MAIN`: All capital.
#   eg: VERBOSE, LEARNING_RATE
#
# 2. `_THIS_WILL_BE_USED_TO_GENERATE_(1)`: Begin with underscore.
#   eg: __BASE_LEARNING_RATE
#
# 3. `this_is_a_simple_helper`: Snake case.
#   eg: eff_batch_size

# toggle more loggings
VERBOSE = False
DEBUG = False

NUM_EPOCHS = 800
# epochs to warmup LR
WARMUP_EPOCHS = 40 if NUM_EPOCHS > 40 else 0

# Interval to save a checkpoint
CHECKPOINT_INTERVAL = 20

# Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
BATCH_SIZE = 4

# Place to save pretrained model
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    eff_batch_size = BATCH_SIZE * gradient_accumulation * misc.get_world_size()
    LEARNING_RATE = _BASE_LEARNING_RATE * eff_batch_size / 256

WEIGHT_DECAY = 0.5

# Use (per-patch) normalized pixels as targets for computing loss
NORM_PIX_LOSS = True

# resume from checkpoint
RESUME = False
if RESUME:
    RESUME_ADDRESS = ""

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
