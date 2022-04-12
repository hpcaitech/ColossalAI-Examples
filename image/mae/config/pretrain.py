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
#   eg: logger, eff_batch_size

logger = get_dist_logger()
logger.info(f"Loading config from file {__file__}")

# ==== Static Configuration ====

# Toggle more loggings
VERBOSE = True

NUM_EPOCHS = 2
BATCH_SIZE = 4
ACCUM_ITER = 1

LEARNING_RATE = 0.01
_BASE_LEARNING_RATE = 1e-3
try:
    LEARNING_RATE
except NameError:
    eff_batch_size = BATCH_SIZE * ACCUM_ITER * misc.get_world_size()
    LEARNING_RATE = _BASE_LEARNING_RATE * eff_batch_size / 256

WEIGHT_DECAY = 0.5

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
