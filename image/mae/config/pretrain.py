import os
from pathlib import Path
from colossalai.amp import AMP_TYPE
from torchvision import transforms
from util.crop import RandomResizedCrop
from colossalai.logging import get_dist_logger

logger = get_dist_logger()
logger.info(f"Loading config from file {__file__}")

# Static Configuration

BATCH_SIZE = 4
NUM_EPOCHS = 2

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

# Dynamic Configuration

try:
    DATAPATH = Path(os.environ["DATA"])
except KeyError:
    DATAPATH = Path(__file__).parent / "data"

logger.info(f"DATAPATH: {DATAPATH.absolute()}")
