from asyncio.log import logger
from pathlib import Path
from colossalai.logging import get_dist_logger
import colossalai
import torch
import os
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
import torchvision.datasets as datasets
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
import models_vit
from util.crop import RandomResizedCrop

TRANSFORM_TRAIN = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

TRANSFORM_VAL = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def load_imgfolder(path, transform):
    return datasets.ImageFolder(path, transform=transform)

def main():
    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    # build mae model
    model = models_vit.vit_large_patch16(
        num_classes=1000,
        global_pool=False,
    )

    # build dataloaders
    datapath = Path(os.environ['DATA'])
    dataset_train = load_imgfolder(datapath/'train', TRANSFORM_TRAIN)
    dataset_val = load_imgfolder(datapath/'val', TRANSFORM_VAL)

    print(dataset_train)
    print(dataset_val)


if __name__ == '__main__':
    main()
