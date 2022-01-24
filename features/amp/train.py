import glob
from math import log
import os
import colossalai
from colossalai.nn.metric import Accuracy
import torch
from pathlib import Path
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_dataloader
from colossalai.trainer import Trainer, hooks
from colossalai.nn.lr_scheduler import LinearWarmupLR
from timm.models import vit_base_patch16_224
from torchvision import datasets, transforms

class all2rgb(torch.nn.Module):
    """Convert all images (rgb or grayscale) to rgb.

    """
    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img: Tensor

        Returns:
            Tensor: RGB image.
        """
        if img.size(dim=-3)==1:
            img = img.repeat(3, 1, 1)
        return img

def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()

    # launch from python 
    colossalai.launch(config=args.config,
                    rank=args.rank,
                    world_size=args.world_size,
                    host=args.host,
                    port=args.port,
                    backend=args.backend
                    )
    
    # launch from slurm batch job
    # colossalai.launch_from_slurm(config=args.config,
    #                              host=args.host,
    #                              port=args.port,
    #                              backend=args.backend
    #                              )

    # launch from torch
    # colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # build model
    model = vit_base_patch16_224(drop_rate=0.1)

    # build dataloader
    train_dataset = datasets.Caltech101(
        root = Path(os.environ['DATA']),
        download = True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ]))

    train_dataloader = get_dataloader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=gpc.config.BATCH_SIZE,
                                  num_workers=1,
                                  pin_memory=True,
                                  )

    # build optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.1)

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # lr_scheduelr
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=50, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, _, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader,
    )
    logger.info("initialized colossalai components", ranks=[0])

    # build trainer
    trainer = Trainer(engine=engine, logger=logger)

    # build hooks
    hook_list = [
        hooks.LossHook(),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),

        # comment if you do not need to use the hooks below
        hooks.SaveCheckpointHook(interval=1, checkpoint_dir='./ckpt'),
        hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        hooks=hook_list,
        display_progress=True,
        test_interval=1
    )


if __name__ == '__main__':
    main()
