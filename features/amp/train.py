import os
from pathlib import Path

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.utils import get_dataloader
from colossalai.trainer import Trainer, hooks
from timm.models import vit_base_patch16_224
from torchvision import datasets, transforms


class Gray2RGB:
    """Convert all images (rgb or grayscale) to rgb.

    """

    def __call__(self, img):
        """
        Args:
            img: Tensor

        Returns:
            Tensor: RGB image.
        """
        if img.size(dim=-3) == 1:
            img = img.repeat(3, 1, 1)
        return img


def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    parser.add_argument('--use_trainer', action='store_true', help='whether use trainer to execute the training')
    args = parser.parse_args()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # build model
    model = vit_base_patch16_224(drop_rate=0.1)

    # build dataloader
    train_dataset = datasets.Caltech101(
        root=Path(os.environ.get('DATA', './data')),
        download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Gray2RGB(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]))

    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=gpc.config.BATCH_SIZE,
                                      pin_memory=True,
                                      )

    # build optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.1)

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # lr_scheduelr
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=1, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, _, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader,
    )
    logger.info("initialized colossalai components", ranks=[0])

    # build trainer
    if not args.use_trainer:
        engine.train()
        for epoch in range(gpc.config.NUM_EPOCHS):
            for img, label in train_dataloader:
                img = img.cuda()
                label = label.cuda()
                engine.zero_grad()
                output = engine(img)
                loss = engine.criterion(output, label)
                engine.backward(loss)
                engine.step()
                lr_scheduler.step()

            logger.info('epoch: {}, loss: {}'.format(epoch, loss.item()))
    else:
        # build trainer
        trainer = Trainer(engine=engine, logger=logger)

        # build hooks
        hook_list = [
            hooks.LossHook(),
            hooks.LogMetricByEpochHook(logger),
            hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
        ]

        # start training
        trainer.fit(
            train_dataloader=train_dataloader,
            epochs=gpc.config.NUM_EPOCHS,
            hooks=hook_list,
            display_progress=True,
            test_interval=1
        )

    gpc.destroy()


if __name__ == '__main__':
    main()
