from cmath import log
import os
from pathlib import Path

import colossalai
import torch
import time
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.trainer import Trainer, hooks
from timm.models import vit_base_patch16_224
from torchvision import datasets, transforms
from titans.utils import barrier_context


def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()

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
    with barrier_context():
        train_dataset = datasets.CIFAR10(
            root=Path(os.environ.get('DATA', './data')),
            download=True,
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
                                      pin_memory=True,
                                      )

    # build optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.1)

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # lr_scheduler
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=1, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, _, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader,
    )
    logger.info("initialized colossalai components", ranks=[0])

    # build trainer
    if not args.use_trainer:
        engine.train()
        for epoch in range(gpc.config.NUM_EPOCHS):
            start = get_time_stamp()
            for img, label in train_dataloader:
                img = img.cuda()
                label = label.cuda()
                engine.zero_grad()
                output = engine(img)
                loss = engine.criterion(output, label)
                engine.backward(loss)
                engine.step()
                lr_scheduler.step()
            end  = get_time_stamp()
            avg_step_time = (end - start) / len(train_dataloader)
            logger.info('epoch: {}, loss: {}, avg step time: {} / s'.format(epoch, loss.item(), avg_step_time))
    else:
        # build trainer
        trainer = Trainer(engine=engine, logger=logger)
        timer = MultiTimer()

        # build hooks
        hook_list = [
            hooks.LossHook(),
            hooks.LogMetricByEpochHook(logger),
            hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
            hooks.LogTimingByEpochHook(timer=timer, logger=logger)
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
