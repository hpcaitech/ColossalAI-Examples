#import torch
#torch.multiprocessing.set_start_method('spawn')
import os
from pathlib import Path

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CosineAnnealingLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torchvision import transforms
#from torchvision.datasets import CIFAR10
#from torchvision.models import resnet18
from tqdm import tqdm
import dataload 
from resnet import resnet18
#
filename='./dataset/hiv/raw/HIV.csv'
train,test,val=dataload.dataprocess(filename)
#training_generator = data.DataLoader(data_process_loader_Property_Prediction(test.index.values, 		test.Label.values, test), **params)


def main():
    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    # build resnet
    model = resnet18(num_classes=2)

    # build dataloaders


    train_dataloader = get_dataloader(dataset=dataload.data_process_loader_Property_Prediction(train.index.values, 
																					 train.Label.values, train),
                                      shuffle=True,
                                      batch_size=64,
                                      num_workers=0,
                                      pin_memory=True,
                                      )

    test_dataloader = get_dataloader(dataset=dataload.data_process_loader_Property_Prediction(test.index.values, 
																					 test.Label.values, test),
                                     add_sampler=False,
                                     batch_size=64,
                                     num_workers=0,
                                     pin_memory=True,
                                     )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                         optimizer,
                                                                         criterion,
                                                                         train_dataloader,
                                                                         test_dataloader,
                                                                         )
    # build a timer to measure time
    timer = MultiTimer()

    # create a trainer object
    trainer = Trainer(
        engine=engine,
        timer=timer,
        logger=logger
    )

    # define the hooks to attach to the trainer
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer, logger),

        # you can uncomment these lines if you wish to use them
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_dataloader=test_dataloader,
        test_interval=1,
        hooks=hook_list,
        display_progress=True
    )


if __name__ == '__main__':
    main()
