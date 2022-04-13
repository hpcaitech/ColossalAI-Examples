from pathlib import Path
from colossalai.logging import get_dist_logger
import colossalai
import torch
import os
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from torchvision.models import resnet34
from tqdm import tqdm
from tqdm import tqdm
import dataload 
from resnet import resnet18
filename='./dataset/hiv/raw/HIV.csv'
train,test,val=dataload.dataprocess(filename)

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

    for epoch in range(gpc.config.NUM_EPOCHS):
        engine.train()
        if gpc.get_global_rank() == 0:
            train_dl = tqdm(train_dataloader)
        else:
            train_dl = train_dataloader
        for img, label in train_dl:
            img = img.cuda()
            label = label.cuda()

            engine.zero_grad()
            output = engine(img)
            train_loss = engine.criterion(output, label)
            engine.backward(train_loss)
            engine.step()
        lr_scheduler.step()

        engine.eval()
        correct = 0
        total = 0
        for img, label in test_dataloader:
            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                output = engine(img)
                test_loss = engine.criterion(output, label)
            pred = torch.argmax(output, dim=-1)
            correct += torch.sum(pred == label)
            total += img.size(0)

        logger.info(
            f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])


if __name__ == '__main__':
    main()
