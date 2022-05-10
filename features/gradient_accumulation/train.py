import colossalai
import torch
import os

from pathlib import Path
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from titans.utils import barrier_context


def main():
    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    # build resnet
    model = resnet18(num_classes=10)

    # build dataloaders
    with barrier_context():
        train_dataset = CIFAR10(root=Path(os.environ.get('DATA', './data')),
                                download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(size=32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                                ]))

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    engine, train_dataloader, _, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader,
    )

    # verify gradient accumulation
    engine.train()
    param_by_iter = []
    for idx, (img, label) in enumerate(train_dataloader):
        img = img.cuda()
        label = label.cuda()

        engine.zero_grad()
        output = engine(img)
        train_loss = engine.criterion(output, label)
        engine.backward(train_loss)
        engine.step()

        # we print the first 10 values of the model param to
        # show param is only updated in last iteration
        ele_1st = next(model.parameters()).flatten()[0]
        param_by_iter.append(ele_1st.item())

        # only run for 4 iterations
        if idx == gpc.config.gradient_accumulation - 1:
            break

    for iteration, val in enumerate(param_by_iter):
        print(f'iteration {iteration} - value: {val}')

    if param_by_iter[-1] != param_by_iter[0]:
        print('The parameter is only updated in the last iteration')


if __name__ == '__main__':
    main()
