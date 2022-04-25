import colossalai
import torch
import os

from colossalai.nn import calc_acc
from colossalai.utils import get_dataloader, save_checkpoint
from colossalai.logging import get_dist_logger
from torch.nn.modules import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import CIFAR10
from model_zoo.vit import vit_tiny_patch4_32

def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=os.environ['DATA'], train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=os.environ['DATA'], train=False, download=True, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader


BATCH_SIZE = 128
NUM_EPOCHS = 10
CONFIG = dict()


def train():
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(config=CONFIG, backend=args.backend)

    logger = get_dist_logger()
    model = vit_tiny_patch4_32()
    train_dataloader , test_dataloader = build_cifar(BATCH_SIZE)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         criterion=criterion,
                                                                         train_dataloader=train_dataloader,
                                                                         test_dataloader=test_dataloader)

    logger.info("Engine is built", ranks=[0])

    for epoch in range(NUM_EPOCHS):
        engine.train()
        for _, (img, label) in enumerate(train_dataloader):
            engine.zero_grad()
            output = engine(img)
            loss = engine.criterion(output, label.cuda())
            engine.backward(loss)
            engine.step()
        logger.info(f"epoch = {epoch}, train loss = {loss}", ranks=[0]) 

        engine.eval()
        acc = 0
        test_cases = 0
        for _, (img, label) in enumerate(test_dataloader):
            output = engine(img)
            acc += calc_acc(output, label.cuda())
            test_cases += len(label)
        logger.info(f"epoch = {epoch}, test acc = {acc/test_cases}", ranks=[0])
        save_checkpoint('vit_cifar.pt', epoch, engine.model)


if __name__ == '__main__':
    train()
