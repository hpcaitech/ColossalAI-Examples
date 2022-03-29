import os
import colossalai
import torch
from pathlib import Path
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_dataloader
from colossalai.trainer import Trainer, hooks
from colossalai.nn.lr_scheduler import LinearWarmupLR
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
    args = parser.parse_args()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # launch from slurm batch job
    # colossalai.launch_from_slurm(config=args.config,
    #                              host=args.host,
    #                              port=args.port,
    #                              backend=args.backend
    #                              )

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # build model
    model = vit_base_patch16_224(drop_rate=0.1)

    # build dataloader
    train_dataset = datasets.Caltech101(
        root=Path(os.environ['DATA']),
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

        if epoch % 10 == 0:
            logger.info('epoch: {}, loss: {}'.format(epoch, loss.item()))


if __name__ == '__main__':
    main()
