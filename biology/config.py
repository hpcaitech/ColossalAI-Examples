from colossalai.amp import AMP_TYPE

BATCH_SIZE = 128
NUM_EPOCHS = 1

CONFIG = dict(
    fp16=dict(
        mode=AMP_TYPE.TORCH
    )
)
"""
train_dataset = CIFAR10(
        root="data",
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                    0.2023, 0.1994, 0.2010]),
            ]
        )
    )
"""