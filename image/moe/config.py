BATCH_SIZE = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 3e-2

NUM_EPOCHS = 200
WARMUP_EPOCHS = 40

parallel = dict()
max_ep_size = 1  # all experts are replicated in the case that user only has 1 GPU
clip_grad_norm = 1.0  # enable gradient clipping and set it to 1.0

LOG_PATH = f"./cifar10_moe"
