import os

import colossalai
import colossalai.utils as utils
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import colo_set_process_memory_fraction
from colossalai.utils.timer import MultiTimer
from colossalai.nn._ops import *

from language.bert.colotensor.dataset import build_data
from language.bert.colotensor.model import build_model

def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=True, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    colossalai.launch_from_torch(config=args.config)

    logger = get_dist_logger()

    logger.info('Build data loader', ranks=[0])
    train_dataloader, test_dataloader = build_data(
        dataset_path=os.environ["DATA"],
        tokenizer_path=os.environ["TOKENIZER"],
        seq_len=gpc.config.SEQ_LENGTH,
        batch_size=gpc.config.BATCH_SIZE,
    )

    logger.info('Build model', ranks=[0])
    use_zero = hasattr(gpc.config, 'zero')

    # TODO(jzy) Add ZERO
    if use_zero:
        raise NotImplemented
    else:
        model = build_model()

    if use_zero:
        raise NotImplemented
    else:
        numel = calc_local_model_size(model)

    tflop = numel * gpc.config.BATCH_SIZE * gpc.config.SEQ_LENGTH \
        * gpc.get_world_size(ParallelMode.MODEL) * gpc.get_world_size(ParallelMode.DATA) * 8 / (1024 ** 4)

    criterion = nn.CrossEntropyLoss()

    logger.info('Build optimizer', ranks=[0])
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=0.001, weight_decay=1e-2)

    lr_scheduler = LinearWarmupLR(optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=2)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)
    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    timier = MultiTimer()

    trainer = Trainer(engine=engine, logger=logger, timer=timier)

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(ignored_steps=10, tflop_per_step=tflop),
        hooks.LogMetricByStepHook(),
        hooks.LogMemoryByEpochHook(logger),
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                test_interval=1,
                hooks=hook_list,
                display_progress=True,
                return_output_label=False,
                max_steps=5)


if __name__ == '__main__':
    main()