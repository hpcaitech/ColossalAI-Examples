import colossalai
import transformers
import torch
from argparse import ArgumentError
from pathlib import Path
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from arguments import parse_args
from processors import PROCESSORS
from utils import (get_model, get_optimizer, get_lr_scheduler, get_eval_dataloader, get_train_dataloader, run_eval,
                   run_train)

from colossalai.engine.gradient_accumulation import GradAccumLrSchedulerByStep
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.nn.parallel import ZeroDDP
from colossalai.gemini import ChunkManager, GeminiManager
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup
from colossalai.utils import get_current_device


def main():
    # init distributed environment
    args = parse_args()
    colossalai.launch_from_torch(config={})

    # get logger
    logger = get_dist_logger()

    if not any([args.train, args.eval, args.predict]):
        raise ArgumentError("At least one of train, eval and predict should be set")

    # exit if the output directory is not empty to avoid overwritting
    output_dir = Path(args.output_dir).absolute()
    args.output_dir = output_dir

    if args.train and output_dir.exists() and next(output_dir.iterdir(), None):
        raise FileExistsError(f"Output directory ({output_dir}) already exists and is not empty.")

    output_dir.mkdir(exist_ok=True)

    # get data processor
    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())

    # get tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(args.vocab_file,
                                                           do_lower_case=args.do_lower_case,
                                                           max_len=512)

    # Prepare model
    with ColoInitContext(device=get_current_device()):
        model = get_model(args.bert_config, num_labels)
    logger.info("Loading model checkpoint from HuggingFace pretrained weights", ranks=[0])

    # added zero ddp
    chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
    pg = ProcessGroup()
    dp_pg = pg.dp_process_group()
    placement_policy = 'auto'
    chunk_manager = ChunkManager(chunk_size,
                                 process_group=pg,
                                 enable_distributed_storage=True,
                                 init_device=GeminiManager.get_default_device(placement_policy))
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager)

    # Prepare optimizer
    optimizer = get_optimizer(model, args.learning_rate)
    optimizer = ZeroOptimizer(optimizer, model, initial_scale=8192)

    # prepare loss function
    criterion = torch.nn.CrossEntropyLoss()

    # train
    if args.train:
        # get dataloader
        train_dataloader = get_train_dataloader(args, tokenizer, processor, logger)

        # prepare lr scheduler
        steps_per_epoch = GradAccumLrSchedulerByStep.compute_effective_steps_per_epoch(
            train_dataloader, gpc.config.get('gradient_accumulation', 1))
        total_steps = args.num_train_epochs * steps_per_epoch
        lr_scheduler = get_lr_scheduler(optimizer, total_steps=total_steps, warmup_ratio=args.warmup_proportion)
    else:
        train_dataloader = None
        lr_scheduler = None

    if args.eval:
        eval_dataloader, eval_examples, label_map = get_eval_dataloader(args, tokenizer, processor, logger)
    else:
        eval_dataloader = None

    if args.train:
        # train
        run_train(args, model, optimizer, criterion, train_dataloader, lr_scheduler, logger)

    if args.eval:
        run_eval(args, model, criterion, eval_dataloader, eval_examples, num_labels, label_map, logger)

    gpc.destroy()


if __name__ == '__main__':
    main()
