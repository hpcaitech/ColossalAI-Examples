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
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy


def main():
    args = parse_args()

    # init distributed environment
    colossalai.launch_from_torch(config='./configs/colossalai_zero.py')

    use_zero = hasattr(gpc.config, 'zero')

    # get logger
    logger = get_dist_logger()

    if not any([args.train, args.eval, args.predict]):
        raise ArgumentError("At least one of train, eval and predict should be set")

    # exit if the output directory is not empty to avoid overwritting
    output_dir = Path(args.output_dir).absolute()
    args.output_dir = output_dir

    if args.train and output_dir.exists and next(output_dir.iterdir(), None):
        raise FileExistsError(f"Output directory ({output_dir}) already exists and is not empty.")

    output_dir.mkdir(exist_ok=True)

    # get data processor
    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())

    # get tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(args.vocab_file,
                                                           do_lower_case=args.do_lower_case,
                                                           max_len=512)

    # check if checkpoint file is given
    if args.init_checkpoint:
        use_hf_pretrain = False
    else:
        use_hf_pretrain = True

    # Prepare model
    if use_zero:
        shard_strategy = TensorShardStrategy()
        with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy,
                             shard_param=True):
            model = get_model(args.bert_config, num_labels, use_hf_pretrain)
    else:
        model = get_model(args.bert_config, num_labels, use_hf_pretrain)

    if use_hf_pretrain:
        logger.info("Loading model checkpoint from HuggingFace pretrained weights", ranks=[0])
    else:
        logger.info(f"Loading model checkpoint from {args.init_checkpoint}", ranks=[0])
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        checkpoint = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Model checkpoint is loaded", ranks=[0])

    # Prepare optimizer
    optimizer = get_optimizer(model, args.learning_rate)

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

    engine, train_dataloader, eval_dataloader, lr_scheduler = colossalai.initialize(model=model,
                                                                                    optimizer=optimizer,
                                                                                    criterion=criterion,
                                                                                    train_dataloader=train_dataloader,
                                                                                    test_dataloader=eval_dataloader,
                                                                                    lr_scheduler=lr_scheduler)

    if args.train:
        # train
        run_train(args, engine, train_dataloader, lr_scheduler, logger)

    if args.eval:
        run_eval(args, engine, eval_dataloader, eval_examples, num_labels, label_map, logger)

    gpc.destroy()


if __name__ == '__main__':
    main()
