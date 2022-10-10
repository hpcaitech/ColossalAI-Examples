import colossalai
from joblib import Parallel
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from arguments import parse_args
from pretrain_utils import get_model, get_optimizer, get_lr_scheduler, get_dataloader_for_pretraining
from loss import LossForPretraining
from torch.utils.tensorboard import SummaryWriter
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy


def main():
    # get args
    args = parse_args()
    colossalai.launch_from_torch(config='./configs/colossalai_zero.py')

    use_zero = hasattr(gpc.config, 'zero')

    # build model, optimizer and criterion
    if use_zero:
        shard_strategy = TensorShardStrategy()
        with ZeroInitContext(target_device=torch.cuda.current_device(), shard_strategy=shard_strategy,
                             shard_param=True):
            config, model = get_model(args.bert_config)
    else:
        config, model = get_model(args.bert_config)

    optimizer = get_optimizer(model, lr=args.lr)
    criterion = LossForPretraining(config.vocab_size)

    # build dataloader
    dataloader = get_dataloader_for_pretraining(
        root=args.data,
        local_rank=gpc.get_local_rank(ParallelMode.DATA),
        vocab_file=args.vocab_file,
        global_batch_size=args.batch_size,
    )

    # build lr scheduler
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epoch
    lr_scheduler = get_lr_scheduler(optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio)

    # initialize with colossalai
    engine, train_dataloader, _, lr_scheduelr = colossalai.initialize(model=model,
                                                                      optimizer=optimizer,
                                                                      train_dataloader=dataloader,
                                                                      criterion=criterion,
                                                                      lr_scheduler=lr_scheduler)

    # build tensorboard writer
    if gpc.get_local_rank(ParallelMode.DATA):
        writer = SummaryWriter(log_dir='./pretrain_runs')

    global_step = 0

    for epoch in range(args.epoch):
        for batch_data in train_dataloader:
            input_ids = batch_data['input_ids'].cuda()
            token_type_ids = batch_data['token_type_ids'].cuda()
            attention_mask = batch_data['attention_mask'].cuda()
            mlm_label = batch_data['labels'].cuda()
            nsp_label = batch_data['next_sentence_labels'].cuda()

            output = engine(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = engine.criterion(output.prediction_logits, output.seq_relationship_logits, mlm_label, nsp_label)
            engine.backward(loss)
            engine.step()
            lr_scheduelr.step()

            # record in tensorboard
            if gpc.get_local_rank(ParallelMode.DATA):
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/lr', lr_scheduelr.get_lr()[0], global_step)

            global_step += 1


if __name__ == '__main__':
    main()
