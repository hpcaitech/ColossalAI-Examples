import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from pathlib import Path
from arguments import parse_args
from pretrain_utils import get_model, get_optimizer, get_lr_scheduler, get_dataloader_for_pretraining, save_checkpoint
from loss import LossForPretraining
from torch.utils.tensorboard import SummaryWriter

from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.nn.parallel import ZeroDDP
from colossalai.gemini import ChunkManager, GeminiManager
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup
from colossalai.utils import get_current_device

from tqdm import tqdm


def main():
    # get args
    args = parse_args()
    colossalai.launch_from_torch(config={})

    # make sure data exists
    data_path = Path(args.data)
    assert data_path.exists()

    # create output dir
    # exit if the output directory is not empty to avoid overwritting
    output_dir = Path(args.output_dir).absolute()
    args.output_dir = str(output_dir)

    if output_dir.exists() and next(output_dir.iterdir(), None):
        raise FileExistsError(f"Output directory ({output_dir}) already exists and is not empty.")

    output_dir.mkdir(exist_ok=True)

    # build model, optimizer and criterion
    with ColoInitContext(device=get_current_device()):
        config, model = get_model(args.bert_config)

    # enable graident checkpointing
    model.gradient_checkpointing_enable()
    model = model.half().cuda()

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

    # create zero optimizer
    optimizer = get_optimizer(model, lr=args.lr)
    optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**14)

    # create loss function
    criterion = LossForPretraining(config.vocab_size)

    # build dataloader
    train_dataloader = get_dataloader_for_pretraining(root=args.data,
                                                      vocab_file=args.vocab_file,
                                                      global_batch_size=args.batch_size,
                                                      process_group=dp_pg)

    # build lr scheduler
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epoch
    lr_scheduler = get_lr_scheduler(optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio)

    # build tensorboard writer
    if gpc.get_local_rank(ParallelMode.DATA):
        writer = SummaryWriter(log_dir='./pretrain_runs')

    global_step = 0

    for epoch in range(args.epoch):
        progress = tqdm(train_dataloader, disable=not gpc.get_global_rank() == 0)
        for batch_data in progress:
            input_ids = batch_data['input_ids'].cuda()
            token_type_ids = batch_data['token_type_ids'].cuda()
            attention_mask = batch_data['attention_mask'].cuda()
            mlm_label = batch_data['labels'].cuda()
            nsp_label = batch_data['next_sentence_labels'].cuda()

            output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = criterion(output['prediction_logits'], output['seq_relationship_logits'], mlm_label, nsp_label)
            optimizer.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            # record in tensorboard
            if gpc.get_local_rank(ParallelMode.DATA):
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], global_step)

            global_step += 1

        if epoch % args.save_checkpoint_interval == 0:
            save_checkpoint(model, args.output_dir, f'epoch{epoch}')


if __name__ == '__main__':
    main()
