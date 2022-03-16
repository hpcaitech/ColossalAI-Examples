# +
import colossalai
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.core import global_context as gpc

from data_processor import build_train_valid_test_data_iterators
from data_processor.tokenizer import initialize_tokenizer, get_padded_vocab_size


# -

def launch_colossalai():
    # handle args
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    disable_existing_loggers()
    
    # init distributed env
    colossalai.launch_from_torch(config=args.config, seed=1234, backend='nccl')
    
    # create logger
    logger = get_dist_logger()
    logger.info(f'Launch Colossal-AI.', ranks=[0])
    return args, logger


def main():
    args, logger = launch_colossalai()
    
    # Dataloader
    initialize_tokenizer(gpc.config.VOCAB_FILE_PATH, tokenizer_type='BertWordPieceLowerCase')
    VOCAB_SIZE = get_padded_vocab_size()
    trainloader, validloader, testloader = build_train_valid_test_data_iterators(train_iters=gpc.config.TRAIN_ITERS,
        global_batch_size=gpc.config.GLOBAL_BATCH_SIZE,
        eval_interval=gpc.config.EVAL_INTERVAL,
        eval_iters=gpc.config.EVAL_ITERS,
        data_prefix=[gpc.config.DATA_PATH],
        data_impl='mmap',
        splits_string='949,50,1',
        max_seq_length=gpc.config.SEQ_LENGTH,
        masked_lm_prob=0.15,
        short_seq_prob=0.1,
        seed=1234,
        skip_warmup=True,
        binary_head=False,
        )
    logger.info(f'Create DataLoaders.', ranks=[0])
    
    # Model
    # Loss
    # Optimizer
    # Learning Rate Scheduler


if __name__ == '__main__':
    main()
