# +
import colossalai
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.core import global_context as gpc
from colossalai.amp import AMP_TYPE
from colossalai.utils import MultiTimer, is_using_pp
from colossalai.kernel import LayerNorm
from colossalai.nn.optimizer import FusedAdam
from colossalai.context.parallel_mode import ParallelMode

from data_processor import build_train_valid_test_data_iterators
from data_processor.tokenizer import initialize_tokenizer, get_padded_vocab_size
from data_processor.bert_helper import get_batch, ParallelDataIterator

from model.bert import BertForPretrain, build_pipeline_bert
from loss_func.bert_loss import BertLoss
from lr_scheduler import AnnealingLR

import torch
import torch.nn as nn


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
    if hasattr(gpc.config, 'fp16') and gpc.config.fp16.get('mode') == AMP_TYPE.NAIVE:
        is_naive_fp16 = True
    else:
        is_naive_fp16 = False

    use_pipeline = is_using_pp()
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=gpc.config.HIDDEN_SIZE,
        max_sequence_length=gpc.config.SEQ_LENGTH,
        num_attettion_heads=gpc.config.NUM_ATTENTION_HEADS,
        convert_fp16_to_fp32_in_softmax=True,
        is_naive_fp16=is_naive_fp16,
        add_binary_head=gpc.config.ADD_BINARY_HEAD
    )

    if use_pipeline:
        model = build_pipeline_bert(num_layers=gpc.config.DEPTH,
                                    num_chunks=1,
                                    **kwargs)
    else:
        model = BertForPretrain(num_layers=gpc.config.DEPTH, **kwargs)

    logger.info(f"Model is built with softmax in fp32 = {is_naive_fp16}", ranks=[0])

    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    logger.info(f"This model has {total_numel} parameters")
    
    # Loss
    criterion = BertLoss()
    logger.info("Criterion is built", ranks=[0])
    
    # Layernorm and bias has no weight decay
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in model.modules():
        if isinstance(module_, LayerNorm):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                    if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                    if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                    if p is not None and n == 'bias'])

    logger.info(
        f"without weight decay param: {len(no_weight_decay_params['params'])}, with weight decay param: {len(weight_decay_params['params'])}")
    
    # Optimizer
    optimizer = FusedAdam((weight_decay_params, no_weight_decay_params),
                          lr=gpc.config.LR,
                          weight_decay=gpc.config.WEIGHT_DECAY)
    logger.info("Optimizer is built", ranks=[0])
    
    # Init
    engine, *dummy = colossalai.initialize(model,
                                           optimizer,
                                           criterion,
                                           )

    # build data iters for pipeline parallel
    if use_pipeline:
        train_data_iter = ParallelDataIterator(trainloader)
        valid_data_iter = ParallelDataIterator(validloader)

    for step in range(1, gpc.config.TRAIN_ITERS+1):
        engine.train()
        tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(trainloader)
        engine.zero_grad()
        lm_loss, sop_output = engine(tokens, padding_mask, types, lm_labels)
        train_loss = engine.criterion(lm_loss, sop_output, loss_mask, sentence_order)
        engine.backward(train_loss)
        engine.step()
        logger.info(f"train_loss: {train_loss}")


if __name__ == '__main__':
    main()
