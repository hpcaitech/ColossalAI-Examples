from cmath import e
import json
import os
import transformers
import torch

import numpy as np
from data import get_train_features, gen_tensor_dataset, convert_examples_to_features
from metrics import compute_metrics
from tqdm import trange, tqdm
from colossalai.core import global_context as gpc

from torch.utils.data import SequentialSampler, DataLoader
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.utils import get_dataloader
from transformers import BertForSequenceClassification

__all__ = [
    'get_model', 'get_optimizer', 'get_lr_scheduler', 'get_train_dataloader', 'run_train', 'get_eval_dataloader',
    'run_eval'
]


def get_model(config_file, num_labels):
    config = transformers.BertConfig.from_json_file(config_file)
    config.num_labels = num_labels
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    return model


def get_optimizer(model, lr):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']

    # configure the weight decay for layernorm and bias
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, bias_correction=False)
    return optimizer


def get_lr_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    return lr_scheduler


def get_train_dataloader(args, tokenizer, processor, logger):
    # build dataset
    train_features = get_train_features(
        args.data_dir,
        args.vocab_file,
        args.max_seq_length,
        args.do_lower_case,
        tokenizer,
        processor,
    )

    # build dataloader
    train_data = gen_tensor_dataset(train_features)
    train_dataloader = get_dataloader(dataset=train_data,
                                      shuffle=True,
                                      add_sampler=True,
                                      batch_size=args.train_batch_size)

    logger.info(f"Num examples = {len(train_features)}", ranks=[0])

    return train_dataloader


def run_train(args, model, optimizer, criterion, train_dataloader, lr_scheduler, logger):
    results = {}

    # prep
    global_step = 0
    num_train_steps = 0
    train_loss = 0
    num_train_examples = 0
    model.train()

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        train_loss, num_train_steps = 0, 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # stop if max steps exceeded
            if args.max_steps > 0 and global_step > args.max_steps:
                break

            optimizer.zero_grad()

            # forward
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = outputs['logits']
            loss = criterion(logits, label_ids)
            optimizer.backward(loss)

            # step
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()
            num_train_examples += input_ids.size(0)
            num_train_steps += 1
            global_step += 1

    train_loss = train_loss / num_train_steps
    results.update({
        'global_step': global_step,
        'train:loss': train_loss,
        'train:num_samples_per_gpu': num_train_examples,
        'train:num_steps': num_train_steps,
    })

    logger.info(results, ranks=[0])

    model_to_save = model
    state_dict = model_to_save.state_dict()

    if gpc.get_global_rank() == 0 and not args.skip_checkpoint:
        torch.save(
            {"model": state_dict},
            args.output_dir.joinpath('glue_weights.pth'),
        )

        # look for the model config and save
        while True:
            if hasattr(model_to_save, 'config'):
                with open(args.output_dir.joinpath('bert_config.json'), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                break
            else:
                if hasattr(model_to_save, 'module'):
                    model_to_save = model_to_save.module
                elif hasattr(model_to_save, 'model'):
                    model_to_save = model_to_save.model
                else:
                    raise AttributeError(
                        f"Cannot find attribute config or inner module in {model_to_save.__class__.__name__}")

    torch.distributed.barrier()


def get_eval_dataloader(args, tokenizer, processor, logger):
    # get eval dataset
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features, label_map = convert_examples_to_features(
        eval_examples,
        processor.get_labels(),
        args.max_seq_length,
        tokenizer,
    )

    logger.info(f"Num examples = {len(eval_examples)}", ranks=[0])
    eval_data = gen_tensor_dataset(eval_features)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
    )
    return eval_dataloader, eval_examples, label_map


def dump_predictions(path, label_map, preds, examples):
    label_rmap = {label_idx: label for label, label_idx in label_map.items()}
    predictions = {example.guid: label_rmap[preds[i]] for i, example in enumerate(examples)}
    with open(path, "w") as writer:
        json.dump(predictions, writer)


def run_eval(args, model, criterion, eval_dataloader, eval_examples, num_labels, label_map, logger):
    results = {}

    model.eval()
    preds = None
    out_label_ids = None
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    cuda_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                   for _ in range(len(eval_dataloader))]
    for i, batch_data in tqdm(
            enumerate(eval_dataloader),
            desc="Evaluating",
    ):
        input_ids, input_mask, segment_ids, label_ids = batch_data
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()

        with torch.no_grad():
            cuda_events[i][0].record()
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = outputs['logits']
            cuda_events[i][1].record()
            if args.eval:
                eval_loss += criterion(
                    logits.view(-1, num_labels),
                    label_ids.view(-1),
                ).mean().item()

        nb_eval_steps += 1
        nb_eval_examples += input_ids.size(0)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids,
                label_ids.detach().cpu().numpy(),
                axis=0,
            )
    torch.cuda.synchronize()
    eval_latencies = [event_start.elapsed_time(event_end) for event_start, event_end in cuda_events]
    eval_latencies = list(sorted(eval_latencies))

    def infer_latency_sli(threshold):
        index = int(len(eval_latencies) * threshold) - 1
        index = min(max(index, 0), len(eval_latencies) - 1)
        return eval_latencies[index]

    eval_throughput = (args.eval_batch_size / (np.mean(eval_latencies) / 1000))

    results.update({
        'eval:num_samples_per_gpu': nb_eval_examples,
        'eval:num_steps': nb_eval_steps,
        'infer:latency(ms):50%': infer_latency_sli(0.5),
        'infer:latency(ms):90%': infer_latency_sli(0.9),
        'infer:latency(ms):95%': infer_latency_sli(0.95),
        'infer:latency(ms):99%': infer_latency_sli(0.99),
        'infer:latency(ms):100%': infer_latency_sli(1.0),
        'infer:latency(ms):avg': np.mean(eval_latencies),
        'infer:latency(ms):std': np.std(eval_latencies),
        'infer:latency(ms):sum': np.sum(eval_latencies),
        'infer:throughput(samples/s):avg': eval_throughput,
    })

    preds = np.argmax(preds, axis=1)

    if args.predict:
        dump_predictions(
            os.path.join(args.output_dir, 'predictions.json'),
            label_map,
            preds,
            eval_examples,
        )

    if args.eval:
        results['eval:loss'] = eval_loss / nb_eval_steps
        eval_result = compute_metrics(args.task_name, preds, out_label_ids)
        results.update(eval_result)
        logger.info(results, ranks=[0])
