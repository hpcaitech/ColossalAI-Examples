import os
from typing import List
import time
import logging

logging.disable(logging.INFO)

import torch
import torch.nn as nn
from titans.dataloader.cifar10 import build_cifar
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from colossalai.pipeline.rpc.utils import parse_args, rpc_run
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine

torch.manual_seed(1024)

def flatten(x):
    return torch.flatten(x, 1)

def AccuracyMetric(out: torch.Tensor, label: torch.Tensor):
    out = out.cpu()
    label = label.cpu()
    pre_lab = out.argmax(1)
    return accuracy_score(label, pre_lab)


def partition(pp_rank: int, chunk: int, stage_num: int):
    pipelinable = PipelinableContext()

    # build model partitions
    with pipelinable:
        # input : [B, 3, 32, 32]
        _ = resnet50()

    exec_seq = [
        'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', (flatten, "behind"), 'fc'
    ]
    pipelinable.to_layer_list(exec_seq)
    partition = pipelinable.partition(1, stage_num, pp_rank)
    return partition

def run_master(args):
    batch_size = args.batch_size
    chunk = args.chunk
    device = args.device
    world_size = args.world_size
    stage_num = world_size
    num_microbatches = args.num_microbatches
    epoch = args.epoch

    # build dataloader
    torch.manual_seed(1024)
    root = os.environ.get('DATA', './data')
    train_dataloader, test_dataloader = build_cifar(batch_size, root, padding=4, crop=32, resize=32)

    criterion = nn.CrossEntropyLoss()  

    pp_engine = OneFOneBPipelineEngine(partition_fn=partition,
                                       stage_num=stage_num,
                                       num_microbatches=num_microbatches,
                                       device=device,
                                       chunk=chunk,
                                       criterion=criterion,
                                       metric=AccuracyMetric,
                                       checkpoint=False)

    pp_engine.initialize_optimizer(getattr(torch.optim, args.optimizer), lr=1e-3)


    s = time.time()
    torch.manual_seed(1024)
    for epoch_id in range(epoch):
        data_iter = tqdm(train_dataloader, desc=f'[Train->Epoch {epoch_id}]')
        for bx, by in data_iter:
            losses, metrics = pp_engine.forward_backward(bx, labels=by, forward_only=False)
            
            batch_info = dict()
            avg_loss = sum(losses) / len(losses)
            batch_info['avg_loss'] = avg_loss
            if metrics is not None and metrics[0] is not None:
                avg_metric = sum(metrics) / len(metrics)
                batch_info['avg_metric'] = avg_metric
            data_iter.set_postfix(batch_info)
   
        val_metrics = []
        for bx, by in test_dataloader:
            _, metrics = pp_engine.forward_backward(bx, labels=by, forward_only=True)
            val_metrics.append(sum(metrics) / len(metrics))

        acc = round(sum(val_metrics) / len(val_metrics), 3)        
        print(f"[Test] metrics: {acc}")


    cost_time = time.time() - s

    print("total cost time :", cost_time)
    print("cost time per batch:", cost_time / len(train_dataloader))


if __name__ == '__main__':
    args = parse_args()
    # this is due to limitation of partition function
    assert args.world_size == 2, "partition for resnet only support world_size == 2"
    rpc_run(args, run_master)
