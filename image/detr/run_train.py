import os
import colossalai
from colossalai.utils import get_dataloader
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR, CosineAnnealingWarmupLR
import torch
from torch.autograd import Variable
from models import build_model
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from datasets import build_dataset, get_coco_api_from_dataset
import util.misc as utils
from torch.utils.data import DataLoader
from datasets.coco_eval import CocoEvaluator
DATASET_PATH = str(os.environ['DATA'])  # The directory of your dataset

def train_detr():
    args_co = colossalai.get_default_parser().parse_args()

    device = torch.device("cuda")
    colossalai.launch_from_torch(config=args_co.config)
    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    args = gpc.config
    model, criterion, postprocessors = build_model(args=args)
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": gpc.config.lr_backbone,
        },
    ]

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, gpc.config.BATCH_SIZE, drop_last=True)

    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, gpc.config.BATCH_SIZE, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    optimizer = torch.optim.AdamW(param_dicts, lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=gpc.config.WARMUP_EPOCHS)

    engine, train_dataloader, val_dataloader, _ = colossalai.initialize(model=model,
                                                                        optimizer=optimizer,
                                                                        criterion=criterion,
                                                                        train_dataloader=dataloader_train,
                                                                        test_dataloader=dataloader_val)

    for epoch in range(gpc.config.NUM_EPOCHS):
        engine.train()
        for samples, targets in train_dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            # loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            engine.zero_grad()
            engine.backward(losses)
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            engine.step()
        logger.info(f"Epoch {epoch} - train loss: {loss_value:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])

        engine.eval()
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        coco_evaluator = CocoEvaluator(base_ds, iou_types)
        for samples, targets in val_dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            loss_dict_reduced = utils.reduce_dict(loss_dict)

            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}

            eval_loss = sum(loss_dict_reduced_scaled.values())
            class_error = loss_dict_reduced['class_error']

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}



            if coco_evaluator is not None:
                coco_evaluator.update(res)
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        the_APs = coco_evaluator.coco_eval['bbox'].stats.tolist()

        logger.info(f"Epoch {epoch} - eval loss: {eval_loss} - the_APs: {the_APs}", ranks=[0])


if __name__ == '__main__':
    train_detr()








