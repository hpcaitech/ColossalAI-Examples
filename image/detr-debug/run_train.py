import os
import colossalai
from colossalai.utils import get_dataloader, MultiTimer, is_using_pp
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR, CosineAnnealingWarmupLR

from colossalai.builder import build_pipeline_model
from colossalai.engine.schedule import (InterleavedPipelineSchedule, PipelineSchedule)
from colossalai.context import ParallelMode
from colossalai.utils.model.pipelinable import PipelinableContext

import torch
from torch.autograd import Variable
from models import build_model
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from datasets import build_dataset, get_coco_api_from_dataset
import util.misc as utils
from torch.utils.data import DataLoader
from datasets.coco_eval import CocoEvaluator
from models.detr import DETR

from models.backbone import build_backbone
from models.matcher import build_matcher
from models.transformer import build_transformer
from models.detr import SetCriterion, PostProcess

DATASET_PATH = str(os.environ['DATA'])  # The directory of your dataset



def train_detr():
    args_co = colossalai.get_default_parser().parse_args()

    device = torch.device("cuda")
    colossalai.launch_from_torch(config='/data/huxin/xjtuhx/projects/oneyear/ColossalAI/examples/image/detr/configs/detr_1d.py')
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])
    
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)
    
    use_pipeline = is_using_pp()
    
    args_mo = gpc.config
    
    num_classes = 557 if args_mo.dataset_file != 'coco' else 91
    device = torch.device(args_mo.device)
    backbone = build_backbone(args_mo)
    transformer = build_transformer(args_mo)
    matcher = build_matcher(args_mo)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args_mo.bbox_loss_coef}
    weight_dict['loss_giou'] = args_mo.giou_loss_coef
    if args_mo.masks:
        weight_dict["loss_mask"] = args_mo.mask_loss_coef
        weight_dict["loss_dice"] = args_mo.dice_loss_coef
    # TODO this is a hack
    if args_mo.aux_loss:
        aux_weight_dict = {}
        for i in range(args_mo.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args_mo.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args_mo.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    if use_pipeline:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = build_model(backbone=backbone, transformer=transformer, num_classes=num_classes)
        pipelinable.to_layer_list()
        pipelinable.load_policy("uniform")
        model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))

        print(model)
    
    else:
        model = build_model(args=args_mo, backbone=backbone, transformer=transformer, num_classes=num_classes)
    
    
    model.to(device)
    
    
    # count number of parameters
    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_stage = 0
    else:
        pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    logger.info(f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": gpc.config.lr_backbone,
        },
    ]

    dataset_train = build_dataset(image_set='train', args=args_mo)
    dataset_val = build_dataset(image_set='val', args=args_mo)

    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, gpc.config.BATCH_SIZE, drop_last=True)

    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args_mo.num_workers)
    dataloader_val = DataLoader(dataset_val, gpc.config.BATCH_SIZE, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args_mo.num_workers)

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
            if args_mo.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args_mo.clip_max_norm)
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








