#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import torch
from colossalai.amp import AMP_TYPE
from torch.utils.data import DataLoader
import colossalai
from colossalai.utils import get_dataloader
from colossalai.logging import get_dist_logger
from dataloader.dataloader import TrainDataset
from titans.loss.embedding_loss import embeddingLoss
from titans.model.knowledge_graph_embedding import KGEModel
from dataloader.dataloader import BidirectionalOneShotIterator
from sklearn.metrics import average_precision_score
from dataloader.dataloader import TestDataset


def parse_args(args=None):
    parser = colossalai.get_default_parser()

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight',
                        action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save(
        {
            **save_variable_list, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(args.save_path, 'checkpoint'))

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, 'entity_embedding'), entity_embedding)

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, 'relation_embedding'), relation_embedding)


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    logger = get_dist_logger()
    for metric in metrics:
        logger.info(f"{mode} {metric} at step {step}: {metrics[metric]}")


def main(args):
    logger = get_dist_logger()
    CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))

    colossalai.launch_from_torch(config=CONFIG)

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be chosen.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be chosen.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logger.info(f"Model: {args.model}")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"#entity: {nentity}")
    logger.info(f"#relation: {nrelation}")

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logger.info(f"#train: {len(train_triples)}")
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logger.info(f"#valid: {len(valid_triples)}")
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logger.info(f"#test: {len(test_triples)}")

    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = KGEModel(model_name=args.model,
                         nentity=nentity,
                         nrelation=nrelation,
                         hidden_dim=args.hidden_dim,
                         gamma=args.gamma,
                         double_entity_embedding=args.double_entity_embedding,
                         double_relation_embedding=args.double_relation_embedding)
    logger.info("Model Parameter Configuration:")
    for name, param in kge_model.named_parameters():
        logger.info(f"Parameter {name}: {param.size()}, require_grad = {param.requires_grad}")

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        train_dataloader_head = get_dataloader(
            dataset=TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=True,
        )

        train_dataloader_tail = get_dataloader(
            dataset=TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=True,
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        criterion = embeddingLoss()

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, kge_model.parameters()),
                                     lr=current_learning_rate)
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

        engine, *dummy = colossalai.initialize(
            kge_model,
            optimizer,
            criterion,
        )

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logger.info(f"Loading checkpoint {args.init_checkpoint}...")
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logger.info(f"Randomly Initializing {args.model} Model...")
        init_step = 0

    step = init_step

    logger.info("Start Training...")
    logger.info(f"init_step = {init_step}")
    logger.info(f"batch_size = {args.batch_size}")
    logger.info(f"negative_adversarial_sampling = {args.negative_adversarial_sampling}")
    logger.info(f"hidden_dim = {args.hidden_dim}")
    logger.info(f"gamma = {args.gamma}")
    logger.info(f"negative_adversarial_sampling = {args.negative_adversarial_sampling}")

    if args.negative_adversarial_sampling:
        logger.info(f"adversarial_temperature = {args.adversarial_temperature}")

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        logger.info(f"learning_rate = {current_learning_rate}")

        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):
            engine.train()
            engine.zero_grad()
            train_loss, positive_sample_loss, negative_sample_loss = engine.criterion(train_iterator, args, kge_model)

            if args.regularization != 0.0:
                # Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (kge_model.entity_embedding.norm(p=3)**3 +
                                                        kge_model.relation_embedding.norm(p=3).norm(p=3)**3)
                train_loss = train_loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}

            log = {
                **regularization_log, 'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': train_loss.item()
            }

            engine.backward(train_loss)
            engine.step()
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logger.info(f"Change learning_rate to {current_learning_rate} at step {step}")
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, kge_model.parameters()),
                                             lr=current_learning_rate)
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logger.info('Evaluating on Valid Dataset...')
                engine.eval()

                if args.countries:
                    # Countries S* datasets are evaluated on AUC-PR
                    # Process test data for AUC-PR evaluation
                    sample = list()
                    y_true = list()
                    for head, relation, tail in test_triples:
                        for candidate_region in args.regions:
                            y_true.append(1 if candidate_region == tail else 0)
                            sample.append((head, relation, candidate_region))

                    sample = torch.LongTensor(sample)
                    if args.cuda:
                        sample = sample.cuda()

                    with torch.no_grad():
                        y_score = engine.model(sample).squeeze(1).cpu().numpy()

                    y_true = np.array(y_true)

                    # average_precision_score is the same as auc_pr
                    auc_pr = average_precision_score(y_true, y_score)

                    metrics = {'auc_pr': auc_pr}

                else:
                    # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
                    # Prepare dataloader for evaluation
                    valid_dataloader_head = DataLoader(TestDataset(valid_triples, all_true_triples, args.nentity,
                                                                   args.nrelation, 'head-batch'),
                                                       batch_size=args.test_batch_size,
                                                       num_workers=max(1, args.cpu_num // 2),
                                                       collate_fn=TestDataset.collate_fn)

                    valid_dataloader_tail = DataLoader(TestDataset(valid_triples, all_true_triples, args.nentity,
                                                                   args.nrelation, 'tail-batch'),
                                                       batch_size=args.test_batch_size,
                                                       num_workers=max(1, args.cpu_num // 2),
                                                       collate_fn=TestDataset.collate_fn)

                    valid_dataset_list = [valid_dataloader_head, valid_dataloader_tail]

                    logs = []

                    step = 0
                    total_steps = sum([len(dataset) for dataset in valid_dataset_list])

                    with torch.no_grad():
                        for valid_dataset in valid_dataset_list:
                            for positive_sample, negative_sample, filter_bias, mode in valid_dataset:
                                if args.cuda:
                                    positive_sample = positive_sample.cuda()
                                    negative_sample = negative_sample.cuda()
                                    filter_bias = filter_bias.cuda()

                                batch_size = positive_sample.size(0)

                                score = engine.model((positive_sample, negative_sample), mode)
                                score += filter_bias

                                # Explicitly sort all the entities to ensure that there is no test exposure bias
                                argsort = torch.argsort(score, dim=1, descending=True)

                                if mode == 'head-batch':
                                    positive_arg = positive_sample[:, 0]
                                elif mode == 'tail-batch':
                                    positive_arg = positive_sample[:, 2]
                                else:
                                    raise ValueError('mode %s not supported' % mode)

                                for i in range(batch_size):
                                    # Notice that argsort is not ranking
                                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                                    assert ranking.size(0) == 1

                                    # ranking + 1 is the true ranking used in evaluation metrics
                                    ranking = 1 + ranking.item()
                                    logs.append({
                                        'MRR': 1.0 / ranking,
                                        'MR': float(ranking),
                                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                                    })

                                if step % args.test_log_steps == 0:
                                    logger.info(f"Evaluating the model... {step}, {total_steps}")

                                step += 1

                    metrics = {}
                    for metric in logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

                log_metrics('Valid', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logger.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logger.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logger.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())
