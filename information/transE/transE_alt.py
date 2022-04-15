import numpy as np
import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils import data as torch_data
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from colossalai.logging import get_dist_logger
import colossalai
from colossalai.nn.lr_scheduler import CosineAnnealingLR


logger = get_dist_logger()

def data_loader(file):
    triple = []
    with open(file, 'r') as f:
        lines = f.read().splitlines()
        for _ in lines:
            triple.append(_.split("\t"))
    return triple


def save_checkpoint(model, optim):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict()
    }, r"checkpoint.tar")


def load_checkpoint(path: str, model: nn.Module, optim: optimizer.Optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])


def test(test_triple, engine, hit_n: int, entity_size: int):
    test_dataset = torch_data.TensorDataset(torch.tensor([int(i[0]) for i in test_triple]),
                                            torch.tensor([int(i[1])
                                                         for i in test_triple]),
                                            torch.tensor([int(i[2]) for i in test_triple]))
    test_generator = torch_data.DataLoader(test_dataset, batch_size=20)
    hit = 0
    total_examples = 0
    for head, relation, tail in test_generator:
        tail = tail.cuda()
        current_batch_size = head.size()[0]
        entity_ids = torch.arange(end=entity_size).unsqueeze(0)
        all_entities = entity_ids.repeat(current_batch_size, 1)
        heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
        triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        triplets = triplets.cuda()
        tails_predictions, _ = engine(triplets, torch.tensor([[0,0,0]]).cuda())
        tails_predictions = tails_predictions.reshape(current_batch_size, -1)
        _, index_predictions = tails_predictions.topk(k=hit_n, largest=False)
        zero_tensor = torch.tensor([0]).cuda()
        one_tensor = torch.tensor([1]).cuda()
        hit += torch.where(index_predictions == tail.reshape(-1, 1),one_tensor, zero_tensor).sum().item()
        total_examples += current_batch_size
    return hit / total_examples


class TransE(nn.Module):
    def __init__(self, entity_size=100, relation_size=100, embedding_dim=100, norm=2):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.norm = norm
        self.entity_emb = self.init_entity_emb()
        self.relation_emb = self.init_relation_emb()

    def init_entity_emb(self):
        entity_embedding = nn.Embedding(
            num_embeddings=self.entity_size, embedding_dim=self.embedding_dim)
        entity_embedding.weight.data.uniform_(-6/np.sqrt(
            self.embedding_dim), 6/np.sqrt(self.embedding_dim))
        entity_embedding.weight.data[:-1, :].div_(
            entity_embedding.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return entity_embedding

    def init_relation_emb(self):
        relation_embedding = nn.Embedding(
            num_embeddings=self.relation_size, embedding_dim=self.embedding_dim)
        relation_embedding.weight.data.uniform_(-6/np.sqrt(
            self.embedding_dim), 6/np.sqrt(self.embedding_dim))
        relation_embedding.weight.data[:-1, :].div_(
            relation_embedding.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relation_embedding

    def forward(self, positive_triplets, negative_triplets):
        self.entity_emb.weight.data[:-1, :].div_(
            self.entity_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        positive_distance = self.distance(positive_triplets)
        negative_diatance = self.distance(negative_triplets)
        return positive_distance, negative_diatance

    def distance(self, triplets):
        return (self.entity_emb(triplets[:, 0])+self.relation_emb(triplets[:, 1])-self.entity_emb(triplets[:, 2])).norm(p=self.norm, dim=1)


class tranETrainer:
    def __init__(self, engine, train_dataloader, max_iteration=100, entity_size=15000):
        self.engine = engine
        self.train_dataloader = train_dataloader
        self.max_iteration = max_iteration
        self.entity_size = entity_size
    def train(self):
        
        for iter in range(self.max_iteration):
            engine.train()
            for real_heads, real_relations, real_tails in self.train_dataloader:
                positive_triples = torch.stack((real_heads, real_relations, real_tails), dim=1)
                random_entities = torch.randint(high=self.entity_size, size=real_heads.size())
                negative_triples = torch.stack((real_heads, real_relations, random_entities), dim=1)
                positive_triples = positive_triples.cuda()
                random_entities = random_entities.cuda()
                negative_triples = negative_triples.cuda()
                
                engine.zero_grad()
                pos_distance, neg_distance = engine(positive_triplets=positive_triples, negative_triplets=negative_triples)
                target = torch.tensor([-1]).cuda()
                
                loss = engine.criterion(pos_distance, neg_distance, target)
                engine.backward(loss.mean())
                engine.step()
            logger.info(f"Epoch {iter} - train loss: {loss.mean():.5}")
            lr_scheduler.step()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    colossalai.launch_from_torch(config='./config.py')
    triples = data_loader("./train.txt")
    test_triples = data_loader("./dev.txt")

    model = TransE(entity_size=15000, relation_size=240, embedding_dim=100, norm=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.05)
    criterion = nn.MarginRankingLoss(margin=0.8, reduction='none')
    lr_scheduler = CosineAnnealingLR(
        optimizer, total_steps=gpc.config.NUM_EPOCHS)
    head = torch.tensor([int(i[0]) for i in triples])
    rel = torch.tensor([int(i[1]) for i in triples])
    tail = torch.tensor([int(i[2]) for i in triples])
    triples_retrieve = torch_data.TensorDataset(head, rel, tail)
    train_dataloader = get_dataloader(
        dataset=triples_retrieve, batch_size=gpc.config.BATCH_SIZE, shuffle=True)

    #load_checkpoint('checkpoint.tar', model, optimizer)
    
    engine, train_dataloader, _, _ = colossalai.initialize(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        train_dataloader=train_dataloader, 
        test_dataloader=None)
    trainer = tranETrainer(engine, train_dataloader, gpc.config.NUM_EPOCHS)
    trainer.train()
    
    #save_checkpoint(model, optimizer)
    logger.info("evaluating...")
    engine.eval()
    hit = test(test_triples, engine, 5, 15000)
    logger.info(f"hit5: {hit}")
