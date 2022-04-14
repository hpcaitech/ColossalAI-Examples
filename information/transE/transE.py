from typing import ForwardRef
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import module
from torch.optim import optimizer
from torch.utils import data as torch_data


def data_loader(file):
    triple = []
    with open(file,'r') as f:
        lines = f.read().splitlines()
        for _ in lines:
            triple.append(_.split("\t"))
    return triple

def save_checkpoint(model,optim):
    torch.save({
        "model":model.state_dict(),
        "optim":optim.state_dict()
    },r"checkpoint.tar")

def makeTrainDict(triple):
    t = {}
    for _ in triple:
        if _[0] not in t:
            t[_[0]]={}
        if _[1] not in t[_[0]]:
            t[_[0]][_[1]] = []
        t[_[0]][_[1]].append(_[2])
    return t


def load_checkpoint(path:str,model:nn.Module,optim:optimizer.Optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])

def predict_print(triple_dict,predict_triple,model:nn.Module,hit_n:int,entity_size:int):
    fileout = open(r"predict_most.txt","w")
    test_dataset = torch_data.TensorDataset(torch.tensor([int(i[0]) for i in predict_triple]),
                                    torch.tensor([int(i[1]) for i in predict_triple]))
    test_generator = torch_data.DataLoader(test_dataset,batch_size=50,shuffle=False)
    result = []
    index = 0
    for head, relation in test_generator:
        current_batch_size  = head.size()[0]
        entity_ids = torch.arange(end=entity_size).unsqueeze(0)
        all_entities = entity_ids.repeat(current_batch_size, 1)
        heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
        triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        tails_predictions = model.distance(triplets).reshape(current_batch_size, -1)
        _,index_predictions = tails_predictions.topk(k=hit_n,largest=False,sorted=True)
        result += index_predictions.tolist()

        # for i in range(current_batch_size):
        #     if predict_triple[i+index][0] in triple_dict:
        #         if predict_triple[i+index][1] in triple_dict[predict_triple[i+index][0]]:
        #             if len(triple_dict[predict_triple[i+index][0]][predict_triple[i+index][1]]) <=5:
        #                 for j in range(len(triple_dict[predict_triple[i+index][0]][predict_triple[i+index][1]])):
        #                     result[index+i][j] = triple_dict[predict_triple[i+index][0]][predict_triple[i+index][1]][j]
        # index += current_batch_size

    for line in result:
        print(*line,sep=',',file=fileout)
        




def test(test_triple,model:nn.Module,hit_n:int,entity_size:int):
    test_dataset = torch_data.TensorDataset(torch.tensor([int(i[0]) for i in test_triple]),
                                    torch.tensor([int(i[1]) for i in test_triple]),
                                    torch.tensor([int(i[2]) for i in test_triple]))
    test_generator = torch_data.DataLoader(test_dataset,batch_size=20)
    hit = 0
    total_examples = 0
    for head, relation, tail in test_generator:
        current_batch_size  = head.size()[0]
        entity_ids = torch.arange(end=entity_size).unsqueeze(0)
        all_entities = entity_ids.repeat(current_batch_size, 1)
        heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
        triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        tails_predictions = model.distance(triplets).reshape(current_batch_size, -1)
        _,index_predictions = tails_predictions.topk(k=hit_n,largest=False)
        zero_tensor = torch.tensor([0])
        one_tensor = torch.tensor([1])
        hit += torch.where(index_predictions == tail.reshape(-1,1), one_tensor, zero_tensor).sum().item()
        total_examples += current_batch_size
    return hit / total_examples


class TransE(nn.Module):
    def __init__(self,entity_size = 100, relation_size = 100,embedding_dim=100, margin=1,norm = 2):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.norm = norm
        self.entity_emb = self.init_entity_emb()
        self.relation_emb = self.init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')



    def init_entity_emb(self):
        entity_embedding  = nn.Embedding(num_embeddings=self.entity_size,embedding_dim=self.embedding_dim)
        entity_embedding.weight.data.uniform_(-6/np.sqrt(self.embedding_dim),6/np.sqrt(self.embedding_dim))
        entity_embedding.weight.data[:-1, :].div_(entity_embedding.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return entity_embedding

    def init_relation_emb(self):
        relation_embedding = nn.Embedding(num_embeddings=self.relation_size,embedding_dim=self.embedding_dim)
        relation_embedding.weight.data.uniform_(-6/np.sqrt(self.embedding_dim),6/np.sqrt(self.embedding_dim))
        relation_embedding.weight.data[:-1, :].div_(relation_embedding.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relation_embedding

    def forward(self,positive_triplets,negative_triplets):
        self.entity_emb.weight.data[:-1, :].div_(self.entity_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        positive_distance = self.distance(positive_triplets)
        negative_diatance = self.distance(negative_triplets)
        return self.loss(positive_distance,negative_diatance)

    def loss(self,pos_distance,neg_distance):
        target = torch.tensor([-1])
        return self.criterion(pos_distance,neg_distance,target)
        #return pos_distance

    def distance(self,triplets):
        return (self.entity_emb(triplets[:,0])+self.relation_emb(triplets[:,1])-self.entity_emb(triplets[:,2])).norm(p=self.norm,dim=1)

class tranETrainer:
    def __init__(self,triples,entity_size = 100, relation_size = 100,embedding_dim=100, margin=1,norm = 2,
                learning_rate = 0.1,batch_size = 10,max_iteration = 1000,load=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.entity_size = entity_size
        self.relation_size = relation_size
        
        self.model = TransE(entity_size = entity_size, relation_size = relation_size,
                            embedding_dim=embedding_dim, margin=margin,norm = norm)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if(load):
            load_checkpoint(path=r"checkpoint250.tar",model=self.model,optim = self.optimizer)
        self.model.train()

        head = torch.tensor([int(i[0]) for i in triples])
        rel = torch.tensor([int(i[1]) for i in triples])
        tail = torch.tensor([int(i[2]) for i in triples])
        self.triples = torch_data.TensorDataset(head,rel,tail)
        self.train_generator = torch_data.DataLoader(self.triples,batch_size=self.batch_size,shuffle = True)
        
    def train(self):
        for iter in range(self.max_iteration):
            self.model.train()
            for real_heads,real_relations,real_tails in self.train_generator:
                positive_triples = torch.stack((real_heads,real_relations,real_tails),dim=1)
                random_entities = torch.randint(high=self.entity_size, size=real_heads.size())
                
                negative_triples = torch.stack((real_heads, real_relations, random_entities), dim=1)
                

                self.optimizer.zero_grad()
                loss = self.model.forward(positive_triplets=positive_triples,negative_triplets=negative_triples)
                loss.mean().backward()
                self.optimizer.step()
            print(loss.mean())


if __name__=="__main__":
    triples = data_loader("./train.txt")
    test_triples = data_loader("./dev.txt")

    trainDict = makeTrainDict(triples)
    #while True:
    trainer = tranETrainer(triples=triples,entity_size=15000,relation_size=240,embedding_dim=100,margin=0.8,norm=2,
                    learning_rate=0.01,batch_size=64,max_iteration=2,load=False
                    )
    load_checkpoint("checkpoint.tar", trainer.model, trainer.optimizer)
    trainer.train()
    save_checkpoint(trainer.model, trainer.optimizer)
    hit = test(test_triples, trainer.model, 5, 15000)
    print(hit)
    # test_triples = data_loader(r"test.txt")
    # predict_print(triple_dict=trainDict,predict_triple=test_triples,model=trainer.model,hit_n=5,entity_size=15000)
        #break
        
