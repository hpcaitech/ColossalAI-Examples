import numpy as np
import argparse
import random
from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger

colossalai.launch_from_torch(config='./config.py')
logger = get_dist_logger()

# prepare arguments (hyperparameters)
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='music',
                    help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum',
                    help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=gpc.config.NUM_EPOCHS,
                    help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int,
                    default=gpc.config.Neighbor_Sample_Size, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16,
                    help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=gpc.config.N_Iter,
                    help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int,
                    default=gpc.config.BATCH_SIZE, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4,
                    help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=gpc.config.Ir, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8,
                    help='size of training dataset')

args = parser.parse_args(['--l2_weight', '1e-4'])

# build dataset and knowledge graph
data_loader = DataLoader(args.dataset)
kg = data_loader.load_kg()
df_dataset = data_loader.load_dataset()

# Dataset class
class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label
    

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    df_dataset, df_dataset['label'], test_size=1 - args.ratio, shuffle=False, random_state=999)
train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size)

# prepare network, loss function, optimizer
num_user, num_entity, num_relation = data_loader.get_num()
user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr,
                       weight_decay=args.l2_weight)

engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
    net, optimizer, criterion, train_loader, test_loader)


# train
loss_list = []
test_loss_list = []
auc_score_list = []

for epoch in range(args.n_epochs):
    engine.train()
    running_loss = 0.0
    for i, (user_ids, item_ids, labels) in enumerate(train_dataloader):
        user_ids, item_ids, labels = user_ids.to(
            device), item_ids.to(device), labels.to(device)
        engine.zero_grad()
        outputs = engine(user_ids, item_ids)
        loss = engine.criterion(outputs, labels)
        engine.backward(loss)

        engine.step()

        running_loss += loss.item()

    # print train loss per every epoch
    logger.info(f'[Epoch {epoch + 1}]train_loss: {running_loss / len(train_loader)}')
    loss_list.append(running_loss / len(train_loader))

    engine.eval()
    # evaluate per every epoch
    with torch.no_grad():
        test_loss = 0
        total_roc = 0
        for user_ids, item_ids, labels in test_dataloader:
            user_ids, item_ids, labels = user_ids.to(
                device), item_ids.to(device), labels.to(device)
            outputs = engine(user_ids, item_ids)
            test_loss += engine.criterion(outputs, labels).item()
            total_roc += roc_auc_score(labels.cpu().detach().numpy(),
                                       outputs.cpu().detach().numpy())
        logger.info(f'[Epoch { epoch+1}] test_loss: {test_loss / len(test_loader)}')
        test_loss_list.append(test_loss / len(test_loader))
        auc_score_list.append(total_roc / len(test_loader))

# plot losses / scores
