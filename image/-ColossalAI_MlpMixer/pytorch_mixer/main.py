import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
from MlpMixer.mixer import mixer_s32


torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
	prog='ColossalAI demo file.',
	usage='Demo with MLP-mixer.',
	description='Accelerate the training of MLP-mixer through colossalai.',
	epilog='end',
	add_help=True)


parser = argparse.ArgumentParser()
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-3)
parser.add_argument('-bs', help='batch size', type=int, default=4)
parser.add_argument('-ep', help='number of epochs', type=int, default=10)
parser.add_argument('-num_classes', help='num_classes', type=int, default=10)
parser.add_argument('-image_size', help='image_size', type=int, default=32)
parser.add_argument('-patch_size', help='patch_size', type=int, default=4)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)


train_dataset = torchvision.datasets.CIFAR10(
    root= './data',
    train= True,
    transform= transform,
    download= True
)

train_loader = torch.utils.data.DataLoader(
    dataset= train_dataset,
    batch_size= args.bs,
    shuffle= True
)
test_dataset = torchvision.datasets.CIFAR10(
    root= './data',
    train= False,
    transform= transform,
    download= True
)
test_loader = torch.utils.data.DataLoader(
    dataset= test_dataset,
    batch_size= args.bs,
    shuffle= True
)

net = mixer_s32(args.num_classes,args.image_size,args.patch_size)

net.to(device)
print(net.parameters)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)


for epoch in range(args.ep):
    print('epoch = %d' % epoch)
    for i, (batch_x, batch_y) in enumerate(train_loader):

        x = net(batch_x)
        loss = loss_function(x, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('loss = %.5f' % loss)


total = 0
correct = 0
net.eval()
for batch_x, batch_y in test_loader:
    x = net(batch_x)
    _, prediction = torch.max(x, 1)
    total += batch_y.size(0)
    correct += (prediction == batch_y).sum()
print('There are ' + str(correct.item()) + ' correct numbers.')
print('Accuracy=%.2f' % (100.00 * correct.item() / total))
