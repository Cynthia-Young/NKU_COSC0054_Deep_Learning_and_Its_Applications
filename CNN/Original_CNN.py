import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import argparse
from tqdm import tqdm

    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 4, 6, 28, 28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
#         x = self.conv1(x)
#         print("output shape of conv1:", x.size())
#         x = F.relu(x)
        
#         x = self.pool(x)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def data_load(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=12)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=12)
    return trainloader, testloader

def train(trainloader, epoch):
    net.train()
    # train_tqdm = tqdm(trainloader, desc="Epoch " + str(epoch))
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

        # train_tqdm.set_postfix({"loss": "%.3g" % loss.item()})
    
def validate(testloader, epoch, loss_vector, accuracy_vector):
    net.eval()
    val_loss, correct, total = 0, 0, 0
    
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1) 
        
        total += labels.size(0)
        val_loss += criterion(outputs, labels.to(device)).data.item()
        correct += (predicted == labels).sum().item()
        
    val_loss /= total
    loss_vector.append(val_loss)
    
    accuracy = 100. * correct / total
    accuracy_vector.append(accuracy)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default="BaseNet", type=str, help="Resnet Base Densenet")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    batch_size = 4
    epochs = 5
    
    trainloader, testloader = data_load(batch_size)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    lossv, accv = [], []
    
    for epoch in range(1, epochs + 1):
        train(trainloader, epoch)
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)
        with torch.no_grad():
            validate(testloader, epoch, lossv, accv)
            
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), lossv)
    plt.title(args.model +' validation loss')
    plt.savefig(os.path.join(args.model +'_loss'))

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), accv)
    plt.title(args.model +' validation accuracy')
    plt.savefig(os.path.join(args.model +'_accuracy'))
    