import os
import argparse
from tqdm import tqdm
from resnet import *
from densenet import *
from mobilenets import *

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

def data_load(batch_size):
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)
    return trainloader, testloader

def train(trainloader, epoch):
    net.train()
    train_tqdm = tqdm(trainloader, desc="Epoch " + str(epoch))
    running_loss = 0.0
    for i, data in enumerate(train_tqdm):
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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-learning_rate", default=0.001, type=int)
    parser.add_argument("-epochs", default=20, type=int)
    parser.add_argument("-model", default="ResNet34", type=str)
    args = parser.parse_args()
    
    trainloader, testloader = data_load(args.batch_size)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
      
    if args.model == 'ResNet50':
        net = resnet50(num_classes=10).to(device)
    elif args.model == 'ResNet34':
        net = resnet34(num_classes=10).to(device)        
    elif args.model == 'ResNet101':
        net = resnet101(num_classes=10).to(device)   
    elif args.model == 'DenseNet121':
        net = DenseNet121().to(device)
    elif args.model == 'MobileNets':
        net = MobileNetV1(3, 10).to(device)
    
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    
    lossv, accv = [], []
    
    for epoch in range(1, args.epochs + 1):
        train(trainloader, epoch)
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)
        with torch.no_grad():
            validate(testloader, epoch, lossv, accv)
            
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, args.epochs + 1), lossv)
    plt.title(args.model +' validation loss')
    plt.savefig(os.path.join(args.model +'_loss'))

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, args.epochs + 1), accv)
    plt.title(args.model +' validation accuracy')
    plt.savefig(os.path.join(args.model +'_accuracy'))
    