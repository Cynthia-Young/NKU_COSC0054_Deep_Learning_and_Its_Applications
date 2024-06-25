import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc1_drop = nn.Dropout(0.2)        
        self.fc2 = nn.Linear(200, 150)
        self.bn2 = nn.BatchNorm1d(150)
        self.fc2_drop = nn.Dropout(0.2)        
        self.fc3 = nn.Linear(150, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc3_drop = nn.Dropout(0.2)       
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc1_drop(x)        
        x = F.relu(self.bn2(self.fc2(x))) 
        x = self.fc2_drop(x)        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc3_drop(x)        
        return F.log_softmax(self.fc4(x), dim=1)
    
    
def train(model, optimizer, criterion, epoch, log_interval=200):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad() 
        output = model(data)

        loss = criterion(output, target)
        loss.backward()   
        optimizer.step()    #  w - alpha * dL / dw
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            

def validate(model, criterion, loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    
    batch_size = 32

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    validation_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    
    for (X_train, y_train) in train_loader:
        print('X_train:', X_train.size(), 'type:', X_train.type())
        print('y_train:', y_train.size(), 'type:', y_train.type())
        break

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 20

            
    lossv, accv = [], []
    
    for epoch in range(1, epochs + 1):
        train(model, optimizer, criterion, epoch)
        validate(model, criterion, lossv, accv)
        
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), lossv)
    plt.title('validation loss')
    plt.savefig('Modified_MLP_validation_loss')

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), accv)
    plt.title('validation accuracy')
    plt.savefig('Modified_MLP_validation_accuracy')