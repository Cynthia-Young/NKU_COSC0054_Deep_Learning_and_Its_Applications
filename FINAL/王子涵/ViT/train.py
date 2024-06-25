import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
from vit import VisionTransformer

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
doc = open("ViT.txt", 'w')

learning_rate = 0.01
batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

classes = [str(i) for i in range(100)]

net = VisionTransformer(image_size=32, patch_size=4, num_classes=100, num_heads=1, num_layers=8, use_conv_patch=True,
                        use_conv_stem=False)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

loss_x = []
loss_y = []
acc_x = []
acc_y = []

def train(epoch):
    best_test_acc = 0
    for e in range(epoch):
        print("Epoch: %d, Learning rate: %f" % (e + 1, optimizer.param_groups[0]['lr']))
        print("Epoch: %d, Learning rate: %f" % (e + 1, optimizer.param_groups[0]['lr']), file=doc)
        # ----------------------------------------------------------------------------------------------------------------------
        net.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            predict = torch.max(outputs, 1)[1].data.squeeze()
            accuracy = (predict == labels).sum().item() / labels.size(0)

            loss_x.append(e * len(trainloader) + i)
            loss_y.append(loss.item())
            if i % 200 == 0:
                print('Epoch: %d, [%6d, %6d], Train loss: %.4f, Train accuracy: %.4f ' % (
                    e + 1, (i + 1) * batch_size, len(trainset), loss.item(), accuracy))
                print('Epoch: %d, [%6d, %6d], Train loss: %.4f, Train accuracy: %.4f ' % (
                    e + 1, (i + 1) * batch_size, len(trainset), loss.item(), accuracy), file=doc)
        scheduler.step()
        net.eval()
        n = 0
        sum_acc = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            predict = torch.max(outputs, 1)[1].data.squeeze()
            sum_acc += (predict == labels).sum().item() / labels.size(0)
            n += 1
        test_acc = sum_acc / n
        acc_x.append(e + 1)
        acc_y.append(test_acc)
        print('Epoch: %d, Test accuracy: %.4f ' % (e + 1, test_acc))
        print('Epoch: %d, Test accuracy: %.4f ' % (e + 1, test_acc), file=doc)
        print("# ---------------------------------------------------- #")
        print("# ---------------------------------------------------- #", file=doc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net, './model/ViT.pth')
    print("Finished!")
    print("Finished!", file=doc)
    print('The best test accuracy is %.4f' % (best_test_acc))
    print('The best test accuracy is %.4f' % (best_test_acc), file=doc)


starttime = datetime.datetime.now()
train(200)
endtime = datetime.datetime.now()
ti = (endtime - starttime).seconds
hou = ti / 3600
ti = ti % 3600
sec = ti / 60
ti = ti % 60
print('Time expended: %dh-%dm-%ds' % (hou, sec, ti))
print('Time expended: %dh-%dm-%ds' % (hou, sec, ti), file=doc)
print('\n')

plt.figure()
plt.plot(loss_x, loss_y)
plt.title("Train Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("Loss.jpg")
plt.show()

plt.figure()
plt.plot(acc_x, acc_y)
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("Accuracy.jpg")
plt.show()
