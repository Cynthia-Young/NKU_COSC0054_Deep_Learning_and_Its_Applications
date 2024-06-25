from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
 
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
 
        self.drop_rate = drop_rate
 
    def forward(self, x):
        out = torch.cat(x, 1)
        out = self.conv1(self.relu1(self.norm1(out))) 
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate>0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out
 

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer %d" %(i+1), layer)
    
    def forward(self, x):
        x = [x]
        for name, layer in self.items():
            out = layer(x)
            x.append(out)
        return torch.cat(x, 1)

class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32,block_config=(6,12,24,26), num_init_features=64,bn_size=4, comparession_rate=0.5, drop_rate=0,num_classes=10):
        super(DenseNet,self).__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ('conv0',nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(3,stride=2,padding=1))
        ]))
        #Denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
 
            self.features.add_module("denseblock%d" %(i+1), block)
            num_features +=num_layers*growth_rate
 
            if i!=len(block_config)-1:
                transition = Transition(num_features,int(num_features*comparession_rate))
                self.features.add_module("transition%d" %(i+1), transition)
                num_features = int(num_features*comparession_rate)
 
        #Final bn+ReLu
        self.features.add_module('norm5',nn.BatchNorm2d(num_features))
 
        #classification layer
        self.classifier = nn.Linear(num_features,num_classes)
 
        #params initalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def forward(self,x ):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
 
def DenseNet121():
    return DenseNet(num_init_features=32, growth_rate=32, block_config=(6, 12, 24, 16))

def DenseNet169():
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32))

def DenseNet201():
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32))

def DenseNet161():
    return DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24))