import argparse
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler

from wdyresnet import *

best_prec1 = 0
best_epoch = 0

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

# 标签平滑 均值方差transform  batch_size


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss
    

def main(args):
    global best_prec1, best_epoch
    
    cudnn.benchmark = True

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])


    train_set = datasets.CIFAR100('../datasets', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100('../datasets', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    CIFAR100_class = 100
    # model = res2net50(num_classes=CIFAR100_class).to(device)
    model = ResidualNet("CIFAR100",
                            depth=50,
                            num_classes=100,
                            att_type='CA',
                            activation='relu',
                            replace_sigmoid='sigmoid',
                            pool_type=0).to(device)
    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))
    
    criterion = CrossEntropyLabelSmooth(num_classes=CIFAR100_class, epsilon=args.smooth)

    MILESTONES = [60, 120, 160]
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
    
    

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    

    for epoch in range(args.start_epoch, args.epochs):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        # adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, warmup_scheduler)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            best_epoch = epoch + 1
        print(' * BestPrec so far@1 {top1:.3f} in epoch {best_epoch}'.format(top1=best_prec1, best_epoch=best_epoch))


def train(args, train_loader, model, criterion, optimizer, epoch, warmup_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        
        data_time.update(time.time() - end)

        output = model(input)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if epoch <= args.warm:
            warmup_scheduler.step()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i, len(val_loader), 
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# def adjust_learning_rate(optimizer, epoch, gammas, schedule):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr
#     assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
#     for (gamma, step) in zip(gammas, schedule):
#         if (epoch >= step):
#             lr = lr * gamma
#         else:
#             break
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.reshape((batch_size, -1))  # Reformat the output of topk to match the shape of input tensor
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--workers", default=15, type=int, metavar="N", help="number of data loading workers (default: 4)",)
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run default =10")
    parser.add_argument("--start_epoch", default=0,type=int, metavar="N", help="manual epoch number (useful on restarts)",)
    parser.add_argument("-b", "--batch_size", default=128, type=int, metavar="N", help="mini-batch size (default: 64)",)
    parser.add_argument("--lr", "--learning_rate", default=0.05, type=float, metavar="LR", help="initial learning rate default = 0.1",)
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument("--weight_decay", "--wd", default=0.0005, type=float, metavar="W", help="weight decay (default: 1e-4)",)
    parser.add_argument("--print_freq", "-p", default=10, type=int, metavar="N", help="print frequency (default: 10)",)
    parser.add_argument("--seed", type=int, default=2024, metavar="BS", help="Selected Seed (default: 2024)",)
    parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluation only")
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args)