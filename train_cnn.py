from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import time
import numpy as np
import os
from torch.autograd import Variable
from scipy.io import loadmat
from NetModels import *
from cpc import cpc

rdata = loadmat('data/slpdb_cnn_dataset.mat')

test_flg = True
bs = 128
total_epoch = 100
learning_rate = 0.01
best_val_acc = 0  
best_val_acc_epoch = 0
print(torch.cuda.get_device_name(0))
print('==> Preparing data..')

transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])
transform_test = transforms.Compose([
        transforms.ToTensor()])

trainset = cpc(rdata, split='train',transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=1)
val_set = cpc(rdata, split='val',transform=transform_test)
valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=1)
test_set = cpc(rdata, split='test', transform=transform_test)
testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=1)

net = LiNet()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
data_length = len(trainset)

if test_flg:
    checkpoint = torch.load('slpstaging.v1')
    net.load_state_dict(checkpoint['net'])

def train(epoch):
    epoch_start = time.time()
    print('\nEpoch: %d' % epoch)
    global Train_acc
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx %5 == 0 and batch_idx > 0:
            print(str(batch_idx) + ' batch out of ' + str(data_length // bs) + ' elapse time since last epoch: ' +
                str(time.time() - epoch_start) + ' training loss: ' + str((train_loss / (batch_idx + 1))))
        targets = torch.tensor(np.squeeze(np.array(targets)), dtype=torch.long)
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        
    Train_acc = 100. * correct / total
    print('Training ACC: ' + str(Train_acc))

def validate(epoch):
    global val_acc
    global best_val_acc
    global best_val_acc_epoch
    
    val_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(valloader):
        targets = torch.tensor(np.squeeze(np.array(targets)), dtype=torch.long)
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        val_loss += loss.cpu().__array__().mean()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    val_acc = 100. * correct / total
    print('Val ACC: ' + str(val_acc) + ' loss: ' + str(val_loss / (batch_idx + 1)))
    # Save checkpoint.
    val_acc = 100. * float(correct) / total
    if val_acc > best_val_acc:
        print('Saving..')
        print("best_val_acc: %0.3f" % val_acc)
        state = {
            'net': net.state_dict(),
            'acc': val_acc,
            'epoch': epoch,
        }
        torch.save(state,'slpstaging.v1')
        best_val_acc = val_acc
        best_val_acc_epoch = epoch

def testing():
    global test_acc
    
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        targets = torch.tensor(np.squeeze(np.array(targets)), dtype=torch.long)
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    val_acc = 100. * correct / total
    print('testing ACC: ' + str(val_acc))

if test_flg:
    testing()
else:
    for epoch in range(total_epoch):
        train(epoch)
        validate(epoch)

print("best_val_acc: %0.3f" % best_val_acc)
print("best_val_acc_epoch: %d" % best_val_acc_epoch)