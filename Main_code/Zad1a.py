from __future__ import print_function, division


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pylab import *
import time
import os
import copy
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt


model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
model.eval()

#Zadanie 1a
for param in model.features.parameters():
   param.requires_grad = False



data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

flowers_dataset = datasets.ImageFolder(root='C:/Users/Aleksandra/PycharmProjects/DataSet/flowers',
                                       transform=data_transform)

dataset_size = len(flowers_dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))

shuffle_dataset = True
if shuffle_dataset :
    np.random.seed(41)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
#
dataset_size = len(flowers_dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.3 * dataset_size))

shuffle_dataset = True
if shuffle_dataset :
    np.random.seed(41)
    np.random.shuffle(indices)
new_set =  indices[:split]

#
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(flowers_dataset, batch_size=4,
                                           sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(flowers_dataset, batch_size=4,
                                                sampler=valid_sampler)
class_names = flowers_dataset.classes


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 5)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

run_losses = []
epoch_accuracy = []

dataiter = iter(validation_loader)
images, labels = dataiter.next()


print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)
for epoch in range(10):
    running_loss = 0.0
    # scheduler.step(epoch)
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for i, data in enumerate(train_loader,0):
        #data_, target_ = data_.to(device), target_.to(device)# on GPU
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==labels).item()
        total += labels.size(0)
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        model.eval()
        for data in validation_loader:
            #data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss_t = criterion(outputs, labels)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs, dim=1)
            correct_t += torch.sum(pred_t==labels).item()
            total_t += labels.size(0)
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss/len(validation_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        # Saving the best weight 
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'model_classification_tutorial.pt')
            print('Detected network improvement, saving current model')
    model.train()

print('Finished Training')

PATH = 'C:/Users/Aleksandra/Desktop/cifar_net.pth'
torch.save(model.state_dict(), PATH)

fig = plt.figure(figsize=(20,10))
plt.title("Train - Validation Loss")
plt.plot( train_loss, label='train')
plt.plot( val_loss, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')



fig1 = plt.figure(figsize=(20,10))
plt.title("Train - Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')

plt.show()

