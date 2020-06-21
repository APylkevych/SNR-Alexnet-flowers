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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
# Zadanie 1a
 for param in model.features.parameters():
    param.requires_grad = False
"""

"""
# Zadanie 2a
for layers in range(9):
    for param in model.features[layers].parameters():
       param.requires_grad = False
"""


# Zadanie 2b
for layers in range(7):
    for param in model.features[layers].parameters():
        param.requires_grad = False


"""
# Zadanie 2c
    
# Pozostawić pozostałe zadani zakomentowane

"""



# Zadanie 2d
"""
PATH = 'C:/Users/pawel/Desktop/cifar_netdeep_2c.pth'
num_fts = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_fts, 5)
model = model.to(device)
model.load_state_dict(torch.load(PATH))
model.eval()
print(model)
model.features = nn.Sequential(*list(model.features.children())[:-3])
print(model)
"""

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

flowers_dataset = datasets.ImageFolder(root='C:/Users/pawel/PycharmProjects/SNR/flowers/',
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


model = model.to(device)

run_losses = []
epoch_accuracy = []

dataiter = iter(validation_loader)
images, labels = dataiter.next()

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    guesses_correct = 0.0
    guesses_total = 0.0

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
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

    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            guesses_total += labels.size(0)
            guesses_correct += (predicted == labels).sum().item()

    epoch_accuracy.append(100 * guesses_correct/guesses_total)
    run_losses.append(100 * running_loss/guesses_total)
    print(running_loss)

    if (epoch >= 9):
        print(epoch_accuracy[epoch])


print('Finished Training')

PATH = 'C:/Users/pawel/Desktop/cifar_netdeep_2c.pth'
torch.save(model.state_dict(), PATH)




plt.plot(epoch_accuracy , label = "accuracy")
plt.plot(run_losses, label = "running losses")
plt.xlabel('epochs')
plt.ylabel('%')
plt.legend()
plt.show()


