import argparse
import datetime
import torch
import sys
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models import resnet18
import argparse


def train(modified_model, num_epochs, criterion, optimizer, trainloader, savepath):
    print("Training...")
    epoch_train = []
    losses_train = []
    starting_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"Time at start of training is: {starting_time}")

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_num = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = modified_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_num += 1
            running_loss += loss.item()
            # print statistics

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss/batch_num:.3f}, Time: {timestamp}")
        #Collect data to plot after tests are done
        losses_train.append(running_loss/batch_num)
        epoch_train.append(epoch + 1)

    plt.figure()
    plt.plot(epoch_train, losses_train)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.savefig("./modified_loss_fig")
    plt.close()

    ending_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"Time at end of training is: {ending_time}")

    PATH = savepath
    torch.save(modified_model.state_dict(), PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')

    # Add arguments for -z, -e, -b, -s, and -p
    parser.add_argument('-z', "--bottleneck_size", type=int, default=8, help='Value for z')
    parser.add_argument('-e', "--epochs", type=int, default=50, help='Value for e')
    parser.add_argument('-b', "--batch_size", type=int, default=2048, help='Value for b')
    parser.add_argument('-s', "--savepath", type=str, default='MLP.8.pth', help='Value for s')
    parser.add_argument('-p', "--lossgraph", type=str, default='loss.MLP.8.png', help='Value for p')

    args = parser.parse_args()
    model = resnet18()
    batch_size = args.batch_size
    device = torch.device("cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # train_set is where the current issues lie
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = # idk yet
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    train(args.epochs, optimizer, model, loss_fn, train_loader, scheduler, device)


