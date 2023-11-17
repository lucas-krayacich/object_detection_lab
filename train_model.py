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
#arg parse


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training...')
    model.train() #keep track of gradient for backtracking
    losses_train = []

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs,lbls in train_loader:
            imgs = imgs.to(device=device)

            batch_size = imgs.size(0)  # Get the batch size
            imgs = imgs.view(batch_size, -1)  # Reshape to (batch_size, num_features)

            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))
        plt.figure()
        plt.plot(range(1, epoch+1), losses_train)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss vs Epoch')
        plt.savefig(args.lossgraph)
        plt.close()

        save_path = args.savepath
        torch.save(model.state_dict(), save_path)

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


