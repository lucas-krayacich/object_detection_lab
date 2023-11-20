import argparse
import datetime
import torch
import sys
import numpy as np
from KittiDataset import KittiDataset
from torch.utils.data import DataLoader, Dataset
from KittiAnchors import Anchors
import cv2
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import os

import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models import resnet18
import argparse
import KittiAnchors

#  input
def resize_dataset(dataset, target_size_x, target_size_y):
    transform = transforms.Compose([
        transforms.Resize((target_size_x, target_size_y)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    resized_dataset = []

    for i in range(len(dataset)):
        sample, label = dataset[i]

        # If the sample is a NumPy array, convert it to a PIL Image
        if isinstance(sample, np.ndarray):
            sample = Image.fromarray(sample)

        # Check if the sample is already a tensor
        if not isinstance(sample, torch.Tensor):
            sample = transform(sample)

        resized_dataset.append((sample, label))

    return resized_dataset

def load_labels(labels_path):
    labels_list = []
    class_names_list = []

    with open(labels_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            label = int(line[1])  # Assuming the label is an integer
            class_name = line[2]

            labels_list.append(label)
            class_names_list.append(class_name)

    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    class_names_tensor = torch.tensor(class_names_list, dtype=torch.str)

    return labels_tensor, class_names_tensor

class KittiCustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.labels = []

        with open(os.path.join("./data/Kitti8_ROIs/train", 'labels.txt'), 'r') as file:
            for line in file:
                image_file, label, _ = line.strip().split()
                self.labels.append(("00000" + image_file[0] + ".png", int(label)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image_file, label = self.labels[i]
        image_path = os.path.join(self.directory, image_file)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def train(modified_model, num_epochs, criterion, optimizer, trainloader, savepath):
    print("Training...")
    epoch_train = []
    losses_train = []
    starting_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"Time at start of training is: {starting_time}")
    labels_path = './data/Kitti8_ROIs/train/labels.txt'

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_num = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            print(inputs.size())
            # use YODA labels - not kitti labels
            #print(inputs)
            print("-------------------------------")
            print(labels)
            # Create an instance of the Anchors class
            anchors = Anchors()
            for image in inputs:
                # Calculate anchor centers for a 4x12 grid
                anchor_centers = anchors.calc_anchor_centers(image.shape[:2], Anchors.grid)
                # Get anchor ROIs
                anchor_ROIs, anchor_boxes = anchors.get_anchor_ROIs(image, anchor_centers, Anchors.shapes)
                ###################################################
                #plt.figure(figsize=(12, 4))
                #plt.subplot(1, 5, 1)
                #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                #plt.title('Original Image')
                ###################################################
                #print("Show")
                #plt.show()
                for region in enumerate(anchor_ROIs, 0):
                    print(region)
                    anchor_inputs = anchor_ROIs[region]
                    # anchor each input into 48
                    # run model on each anchor zone
                    # zero the parameter gradients
                    plt.subplot(1, 5, i + 2)
                    plt.imshow(cv2.cvtColor(anchor_ROIs[i], cv2.COLOR_BGR2RGB))
                    plt.title(f'Anchor {i + 1}')

                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = modified_model(anchor_inputs)
                    loss = criterion(outputs, labels[region])
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


class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        self.data = KittiDataset(dir=data_path, transform=transform)
        self.labels = self.load_labels(labels_path)
        self.transform = transform

    def load_labels(path):
        labels_list = []
        class_names_list = []

        with open(path, 'r') as file:
            for line in file:
                line = line.strip().split()
                label = int(line[1])  # Assuming the label is an integer
                class_name = line[2]

                labels_list.append(label)
                class_names_list.append(class_name)

        labels_tensor = torch.tensor(labels_list, dtype=torch.int) #used to be 'long'
        class_names_tensor = torch.tensor(class_names_list, dtype=torch.str)

        return labels_tensor, class_names_tensor
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')

    # Add arguments for -z, -e, -b, -s, and -p

    parser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
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

    # set the x and y values that we want the images to be at
    desired_y = 1242
    desired_x = 375

    #Create the transform and load the train set
    train_transform = transforms.Compose(
        [transforms.Resize((desired_x, desired_y)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = KittiCustomDataset(args.i, train_transform)
    #train_set = KittiDataset(args.i, training=True, transform=train_transform)

    #Resize the train set to make sure that all of the images are uniform
    #print("Resizing...")
    #resized_train_set = resize_dataset(train_set, desired_x, desired_y)
    #print("Resizing Finished.")

    #Link the resized train set to the train loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

    train(num_epochs=args.epochs, optimizer=optimizer, modified_model=model, criterion=loss_fn, trainloader=train_loader, savepath=args.savepath)


