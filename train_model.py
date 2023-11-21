import os
import torch
import argparse
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

class CustomDataset(Dataset):

    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.labels = []

        # Read the labels.txt file
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, 'labels.txt'), 'r') as file:
            for line in file:
                image_file, label, _ = line.strip().split()
                self.labels.append((image_file, int(label)))

    def __len__(self):
        return len(self.labels)



    def __getitem__(self, idx):

        image_file, label = self.labels[idx]
        image_path = os.path.join(self.directory, image_file)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def train(train_set, batch_size, num_epochs, pth, model):
    device = torch.device('cpu')
    print('using CPU')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.01)
    training_loss = []
    #changed
    all_pred_labels = []
    all_true_labels = []

    for epoch in range(num_epochs):
        model.train()
        batch_loss = 0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            #changed
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            # changed
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())
            optimizer.step()
            batch_loss += loss.item()
        epoch_loss = batch_loss / len(train_loader)
        current_time = datetime.now().strftime('%H:%M:%S')  # Get current timestamp
        print(f'Current Time: {current_time}, Epoch {epoch + 1} out of {num_epochs}, Loss: {epoch_loss}')
        training_loss.append(epoch_loss)
        # scheduler.step()
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.legend()

    plt.figure(figsize=(8,8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    torch.save(model.state_dict(), pth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for train")
    parser.add_argument("-train_set", required=True, default="/datasets/COCO100/", help="Path to the content images directory")
    parser.add_argument("-e", type=int, default=20, help="Number of training epochs")
    parser.add_argument("-b", type=int, default=32, help="Batch size for training")
    parser.add_argument("-s", required=True, help="Path to save the decoder model")
    parser.add_argument("-cuda", choices=("Y", "N"), default="N", help="Use CUDA for training (Y/N)")
    args = parser.parse_args()
    transform = transforms.Compose((
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(), # Random horizontal flip
        # transforms.RandomRotation(15), # Random rotation
        transforms.ToTensor(),
    ))

    train_set = CustomDataset(directory=args.train_set, transform=transform)
    resnet_model = resnet18(weights=None)
    # train_set, batch_size, num_epochs, cuda
    train(train_set, args.b, args.e, args.s, resnet_model)