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

class CustomTestDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.labels = []

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

def test(test_set, model, num_classes, trained_model_path, batch_size=32):
    device = torch.device('cpu')
    print(f"Using {device}")

    # Modify the last fully connected layer to match the number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the trained model weights
    model.load_state_dict(torch.load(trained_model_path))

    model.to(device)
    model.eval()

    all_pred_labels = []
    all_true_labels = []

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())

    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)

    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    torch.save(model.state_dict(), "test_confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for test")
    parser.add_argument("-test_set", required=True, default="/datasets/COCO100/",
                        help="Path to the content images directory")
    parser.add_argument("-b", type=int, default=32, help="Batch size for testing")
    parser.add_argument("-w", required=True, default="N", help="Weight path")
    args = parser.parse_args()

    # Set the path to your test dataset
    test_dataset_directory = args.test_set

    transform = transforms.Compose((
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(), # Random horizontal flip
        # transforms.RandomRotation(15), # Random rotation
        transforms.ToTensor(),
    ))

    # Create a test dataset
    test_set = CustomTestDataset(directory=test_dataset_directory, transform=transform)

    # Initialize the ResNet model
    resnet_model = resnet18(weights=None)

    # Set the number of classes (adjust as needed)
    num_classes = 2

    # Test the model on the provided test dataset
    true_labels, pred_labels = test(test_set, resnet_model, num_classes, trained_model_path=args.w)

    # Evaluate or visualize the results as needed
    # For example, you can use sklearn.metrics for evaluation
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(true_labels, pred_labels)
    classification_rep = classification_report(true_labels, pred_labels)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", classification_rep)
