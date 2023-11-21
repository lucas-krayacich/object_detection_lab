import os
import torch
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from KittiDataset import KittiDataset
from KittiAnchors import Anchors
import random
import torch.nn.functional as F
import cv2

def test_yoda(test_set, trained_model_path, batch_size):
    # initialize Yoda
    print('Testing YODA...')
    device = torch.device('cpu')
    num_classes = 2
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(trained_model_path))
    #model.conv1.weight = torch.nn.Parameter(torch.randn(64, 150, 7, 7))
    model.to(device)
    model.eval()

    # initialize transform
    transform = transforms.Compose((
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(), # Random horizontal flip
        # transforms.RandomRotation(15), # Random rotation
        transforms.ToTensor(),
    ))

    # subdivide Kitti image into set of ROIs using KittiAnchors.py
    kitti_data = KittiDataset(dir=test_set, training=False, transform=transform)
    anchors = Anchors()
    # dataset_length = len(kitti_data)
    # Generate a random index
    # random_index = random.randint(0, dataset_length - 1)
    # item = kitti_data[random_index]
    # idx = item[0]
    # image = item[1][0]
    # label = item[1][1]
    random_data_point = random.choice(kitti_data)

    image = random_data_point[0]
    label = random_data_point[1]
    idx = kitti_data.class_label['Car']
    # car_ROIs = kitti_data.mod_strip_ROIs(class_ID=idx, label_list=label)
    car_ROIs = kitti_data.strip_ROIs(class_ID=idx, label_list=label)

    anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)

    show_images = True

    if show_images:
        image1 = image.copy()
        for j in range(len(anchor_centers)):
            x = anchor_centers[j][1]
            y = anchor_centers[j][0]
            cv2.circle(image1, (x, y), radius=4, color=(255, 0, 255))
            cv2.imshow('image', image1)

    ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
    print("ROIS Shape: ", ROIs[0].shape)
    #print("BOXES: ", boxes)

    plt.imshow(ROIs[0])
    plt.show()

    IoU_threshold = 0.02
    ROI_IoUs = []
    for idx in range(len(ROIs)):
        ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], ROIs[idx])]

    combined_roi_box = zip(ROIs, ROI_IoUs)
    # Create batch of ROIs
    output_batch = []
    num_cars = 0
    for roi, iou in combined_roi_box:
        input_tensor = torch.tensor(roi, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0)
        #print(input_tensor.shape)
        with torch.no_grad():
            #Adjust the size of the input and input the roi tensor into the model
            num_channels = input_tensor.shape[1]
            model.conv1.weight = torch.nn.Parameter(torch.randn(64, num_channels, 7, 7))
            output = model(input_tensor)

            probabilities = F.softmax(output, dim=1)

            prob_class_1 = probabilities[0][1].item()
            print(probabilities)
            if prob_class_1 < IoU_threshold:
                #print("Classified as a car")
                output_batch.append(int(1))
                num_cars += 1
            else:
                #print("Not a car...")
                output_batch.append(int(0))
            #print("Output is: " + output)
            #if output == 1:
            #    print("Classified as car...")
            #    iou_score = calc_IoU(box, image)
            #    print("IOU Score: ", iou_score)
            #else:
            #    print("IDK")

    # Pass batch through Yoda Classifier
    print("The Labels would be: ")
    print(output_batch)
    print("The number of cars would be: " + str(num_cars))

    image_with_boxes = image.copy()

    for idx in range(len(output_batch)):
        if output_batch[idx] == 1:
            box = boxes[idx]
            cv2.rectangle(image_with_boxes, (box[0][1], box[0][0]), (box[1][1], box[1][0]), (255, 0, 255), 3)

    plt.imshow(image_with_boxes)
    plt.show()
    # For each ROI classified as 'Car' calculate its IoU score against original Kitti image


def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for test")
    parser.add_argument("-test_set", required=True, default="/datasets/COCO100/",
                        help="Path to the content images directory")
    parser.add_argument("-b", type=int, default=32, help="Batch size for testing")
    parser.add_argument("-w", required=True, default="N", help="Weight path")
    args = parser.parse_args()

    test_yoda(
        test_set=args.test_set,
        batch_size=args.b,
        trained_model_path=args.w
    )