import torch
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from data_aug.view_generator import ContrastiveLearningViewGenerator
from data_aug.visda import Visda 

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device = "cuda:1"
print("Using device:", device)

from models.resnet_simclr import ResNetSimCLR

model = ResNetSimCLR(base_model="resnet18", out_dim=128).to(device)

checkpoint = torch.load('runs/visda-train/checkpoint_epoch0099.pth.tar', map_location=device)
state_dict = checkpoint['state_dict']

test_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                     ])


log = model.load_state_dict(state_dict, strict=False)

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    output = model(input)

    return output.data.cpu().numpy()


dataset = Visda("visda", train=False)
stored_points = {}

for item, index in tqdm(dataset):
    try:
        output = predict_image(item)
        if index in stored_points:
            stored_points[index].append(output)
        else:
            stored_points[index] = [output]
    except:
        continue

torch.save(stored_points, "stored_features_test.data")


