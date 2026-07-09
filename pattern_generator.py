#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:22:13 2026

@author: haileyvan
"""

''' my sewing code for pattern generation '''

import os
import json
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.models import resnet18, ResNet18_Weights


# ------------------------------
# 1. Dataset class
# ------------------------------

class ShirtPatternDataset(Dataset):
    def __init__(self, image_dir, label_dir, garment_type, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.garment_type = garment_type
        self.transform = transform

        # Only keep image files
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Example: shirt_001.jpg -> shirt_001.json
        label_name = os.path.splitext(image_name)[0] + ".json"
        label_path = os.path.join(self.label_dir, label_name)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load JSON label
        with open(label_path, "r") as f:
            label = json.load(f)

        # Convert JSON measurements into one flat list of numbers
        measurements = [
            label["front"]["body_length_mm"],
            label["front"]["shoulder_width_mm"],
            label["front"]["chest_width_mm"],
            label["front"]["waist_width_mm"],
            label["front"]["hem_width_mm"],
            label["front"]["side_length_mm"],
            label["front"]["strap_width1_mm"],
            label["front"]["strap_width2_mm"],
            label["front"]["strap_length_mm"],
        ]

        garment_type = torch.tensor(self.garment_type,dtype=torch.long)
        measurements = torch.tensor(measurements, dtype=torch.float32) / 1000.0
        
        # Apply image transform
        if self.transform:
            image = self.transform(image)
        
        if self.garment_type == 0:
            mask = torch.tensor([
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
                1, 1, 1,
                1, 1
            ], dtype=torch.float32)
        
        else:
            mask = torch.tensor([
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
                0, 0, 0,
                1, 1
            ], dtype=torch.float32)
        
        return image, measurements, garment_type, mask


shirt_dir = "synthetic_dataset/images_shirt"
tank_dir = "synthetic_dataset/images_tank"
shirt_label_dir = "synthetic_dataset/labels_shirt"
tank_label_dir = "synthetic_dataset/labels_tank"

    
class GarmentPatternModel(nn.Module):
    def __init__(self, num_outputs=17, num_garment_types=2):
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        num_image_features = self.resnet.fc.in_features

        # Remove the normal ResNet classification layer
        self.resnet.fc = nn.Identity()
        
        #t-shirt specific prediction head
        self.shirt_head = nn.Sequential(nn.Linear(num_image_features, 128), nn.ReLU(), nn.Linear(128, num_outputs), nn.Softplus())
        
        self.tank_head = nn.Sequential(nn.Linear(num_image_features, 128), nn.ReLU(), nn.Linear(128, num_outputs), nn.Softplus())

        self.regression_head = nn.Sequential(
            nn.Linear(num_image_features + 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
            nn.Softplus())

    def forward(self, images, garment_types):
        image_features = self.resnet(images)
        shirt_predictions = self.shirt_head(image_features)
        tank_predictions = self.tank_head(image_features)
        garment_types = garment_types.view(-1,1)
        predictions = torch.where(garment_types ==0, shirt_predictions, tank_predictions)
        return predictions


# ResNet normally predicts 1000 ImageNet classes.
num_outputs = 9
model = GarmentPatternModel(num_outputs=num_outputs, num_garment_types=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

'''CREATE DISTINCT DATASETS'''
dataset = ShirtPatternDataset(
    image_dir=shirt_dir,
    label_dir=shirt_label_dir,garment_type=0,
    transform=transform)

dataset2 = ShirtPatternDataset(
    image_dir=tank_dir,
    label_dir=tank_label_dir, garment_type=1,
    transform=transform)

combined_dataset = ConcatDataset([dataset, dataset2])

'''TRAINING MODEL'''

train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size

train_dataset, val_dataset = random_split(
    combined_dataset,
    [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)

shirt_mask = torch.tensor([
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1,
    1, 1
], dtype=torch.float32)

tank_mask = torch.tensor([
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0,
    0, 0
], dtype=torch.float32)

def pattern_loss(predictions, targets, masks):
    squared_error = (predictions - targets) ** 2
    measurement_loss = (squared_error * masks).sum() / masks.sum().clamp_min(1.0)
    # Example indices
    shoulder_width = predictions[:, 1]
    waist_width = predictions[:, 3]
    body_length = predictions[:, 0]
    side_length = predictions[:, 5]
    left_strap = predictions[:, 6]
    right_strap = predictions[:, 7]

    '''HIGH VALUE ERROR'''
    side_length = body_length-side_length
    if side_length >= 0:
        side_penalty = torch.mean(side_length**2)
    
    strap_difference = left_strap - right_strap 
    if strap_difference != 0: 
        strap_penalty = torch.mean((abs(strap_difference))**2)
    
    '''LOW VALUE ERRORS'''
    waist_shoulder_diff = waist_width-shoulder_width
    if waist_shoulder_diff >=0: 
        sizing_issue = torch.mean((waist_shoulder_diff)**2)
    
    total_loss = measurement_loss + 0.50*side_penalty + 0.50*strap_penalty + 0.10*sizing_issue
    return total_loss



# ------------------------------
# 8. Training loop
# ------------------------------

num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, targets, garment_types, masks in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        garment_types = garment_types.to(device)
        predictions = model(images, garment_types)

        loss = pattern_loss(predictions, targets, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0


    with torch.no_grad():
        for images, targets, garment_types, masks in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            garment_types = garment_types.to(device)

            predictions = model(images, garment_types)
            loss = pattern_loss(predictions, targets, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)



# ------------------------------
# 9. Save trained model
# ------------------------------

torch.save(model.state_dict(), "shirt_pattern_resnet18.pth")
print("Model saved as shirt_pattern_resnet18.pth")

