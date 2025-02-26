import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import sys

class FaceNetLite(nn.Module):
    def __init__(self):
        super(FaceNetLite, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.fc1(x)
        return x

class FaceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('.jpg')]
        self.labels = [0] * len(self.images)  # All same student for now (will expand for multiple)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(0)
        return img, self.labels[idx]

def train_model(reg_no, dept, year, section):
    model = FaceNetLite()
    weights_path = 'model_weights.pt'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print("Loaded existing model weights for incremental training.")

    aug_dir = f"students/{dept}/{year}/{section}/{reg_no}/augmented"
    if not os.path.exists(aug_dir):
        print("No augmented data yet—training skipped for now.")
        return

    dataset = FaceDataset(aug_dir)
    loader = DataLoader(dataset, batch_size=3, shuffle=True)  # Batch size 3 for triplets
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    triplet_loss = nn.TripletMarginLoss(margin=1.0)

    model.train()
    for epoch in range(5):  # More epochs for better training
        total_loss = 0
        for batch in loader:
            images, labels = batch
            # Create triplets: anchor, positive (same person), negative (different person)
            # For now, use all as same person (expand later for multiple students)
            anchor, positive, negative = images[0:1], images[1:2], images[2:3]  # Simple triplet
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), weights_path)
    print("Model weights updated globally.")

if __name__ == "__main__":
    reg_no, dept, year, section = sys.argv[1:5]
    train_model(reg_no, dept, year, section)