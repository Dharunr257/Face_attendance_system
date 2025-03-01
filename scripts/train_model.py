import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import sys

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        self.labels = [0] * len(self.images)  # All same student for now

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {self.images[idx]}")
        img_tensor = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        return img_tensor, self.labels[idx]

def train_model(reg_no, dept, year, section):
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()  # Clear any residual memory

    model = FaceNetLite().to(device)
    weights_path = 'model_weights.pt'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded existing model weights for incremental training.")

    aug_dir = f"students/{dept}/{year}/{section}/{reg_no}/augmented"
    if not os.path.exists(aug_dir):
        print("No augmented data yet—training skipped for now.")
        return

    dataset = FaceDataset(aug_dir)
    # Use batch_size=16 for RTX 3050
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True if device.type == "cuda" else False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    triplet_loss = nn.TripletMarginLoss(margin=1.0)

    model.train()
    for epoch in range(10):
        total_loss = 0
        batch_count = 0
        total_batches = len(loader)  # For percentage calculation
        if device.type == "cuda":
            torch.cuda.empty_cache()  # Clear memory before each epoch

        for i, batch in enumerate(loader):
            # Calculate and display running percentage
            percentage = ((i + 1) / total_batches) * 100
            print(f"Epoch {epoch+1}/10 - Executed {percentage:.1f}%", end='\r')  # Update on same line

            images, labels = batch
            if len(images) < 16:  # Ensure full batch for triplets
                print(f"\nWarning: Batch {i} has only {len(images)} images—skipping.")
                continue
            # Form triplets: anchor, positive, negative
            anchor = images[0:1]
            positive = images[1:2]
            negative = images[2:3]
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        print()  # New line after epoch completion
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/10, Loss: 0.0000 (No valid batches)")

    torch.save(model.state_dict(), weights_path)
    print("Model weights updated globally.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        # Prompt for input without defaults
        reg_no = input("Enter student register number: ")
        dept = input("Enter department (e.g., AI&DS): ")
        year = input("Enter year (e.g., 2022): ")
        section = input("Enter section (e.g., Section_A): ")
        if not all([reg_no, dept, year, section]):  # Ensure all inputs are provided
            raise ValueError("All student details (reg_no, dept, year, section) must be provided.")
    else:
        reg_no, dept, year, section = sys.argv[1:5]
    train_model(reg_no, dept, year, section)