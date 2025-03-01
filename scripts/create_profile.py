import os
import json
import shutil
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import subprocess

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

def augment_images(image_path, output_dir, num_samples=10):
    img = Image.open(image_path).convert('L')
    transforms = [
        T.RandomRotation(10), T.RandomRotation(20),
        T.ColorJitter(brightness=0.2), T.ColorJitter(brightness=0.4),
        T.ColorJitter(contrast=0.2), T.RandomHorizontalFlip(),
    ]
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        aug_img = img
        for t in np.random.choice(transforms, size=2):
            aug_img = t(aug_img)
        aug_img.save(f'{output_dir}/aug_{i}.jpg')

def create_profile():
    reg_no = input("Enter student register number: ")
    name = input("Enter student name: ")
    dept = input("Enter department (e.g., CSE): ")
    year = input("Enter year (e.g., 2024): ")
    section = input("Enter section (e.g., Section_A): ")
    photo_dir = input("Enter path to 10-15 photos folder: ")

    student_path = f"students/{dept}/{year}/{section}/{reg_no}"
    os.makedirs(student_path, exist_ok=True)
    os.makedirs(f"{student_path}/images", exist_ok=True)

    for photo in os.listdir(photo_dir):
        if photo.endswith('.jpg'):
            shutil.copy(os.path.join(photo_dir, photo), f"{student_path}/images/{photo}")

    profile = {"reg_no": reg_no, "name": name, "dept": dept, "year": year, "section": section}
    with open(f"{student_path}/profile.json", 'w') as f:
        json.dump(profile, f)

    index_path = "students/students_index.json"
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
    else:
        index = {}
    index[reg_no] = {"dept": dept, "year": year, "section": section}
    with open(index_path, 'w') as f:
        json.dump(index, f)

    python_path = "C:/Program Files/Python310/python.exe"
    extract_script = "e:/Py/face attendance updated/scripts/extract_features.py"
    train_script = "e:/Py/face attendance updated/scripts/train_model.py"
    
    print(f"Running extract_features for {reg_no}...")
    subprocess.run([python_path, extract_script, reg_no, dept, year, section], check=True)
    print(f"Augmented data created—running train_model...")
    subprocess.run([python_path, train_script, reg_no, dept, year, section], check=True)
    print(f"Running extract_features again with trained model...")
    subprocess.run([python_path, extract_script, reg_no, dept, year, section], check=True)

    print(f"Profile created for {reg_no}")

if __name__ == "__main__":
    create_profile()