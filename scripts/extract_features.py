import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import sys
import shutil

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
        aug_img.save(f'{output_dir}/aug_{len(os.listdir(output_dir))}.jpg')

def extract_and_save(reg_no, dept, year, section, model):
    model.eval()
    model = model.to(device)

    cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found at {cascade_path}")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    student_path = f"students/{dept}/{year}/{section}/{reg_no}"
    photo_dir = f"{student_path}/images"
    aug_dir = f"{student_path}/augmented"
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)

    embeddings = []
    for photo in os.listdir(photo_dir):
        if photo.endswith('.jpg'):
            augment_images(f'{photo_dir}/{photo}', aug_dir)

    unique_images = set()
    for img_file in os.listdir(aug_dir):
        img_path = f'{aug_dir}/{img_file}'
        with open(img_path, 'rb') as f:
            img_hash = hash(f.read())
        if img_hash in unique_images:
            print(f"Warning: Duplicate image detected, removing {img_file}")
            os.remove(img_path)
        else:
            unique_images.add(img_hash)

    for img_file in os.listdir(aug_dir):
        img = cv2.imread(f'{aug_dir}/{img_file}', cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_file}")
            continue
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (96, 96))
            face_tensor = torch.tensor(face_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(face_tensor)
                embedding = embedding.cpu().numpy()  # Move to CPU for numpy conversion
            embeddings.append(embedding)

    if not embeddings:
        raise ValueError("No faces detected in any images")
    avg_embedding = np.mean(np.stack(embeddings), axis=0)
    np.save(f"{student_path}/embeddings.npy", avg_embedding)
    print(f"Saved embedding for {reg_no} with {len(embeddings)} unique images")

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
    model = FaceNetLite()
    weights_path = 'model_weights.pt'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("No model weights found—using untrained model for initial extraction.")
    model.eval()
    extract_and_save(reg_no, dept, year, section, model)