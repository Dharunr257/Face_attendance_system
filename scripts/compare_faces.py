import torch
import cv2
import os
import torch.nn as nn
from datetime import datetime
import sys
from database import update_attendance, init_db
import json
import shutil
import numpy as np
import subprocess

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

def compare_faces(student_id, model, retries=3):
    index_path = "students/students_index.json"
    if not os.path.exists(index_path):
        raise FileNotFoundError("Student index not found—create profiles first.")
    with open(index_path, 'r') as f:
        index = json.load(f)
    if student_id not in index:
        raise ValueError(f"No profile found for student {student_id}")
    dept, year, section = index[student_id]["dept"], index[student_id]["year"], index[student_id]["section"]

    model = model.to(device)
    model.eval()
    weights_path = 'model_weights.pt'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Warning: No model weights found—using untrained model. This may affect accuracy.")
    model.eval()

    today = datetime.now().strftime("%Y/%m_February/%d")
    att_path = f"attendance/{today}/{dept}/{year}/{section}/{student_id}"
    status = input("Enter 'in' or 'out': ").lower()
    temp_img_path = f"temp/{student_id}.jpg"

    attempt = 0
    while attempt < retries:
        temp_img = cv2.imread(temp_img_path, cv2.IMREAD_GRAYSCALE)
        if temp_img is None:
            raise FileNotFoundError(f"Could not load temporary image for {student_id} at {temp_img_path}")
        
        face_tensor = torch.tensor(temp_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            temp_embedding = model(face_tensor)
            temp_embedding = temp_embedding.cpu()  # Move to CPU for comparison
        
        stored_path = f"students/{dept}/{year}/{section}/{student_id}/embeddings.npy"
        if not os.path.exists(stored_path):
            raise FileNotFoundError(f"No embedding found for {student_id}")
        stored_embedding = np.load(stored_path)
        stored_embedding = torch.tensor(stored_embedding, dtype=torch.float32).to(device)

        similarity = torch.nn.functional.cosine_similarity(temp_embedding, stored_embedding, dim=1).item()
        match_percentage = (similarity + 1) / 2 * 100  # Cosine similarity ranges from -1 to 1, map to 0-100%
        threshold = 80  # 80% match threshold
        result = "present" if match_percentage > threshold else "absent"
        print(f"Attempt {attempt + 1}/{retries} - Match: {match_percentage:.2f}%, Result: {result}")

        if result == "present":
            os.makedirs(att_path, exist_ok=True)
            final_img_path = f"{att_path}/{status}_time.jpg"
            shutil.move(temp_img_path, final_img_path)
            timestamp = datetime.now().strftime('%H:%M:%S')
            update_attendance(student_id, status, timestamp, final_img_path, result, dept, year, section)
            break
        else:
            attempt += 1
            if attempt < retries:
                print(f"Retry {attempt + 1} - Please try again.")
                python_path = "C:/Program Files/Python310/python.exe"
                capture_script = "e:/Py/face attendance updated/scripts/capture_face.py"
                subprocess.run([python_path, capture_script, student_id])
            elif attempt == retries:
                print(f"Failed after {retries} attempts - Flagged for manual override.")
                update_attendance(student_id, status, None, None, "pending_manual", dept, year, section)

if __name__ == "__main__":
    init_db()
    model = FaceNetLite()
    
    student_id = sys.argv[1] if len(sys.argv) > 1 else input("Enter student register number: ")
    try:
        compare_faces(student_id, model)
    except Exception as e:
        print(f"Error: {e}")