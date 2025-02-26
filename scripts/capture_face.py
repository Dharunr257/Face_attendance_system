import cv2
import mediapipe as mp
import numpy as np
import os
from time import time
from datetime import datetime
import subprocess
import json

def eye_aspect_ratio(eye_landmarks, facial_landmarks):
    def to_np(point_idx):
        return np.array([facial_landmarks[point_idx][0], facial_landmarks[point_idx][1]])
    v1 = np.linalg.norm(to_np(eye_landmarks[1]) - to_np(eye_landmarks[5]))
    v2 = np.linalg.norm(to_np(eye_landmarks[2]) - to_np(eye_landmarks[4]))
    h = np.linalg.norm(to_np(eye_landmarks[0]) - to_np(eye_landmarks[3]))
    return (v1 + v2) / (2.0 * h)

def is_face_in_box(landmarks, box):
    x_min, y_min, x_max, y_max = box
    face_x = [lm[0] for lm in landmarks]
    face_y = [lm[1] for lm in landmarks]
    return (min(face_x) > x_min and max(face_x) < x_max and
            min(face_y) > y_min and max(face_y) < y_max)

def capture_on_blink(student_id):
    # Load student details from index (no defaults)
    index_path = "students/students_index.json"
    if not os.path.exists(index_path):
        raise FileNotFoundError("Student index not found—create profiles first.")
    with open(index_path, 'r') as f:
        index = json.load(f)
    if student_id not in index:
        raise ValueError(f"No profile found for student {student_id}")
    dept, year, section = index[student_id]["dept"], index[student_id]["year"], index[student_id]["section"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    frame_w, frame_h = int(cap.get(3)), int(cap.get(4))
    box_size = 300
    box_x, box_y = (frame_w - box_size) // 2, (frame_h - box_size) // 2
    capture_box = (box_x, box_y, box_x + box_size, box_y + box_size)

    start_time = time()
    time_limit = 10
    blinked = False
    last_blink_time = time()
    blink_cooldown = 0.3
    ear_threshold = 0.18
    ear_open_threshold = 0.30
    eyes_closed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        cv2.rectangle(frame, (capture_box[0], capture_box[1]), (capture_box[2], capture_box[3]), (255, 0, 0), 2)
        elapsed = time() - start_time
        remaining = max(0, time_limit - elapsed)
        cv2.putText(frame, f"Time: {remaining:.1f}s", (frame_w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if remaining <= 0:
            print("Time’s up!")
            break

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                left_eye = [362, 385, 387, 263, 373, 380]
                right_eye = [33, 160, 158, 133, 153, 144]
                
                left_ear = eye_aspect_ratio(left_eye, landmarks)
                right_ear = eye_aspect_ratio(right_eye, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                in_box = is_face_in_box(landmarks, capture_box)
                box_color = (0, 255, 0) if in_box else (255, 0, 0)
                cv2.rectangle(frame, (capture_box[0], capture_box[1]), (capture_box[2], capture_box[3]), box_color, 2)

                if in_box:
                    current_time = time()
                    if avg_ear < ear_threshold and not eyes_closed and (current_time - last_blink_time > blink_cooldown):
                        eyes_closed = True
                    elif avg_ear > ear_open_threshold and eyes_closed and (current_time - last_blink_time > blink_cooldown):
                        blinked = True
                        last_blink_time = current_time
                        face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        x, y = min([lm[0] for lm in landmarks]), min([lm[1] for lm in landmarks])
                        w, h = max([lm[0] for lm in landmarks]) - x, max([lm[1] for lm in landmarks]) - y
                        face = face[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (96, 96))
                        
                        os.makedirs('temp', exist_ok=True)
                        temp_path = f"temp/{student_id}.jpg"
                        cv2.imwrite(temp_path, face_resized)
                        print(f"Captured face for {student_id}")
                        cap.release()
                        face_mesh.close()
                        cv2.destroyAllWindows()
                        
                        python_path = "C:/Program Files/Python310/python.exe"
                        compare_script = "e:/Py/face attendance updated/scripts/compare_faces.py"
                        subprocess.run([python_path, compare_script, student_id])
                        return

                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Blink to Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    student_id = input("Enter student register number: ")
    try:
        capture_on_blink(student_id)
    except Exception as e:
        print(f"Error: {e}")