import os
import cv2
import numpy as np
import json
from utils import get_haar_cascade, save_labels_map

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascade = get_haar_cascade()

faces = []
labels = []
label_map = {}  # numeric label -> name
current_label = 0

# Iterate dataset structure: each subfolder is id_name
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):
        continue
    # folder format: id_name
    label_id = folder.split("_")[0]
    label_name = "_".join(folder.split("_")[1:])
    numeric_label = int(label_id)
    label_map[str(numeric_label)] = label_name
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(numeric_label)

if len(faces) == 0:
    raise RuntimeError("No face images found in dataset/. Capture images first with capture_images.py")

recognizer.train(faces, np.array(labels))
recognizer.write("trainer.yml")
save_labels_map(label_map, "labels.json")
print("Training complete. Saved trainer.yml and labels.json")
