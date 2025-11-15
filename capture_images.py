import cv2
import os
import argparse
from utils import get_haar_cascade, ensure_dataset_dir

parser = argparse.ArgumentParser(description="Capture face images for a user")
parser.add_argument("--id", required=True, help="Numeric id for the person (e.g. 1)")
parser.add_argument("--name", required=True, help='Person name, e.g. "John Doe"')
parser.add_argument("--count", type=int, default=50, help="Number of images to capture")
args = parser.parse_args()

ensure_dataset_dir()
user_folder = os.path.join("dataset", f"{args.id}_{args.name.replace(' ', '_')}")
os.makedirs(user_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
cascade = get_haar_cascade()
print("Starting webcam. Press 'q' to quit early.")
captured = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        file_path = os.path.join(user_folder, f"{captured}.jpg")
        cv2.imwrite(file_path, face_img)
        captured += 1
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Captured: {captured}/{args.count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if captured >= args.count:
            break
    cv2.imshow('Capture', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or captured >= args.count:
        break
cap.release()
cv2.destroyAllWindows()
print(f"Saved {captured} images to {user_folder}")
