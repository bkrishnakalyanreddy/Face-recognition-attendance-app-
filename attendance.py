import cv2
import time
import csv
from datetime import datetime
from utils import get_haar_cascade, load_labels_map

cascade = get_haar_cascade()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels = load_labels_map("labels.json")  # keys are string ids

ATTENDANCE_FILE = "attendance.csv"
CONFIDENCE_THRESHOLD = 70  # lower is better match for LBPH (experiment)

def mark_attendance(person_id, person_name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Avoid duplicate marking within the same session for same person
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([person_id, person_name, now])

cap = cv2.VideoCapture(0)
recognized_this_session = set()
print("Starting attendance. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
        label, confidence = recognizer.predict(roi)
        # LBPH returns lower confidence for better match
        if confidence < CONFIDENCE_THRESHOLD:
            pid = str(label)
            pname = labels.get(pid, "Unknown").replace("_", " ")
            if pid not in recognized_this_session:
                mark_attendance(pid, pname)
                recognized_this_session.add(pid)
            cv2.putText(frame, f"{pname} ({confidence:.1f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Attendance session ended. Entries appended to", ATTENDANCE_FILE)
