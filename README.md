# Face Recognition Attendance App (Python)

**Contents**
- `capture_images.py` — Capture images for each person using webcam.
- `train_model.py` — Train an LBPH face recognizer on captured images.
- `attendance.py` — Run real-time recognition and log attendance to `attendance.csv`.
- `utils.py` — Helper functions.
- `requirements.txt` — Python package requirements.
- `dataset/` — Where `capture_images.py` will save images (created at runtime).

## Quick setup
1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate    # macOS / Linux
   venv\Scripts\activate     # Windows
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   > If you get an error about `cv2.face` missing, install `opencv-contrib-python` (it's already listed).

3. Capture images for people:
   ```
   python capture_images.py --id 1 --name "B.Krishna"
   ```
   Follow on-screen prompts; it will save ~50 face images to `dataset/1_B.Krishna/`.

4. Train the model:
   ```
   python train_model.py
   ```
   This creates `trainer.yml` and `labels.json`.

5. Run attendance:
   ```
   python attendance.py
   ```
   The app opens the webcam, recognizes faces and appends entries to `attendance.csv`.

## Notes and troubleshooting
- Uses OpenCV LBPH face recognizer and Haar cascade for detection.
- Good lighting and frontal faces improve accuracy.
- If `cv2.face` is not found, make sure `opencv-contrib-python` is installed.
- To reduce false positives, adjust `CONFIDENCE_THRESHOLD` value in `attendance.py`.

## License
MIT — adapt freely for learning and small projects.
