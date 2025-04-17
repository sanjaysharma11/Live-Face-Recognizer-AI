import face_recognition
import cv2
import os
import numpy as np

def load_and_encode_images(train_dir):
    known_faces = []
    known_names = []

    print("[INFO] Loading and encoding known faces...")
    for filename in os.listdir(train_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filepath = os.path.join(train_dir, filename)
        name = os.path.splitext(filename)[0]

        # Read image
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            print(f"[ERROR] Unable to read image: {filename}")
            continue

        # Convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Validate image format
        if img_rgb.dtype != np.uint8 or len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
            print(f"[ERROR] Unsupported image format in file: {filename}")
            continue

        # Encode
        encodings = face_recognition.face_encodings(img_rgb)
        if not encodings:
            print(f"[WARNING] No face detected in: {filename}")
            continue

        known_faces.append(encodings[0])
        known_names.append(name)
        print(f"[INFO] Encoded: {name}")

    return known_faces, known_names

# Load training data
train_path = "train"
if not os.path.exists(train_path):
    raise FileNotFoundError(f"[ERROR] Folder not found: {train_path}")

known_faces, known_names = load_and_encode_images(train_path)

# Start webcam
print("[INFO] Starting webcam...")
video = cv2.VideoCapture(0)

if not video.isOpened():
    raise RuntimeError("[ERROR] Unable to access the webcam.")

while True:
    ret, frame = video.read()
    if not ret:
        print("[ERROR] Webcam frame not captured.")
        break

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if rgb_frame.dtype != np.uint8 or len(rgb_frame.shape) != 3 or rgb_frame.shape[2] != 3:
            print("[ERROR] Invalid webcam frame format.")
            continue

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if matches:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Face Recognition - Press 'q' to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        continue

video.release()
cv2.destroyAllWindows() 