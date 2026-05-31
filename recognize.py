import cv2
import numpy as np
from tensorflow.keras.models import load_model
from attendance import mark_attendance

# =========================
# Load Model + Names
# =========================
model = load_model("model/face_model.keras")
names = np.load("model/names.npy", allow_pickle=True)

# =========================
# Face Detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Camera Start
# =========================
cap = cv2.VideoCapture(0)

prev_frame = None

print("🎥 AI Face Attendance Started...")

# =========================
# Real Face Check
# =========================
def is_real_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur detection
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur < 50:
        return False

    return True

# =========================
# Main Loop
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # -------------------------
        # Face Crop
        # -------------------------
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))

        face_input = face.reshape(1,100,100,1) / 255.0

        # -------------------------
        # Prediction
        # -------------------------
        pred = model.predict(face_input, verbose=0)

        class_id = np.argmax(pred)
        confidence = np.max(pred)

        # -------------------------
        # Motion Detection
        # -------------------------
        motion_score = 0

        if prev_frame is not None:

            diff = cv2.absdiff(prev_frame, gray)

            motion_score = np.sum(diff) / 1000000

        prev_frame = gray.copy()

        # -------------------------
        # Real/Fake Check
        # -------------------------
        real_face = is_real_face(frame)

        # -------------------------
        # Final Decision
        # -------------------------
        if confidence > 0.80 and motion_score > 0.5 and real_face:

            name = names[class_id]

            color = (0,255,0)

            mark_attendance(name)

        else:

            name = "Fake / Unknown"

            color = (0,0,255)

        # -------------------------
        # Display
        # -------------------------
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        cv2.putText(
            frame,
            f"{name} ({confidence:.2f})",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("AI Face Attendance", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()