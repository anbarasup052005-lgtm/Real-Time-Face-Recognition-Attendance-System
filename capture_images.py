import cv2
import os

name = input("Enter Name: ")

# Create dataset folder
dataset_path = "dataset/" + name
os.makedirs(dataset_path, exist_ok=True)

# 🔥 Correct Haarcascade loading
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Loaded:", not face_cascade.empty())

cap = cv2.VideoCapture(0)

count = 0

print("📸 Look at camera... capturing starts!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))

        file_name = f"{dataset_path}/{count}.jpg"
        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Capturing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Images Saved Successfully!")