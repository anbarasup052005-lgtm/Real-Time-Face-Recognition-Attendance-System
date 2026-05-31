import cv2
import face_recognition
import os
import pickle

if not os.path.exists("faces"):
    os.makedirs("faces")

known_encodings = []
known_names = []

# load old data
if os.path.exists("faces/data.pkl"):
    with open("faces/data.pkl", "rb") as f:
        known_encodings, known_names = pickle.load(f)

name = input("Enter Name: ")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

        print("Face Registered Successfully!")
        cap.release()
        cv2.destroyAllWindows()

        with open("faces/data.pkl", "wb") as f:
            pickle.dump((known_encodings, known_names), f)

        exit()

    cv2.imshow("Register Face", frame)
    if cv2.waitKey(1) == 27:
        break

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")