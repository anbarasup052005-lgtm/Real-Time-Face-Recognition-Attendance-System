import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace
from attendance import mark_attendance

def load_embeddings():
    db = {}
    for file in os.listdir("embeddings"):
        with open(f"embeddings/{file}", "rb") as f:
            db[file.split(".")[0]] = pickle.load(f)
    return db

def find_match(embedding, db):
    min_dist = 999
    identity = "Unknown"

    for name, db_emb in db.items():
        dist = np.linalg.norm(np.array(embedding) - np.array(db_emb))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist < 10:  # threshold
        return identity
    else:
        return "Unknown"

db = load_embeddings()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imwrite("temp.jpg", frame)

    try:
        embedding = DeepFace.represent(img_path="temp.jpg", model_name="Facenet")[0]["embedding"]

        name = find_match(embedding, db)

        # Liveness check (basic)
        if name != "Unknown":
            print(f"✅ {name} recognized")
            mark_attendance(name)
        else:
            print("❌ Unknown person")

    except:
        print("Face not detected")

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()