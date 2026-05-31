import cv2
import os
import pickle
from deepface import DeepFace

name = input("Enter Name: ")

if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

cap = cv2.VideoCapture(0)

print("📸 Look at camera...")

ret, frame = cap.read()

cv2.imwrite("temp.jpg", frame)

embedding = DeepFace.represent(img_path="temp.jpg", model_name="Facenet")[0]["embedding"]

file_path = f"embeddings/{name}.pkl"

with open(file_path, "wb") as f:
    pickle.dump(embedding, f)

print(f"✅ Face Registered for {name}")

cap.release()
cv2.destroyAllWindows()