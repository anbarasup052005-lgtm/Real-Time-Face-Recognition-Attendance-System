import os
import cv2
import numpy as np
from tensorflow.keras import layers, models

data = []
labels = []
names = []

dataset_path = "dataset"

for idx, person in enumerate(os.listdir(dataset_path)):
    names.append(person)
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (100,100))
        data.append(img)
        labels.append(idx)

data = np.array(data) / 255.0
data = data.reshape(-1,100,100,1)
labels = np.array(labels)

model = models.Sequential([
    layers.Input(shape=(100,100,1)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(data, labels, epochs=15)

os.makedirs("model", exist_ok=True)
model.save("model/face_model.keras")
np.save("model/names.npy", names)

print("✅ Model Trained!")