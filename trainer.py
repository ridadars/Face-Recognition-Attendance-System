import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

def getImagesAndLabels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for imagePath in image_paths:
        gray_img = Image.open(imagePath).convert('L')
        img_np = np.array(gray_img, 'uint8')
        user_id = int(os.path.split(imagePath)[-1].split(".")[1])

        face_samples.append(img_np)
        ids.append(user_id)

    return face_samples, ids

faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids))

if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.save("trainer/trainer.yml")

print("Model training completed!")
