import cv2
import numpy as np
from PIL import Image
import os

path='dataset'

recognizer=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

    faceSamples=[]
    ids=[]

    for imagePath in imagePaths:

        PIL_img=Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_img,'uint8')

        id=int(os.path.split(imagePath)[-1].split(".")[1])

        # Images in dataset are already tightly cropped by camera.py
        faceSamples.append(img_numpy)
        ids.append(id)

    return faceSamples,ids

faces,ids=getImagesAndLabels(path)

recognizer.train(faces,np.array(ids, dtype=np.int32))

recognizer.write('trainer/trainer.yml')

print("Training xong")