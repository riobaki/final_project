import cv2
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

label_encoder = LabelEncoder()

def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    label, confidence = recognizer.predict(face)
    
    return label, confidence

def preprocess_dataset(dataset_folder='dataset'):
    faces = []
    labels = []
    
    for file_name in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file_name)
        image = cv2.imread(file_path)
        
        if image is None:
            continue

        label, _ = get_face_embedding(image)
        if label is not None:
            faces.append(label)
            labels.append(file_name)

    labels = label_encoder.fit_transform(labels)
    recognizer.train(faces, np.array(labels))
    recognizer.save('face_recognition_model.yml')

def search_similar_face(query_image):
    label, confidence = get_face_embedding(query_image)
    
    if label is not None:
        return label_encoder.inverse_transform([label])[0], confidence
    else:
        return "No face detected", None



