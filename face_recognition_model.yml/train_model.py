import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_faces_and_labels(dataset_folder):
    faces = []
    labels = []
    for person in os.listdir(dataset_folder):
        person_folder = os.path.join(dataset_folder, person)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                for (x, y, w, h) in detected_faces:
                    face = gray[y:y+h, x:x+w]
                    faces.append(face)
                    labels.append(person)
                    
    return faces, labels

def train_model(dataset_folder):
    faces, labels = get_faces_and_labels(dataset_folder)
    label_encoder = {name: index for index, name in enumerate(set(labels))}
    labels = [label_encoder[label] for label in labels]
    recognizer.train(faces, np.array(labels))
    recognizer.save('face_recognition_model.yml')
    print("Model trained and saved as 'face_recognition_model.yml'")

train_model('dataset')
