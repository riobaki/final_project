import os
import cv2
import dlib
import faiss
import numpy as np
from flask import Flask, request, render_template, redirect, url_for

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat') 

dimension = 128 
index = faiss.IndexFlatL2(dimension)  

face_labels = []

def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None  

    shape = sp(image, faces[0])
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

def preprocess_dataset(dataset_path):
    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)
        image = cv2.imread(file_path)
        if image is None:
            continue

        embedding = get_face_embedding(image)
        if embedding is not None:
            face_labels.append(file_name)
            index.add(np.array([embedding]).astype(np.float32))  


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
       
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

            query_embedding = get_face_embedding(image)

            if query_embedding is not None:
                distances, indices = index.search(np.array([query_embedding]).astype(np.float32), 1)
                closest_index = indices[0][0]
                closest_label = face_labels[closest_index]
                
                return f"Closest match: {closest_label} with distance {distances[0][0]:.2f}"
            else:
                return "No face detected. Try another image."
    
    return render_template('index.html')

if __name__ == '__main__':
    preprocess_dataset('path_to_your_dataset_folder')  

    app.run(debug=True)
