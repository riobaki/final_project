from flask import Flask, request, render_template
import cv2
import numpy as np
from face_recognition import get_face_embedding, search_similar_face

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            query_embedding = get_face_embedding(image)

            if query_embedding is not None:
                closest_label, distance = search_similar_face(query_embedding)
                return f"Closest match: {closest_label} with distance {distance:.2f}"
            else:
                return "No face detected. Try another image."

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
