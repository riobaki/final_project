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
            name, confidence = search_similar_face(image)
            return f"Closest match: {name} with confidence {confidence:.2f}" if confidence is not None else "No face detected."

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

