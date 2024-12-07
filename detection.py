import cv2
from keras.models import model_from_json
import numpy as np
import os

class EmotionDetector:
    def __init__(self):
        # Load model architecture and weights
        json_file_path = r"E:\Desktop\face\emotiondetector.json"
        h5_file_path = r"E:\Desktop\face\emotiondetector.h5"

        if not os.path.isfile(json_file_path) or not os.path.isfile(h5_file_path):
            raise FileNotFoundError("Model files not found.")

        with open(json_file_path, 'r') as json_file:
            model_json = json_file.read()
            self.model = model_from_json(model_json)
        self.model.load_weights(h5_file_path)

        self.haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.haar_file)

        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    def extract_features(self, image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    def detect_emotions(self):
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            raise RuntimeError("Could not open webcam.")

        while True:
            ret, im = webcam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

                face = cv2.resize(face, (48, 48))
                img = self.extract_features(face)

                pred = self.model.predict(img)
                prediction_label = self.labels[np.argmax(pred)]

                cv2.putText(im, prediction_label, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

            _, buffer = cv2.imencode('.jpg', im)
            frame = buffer.tobytes()
            yield frame

        webcam.release()