from flask import Flask, render_template, Response, request
from detection import EmotionDetector

app = Flask(__name__)

# Initialize the emotion detector
emotion_detector = EmotionDetector()

@app.route('/')
def index():
    return render_template('index.html')  # Home page

def gen_frames():
    """ Generate video frames with emotion detection. """
    for frame in emotion_detector.detect_emotions():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
