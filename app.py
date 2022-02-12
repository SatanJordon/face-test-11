import os
import time
from flask import Flask, render_template, Response, request, jsonify, redirect
import cv2
from PIL import Image
import numpy as np
import pickle

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)

            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def take_photo(name):
    gen_frames()
    os.makedirs('images/' + name)
    for i in range(5):
        time.sleep(1)
        return_value, image = camera.read()

        cv2.imwrite('images/' + name + '/opencv' + str(i) + '.png', image)


@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        butt = request.form.get('but')
        username = request.form.get('username')
        if butt == 'Click Me':
            take_photo(username)
            camera.release()
            cv2.destroyAllWindows()
            train_face()
            return redirect('login')

            # face_recog.run_con()

    return render_template("index.html")


@app.route("/content")
def content():
    return render_template('content.html')


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        butt = request.form.get('but')
        username = request.form.get('username')
        if butt == 'Click Me':
            if check_face(username):
                return redirect('content')
    return render_template('login.html')


@app.route("/")
def health():
    return jsonify(status='UP')


def train_face():
    image_dir = r'E:\PythonBasics\pythonProject\face-test-11\images'
    face_cascade = cv2.CascadeClassifier('face_recog/cascades/data/haarcascade_frontalface_alt2.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    x_train = []
    y_labels = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(' ', '-').lower()
                # y_labels.append(label)
                # x_train.append(path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                pil_image = Image.open(path).convert("L")  # grayscale
                image_array = np.array(pil_image, "uint8")
                faces = face_cascade.detectMultiScale(image_array, minNeighbors=5)

                for x, y, w, h in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
    with open("dumps/labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('dumps/trainer.yml')


def check_face(username):
    face_cascade = cv2.CascadeClassifier('face_recog/cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('dumps/trainer.yml')
    labels = {}
    with open("dumps/labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    ret, frame = camera.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, minNeighbors=5)
    for x, y, w, h in faces:
        roi_gray = gray_img[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf <= 85:
            print(conf)
            print(id_)
            print(labels[id_])
            if username.__eq__(labels[id_]):
                return True
        img_item = 'my-image.png'
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        encord_x = x + w
        encord_y = y + h
        cv2.rectangle(frame, (x, y), (encord_x, encord_y), color, stroke)
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    app.run(debug=True)
