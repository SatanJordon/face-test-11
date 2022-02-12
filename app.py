import os
import time
from flask import Flask, render_template, Response, request, jsonify, redirect
import cv2

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


def check_photo():
    gen_frames()


@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        butt = request.form.get('but')
        username = request.form.get('username')
        if butt == 'Click Me':
            take_photo(username)
            camera.release()
            cv2.destroyAllWindows()
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
            camera.release()
            cv2.destroyAllWindows()
            if True:
                return redirect('content')
            else:
                raise ValueError('Error')
    check_photo()
    return render_template('login.html')


@app.route("/")
def health():
    return jsonify(status='UP')


if __name__ == "__main__":
    app.run(debug=True)
