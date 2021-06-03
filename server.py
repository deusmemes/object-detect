import os
import base64
import json
import cv2 as cv
import numpy as np
import obj_analyzer.analyzer as an
import obj_analyzer.image as im
from obj_analyzer.model import get_all_models

from flask import Flask, redirect, flash, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.curdir, '..', 'data')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run():
    app.run(debug=True)


@app.route('/test', methods=['GET'])
def test():
    img = cv.imread('../test3.jpg')
    string = base64.b64encode(cv.imencode('.jpg', img)[1]).decode()
    return json.dumps({'first': string})


@app.route('/analyze', methods=['POST'])
def upload_file():
    files = request.files.getlist("files")
    print(files)
    cv_images = []

    for file in files:
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            frame = cv.imdecode(np.fromstring(file.stream.read(), np.uint8), cv.IMREAD_COLOR)
            cv_images.append(frame)

    matched_images, areas = an.analyze(model_name=request.form.get('model'), images=cv_images)
    base64_images = [im.convert_to_base64(img) for img in matched_images]
    data_json = json.dumps({'areas': list(areas), 'images': base64_images})

    return data_json


@app.route('/models', methods=['GET'])
def get_models():
    return get_all_models()
