from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'static/segmented'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, k=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / np.sum(counts) * 100
    colors = [centers[i].tolist() for i in range(k)]
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))
    segmented_image_path = os.path.join(app.config['SEGMENTED_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(segmented_image_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    return colors, percentages, os.path.basename(segmented_image_path)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or 'clusters' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        clusters = int(request.form['clusters'])
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            colors, percentages, segmented_image_filename = process_image(filepath, clusters)
            return render_template('result.html', colors=colors, percentages=percentages, zip=zip, segmented_image_filename=segmented_image_filename)
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(SEGMENTED_FOLDER):
        os.makedirs(SEGMENTED_FOLDER)
    app.run(debug=True)
