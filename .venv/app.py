from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def extract_colors(image_path, num_colors=10):
    # Open and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))  # Reduce image size

    # Convert to numpy array
    arr = np.array(img)
    pixels = arr.reshape(-1, 3)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)

    # Get colors and counts
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)

    # Sort by frequency
    sorted_idx = np.argsort(counts)[::-1]
    return [rgb_to_hex(colors[i]) for i in sorted_idx]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            colors = extract_colors(filepath)
            os.remove(filepath)

            return render_template('result.html', colors=colors)
    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)