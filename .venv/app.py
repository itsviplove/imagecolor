from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = os.urandom(24)  # Generate secure secret key

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def extract_colors(image_path, num_colors=10):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((150, 150))
        arr = np.array(img)
        pixels = arr.reshape(-1, 3)

        if len(pixels) < num_colors:
            raise ValueError("Image has too few pixels for analysis")

        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(kmeans.labels_)
        sorted_idx = np.argsort(counts)[::-1]

        return [rgb_to_hex(colors[i]) for i in sorted_idx]

    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        raise


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Please select a file first', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                colors = extract_colors(filepath)
                os.remove(filepath)

                return render_template('result.html', colors=colors)

            except UnidentifiedImageError:
                flash('Invalid or corrupted image file', 'error')
            except ValueError as ve:
                flash(str(ve), 'error')
            except Exception as e:
                flash('Error processing image. Please try again.', 'error')

            return redirect(url_for('index'))

    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)