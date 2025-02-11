from flask import Flask, request, render_template
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi direktori upload
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model yang sudah dilatih
MODEL_PATH = "model/model_padi.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Label kelas sesuai dataset
class_labels = ['Hawar', 'Jamur', 'Sehat', 'Wareng']  # Ganti sesuai dengan dataset

# Ukuran gambar yang sesuai dengan model
img_size = (224, 224)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Periksa apakah file diunggah
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Proses gambar dan lakukan prediksi
            img = load_img(file_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalisasi
            img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

            prediction = model.predict(img_array)[0]
            predicted_class = class_labels[np.argmax(prediction)]

            # Tampilkan hasil prediksi
            return render_template('index.html', 
                                image_path=file_path, 
                                predicted_class=predicted_class, 
                                probabilities={class_labels[i]: f"{prob*100:.2f}%" for i, prob in enumerate(prediction)})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
