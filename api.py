from flask import Flask, jsonify, request, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES
import cv2
import torch
import os
import tempfile
from google.cloud import storage
from modelo import CustomDenseNet, procesar_imagen, predecir_neumonia

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

def download_model(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.")

bucket_name = 'mejormodelo'
source_blob_name = 'mejor_modelo.pth'
destination_file_name = 'mejor_modelo.pth'

# Descargar el modelo antes de cargarlo
download_model(bucket_name, source_blob_name, destination_file_name)

# Cargar el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = CustomDenseNet(num_classes=2)
modelo.load_state_dict(torch.load(destination_file_name, map_location=device))
modelo.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    if 'photo' not in request.files:
        return jsonify({"error": "No se indicó el archivo de imagen"}), 400

    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    if photo and allowed_file(photo.filename):
        filename = photos.save(photo)
        imagen_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)

        imagen_tensor = procesar_imagen(image_path)
        if imagen_tensor is None:
            return jsonify({"error": "Error al procesar la imagen"}), 500

        imagen_tensor = imagen_tensor.to(device)
        prediccion = predecir_neumonia(modelo, imagen_tensor)

        if prediccion == 1:
            result = "La imagen muestra signos de neumonía."
        else:
            result = "La imagen no muestra signos de neumonía."

        return jsonify({"respuesta": result})

    return jsonify({"error": "Formato de archivo no válido"}), 400

@app.route('/')
def upload_form():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
