from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
import cv2
import torch
import os
import tempfile
from google.cloud import storage
from modelo import CustomDenseNet, procesar_imagen, predecir_neumonia

# Crear la instancia de la aplicación Flask con la configuración de la carpeta estática
app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    if 'file' not in request.files:
        return jsonify({"error": "No se indicó el archivo de imagen"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        imagen_tensor = procesar_imagen(filepath)
        if imagen_tensor is None:
            return jsonify({"error": "Error al procesar la imagen"}), 500

        imagen_tensor = imagen_tensor.to(device)
        prediccion = predecir_neumonia(modelo, imagen_tensor)

        if prediccion == 1:
            result = "La imagen muestra signos de neumonía."
        else:
            result = "La imagen no muestra signos de neumonía."

        return jsonify({"respuesta": result})

# Servir el archivo index.html en la ruta raíz '/'
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
