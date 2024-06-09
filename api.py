from flask import Flask, jsonify, request
import cv2
import torch
import os
import tempfile
from google.cloud import storage
from modelo import CustomDenseNet, procesar_imagen, predecir_neumonia

app = Flask(__name__)

# Función para descargar el modelo desde Google Cloud Storage
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
        return jsonify({"error": "No se indicó el nombre del archivo"}), 400

    file = request.files['file']

    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_image_path = temp.name
        file.save(temp_image_path)

    try:
        # Leer y procesar la imagen
        imagen = cv2.imread(temp_image_path)
        if imagen is None:
            return jsonify({"error": f"No se pudo leer la imagen en la ruta: {temp_image_path}"}), 400

        imagen_tensor = procesar_imagen(temp_image_path)
        if imagen_tensor is None:
            return jsonify({"error": "Error al procesar la imagen"}), 500

        imagen_tensor = imagen_tensor.to(device)
        prediccion = predecir_neumonia(modelo, imagen_tensor)

        if prediccion == 1:
            result = "La imagen muestra signos de neumonía."
        else:
            result = "La imagen no muestra signos de neumonía."

        return jsonify({"respuesta": result})

    finally:
        # Asegurarse de eliminar el archivo temporal
        os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
