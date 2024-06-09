from flask import Flask, jsonify, request
import cv2
import torch
import os
from modelo import CustomDenseNet, procesar_imagen, predecir_neumonia

app = Flask(__name__)

# Se carga el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = CustomDenseNet(num_classes=2)
modelo.load_state_dict(torch.load('C:/Users/catyc/Downloads/mejor_modelo.pth'))
modelo.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se indico el nombre del arhivo"}), 400

    file = request.files['file']
    nombre_imagen = file.filename
    carpeta_principal_imagenes = request.form.get('carpeta_principal_imagenes')

    if not carpeta_principal_imagenes:
        return jsonify({"error": "No se indicó la carpeta_principal_imagenes"}), 400

    if not os.path.exists(carpeta_principal_imagenes):
        return jsonify({"error": "La carpeta especificada no existe"}), 400

    temp_image_path = os.path.join(carpeta_principal_imagenes, nombre_imagen)
    file.save(temp_image_path)
    print(f"Imagen guardada temporalmente en: {temp_image_path}")
    if not os.path.isfile(temp_image_path):
        return jsonify({"error": f"No se pudo guardar la imagen en la ruta: {temp_image_path}"}), 400

    imagen = cv2.imread(temp_image_path)
    if imagen is None:
        return jsonify({"error": f"No se pudo leer la imagen en la ruta: {temp_image_path}"}), 400

    imagen_tensor = procesar_imagen(temp_image_path,carpeta_principal_imagenes)
    if imagen_tensor is None:
        return jsonify({"error": "Error al procesar la imagen"}), 500

    imagen_tensor = imagen_tensor.to(device)

    prediccion = predecir_neumonia(modelo, imagen_tensor)

    if prediccion == 1:
        result = "La imagen muestra signos de neumonía."
    else:
        result = "La imagen no muestra signos de neumonía."

    return jsonify({"respuesta": result})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
