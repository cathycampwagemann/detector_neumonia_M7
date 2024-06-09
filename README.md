# Descripción del proyecto
Es una API que sirve para detectar neumonía en radiografías de tórax

# Instalación
Se debe crear un entorno virtual limpio

Se deben instalar las siguientes librerías:
pip install Flask
pip install opencv-python
pip install torch torchvision 
pip install tensorflow
pip install requests

# Ejemplo de uso

carpeta_principal_imagenes = # Indica la ruta donde están las imágenes

nombre_imagenes = # Indica el nombre de la imagen

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelo = CustomDenseNet(num_classes=2)
    modelo.load_state_dict(torch.load('https://drive.google.com/file/d/1Ed9g2Rj_k7CPF8ClBalaYfDhfbNlsuTC/view?usp=drive_link'))
    modelo.to(device)

    resultados=[]

    for nombre_imagen in nombre_imagenes:
        imagen_tensor = procesar_imagen(nombre_imagen, carpeta_principal_imagenes)
        if imagen_tensor is not None:
            imagen_tensor = imagen_tensor.to(device)
            prediccion = predecir_neumonia(modelo, imagen_tensor)
            if prediccion == 1:
                resultados.append("La imagen {} muestra signos de neumonía.".format(nombre_imagen))
            else:
                resultados.append("La imagen {} no muestra signos de neumonía.".format(nombre_imagen))
        else:
            resultados.append("Error: No se pudo procesar la imagen {}.".format(nombre_imagen))

    for resultado in resultados:
        print(resultado)

# Estructura de directorios

1. experimentos: contiene el notebook con el análisis y exploración de los datos, la transformación de las imágenes y el entrenamiento, prueba y validación del modelo.
2. mi proyecto: contiene los archivos api.py y modelo.py que es necesario guardarlos en la misma carpeta para poder ejecutarlos, el modelo (por si no se puede acceder al drive antes indicado) y los requirements

# Modelo

Es una red neuronal convulocional que tiene como base una DenseNet, que customicé según los requerimientos de mi data (entre ellos, el hecho que mi modelo es de clasificación binaria)-

# EDA

El dataset utilizado fueron radiografías de tórax de niños del Centro Médico para mujeres y niños de Cantón, China.
En el análisis exploratorio se detectó que las imágenes tenían distintas dimensiones; la mayoría de las imágenes estaba en escala de grises, pero otras en RGA; existía desbalanceo de las categorías.

# Tansformación de imágenes

Para que las imágenes fueran procesadas correctamente, fueron:
1. Redimensionadas según 3 altos distintos (447, 800 y 1440 píxeles). El ancho se redimensionó según las proporcionales originales.
2. Convertidas a escala de grises.
3. Para que tuvieran el mismo alto y ancho, sin que perdieran la calidad, se le aplicó padding a todas las imágenes para que quedaran en 1440x1440 píxeles.
4. Luego, fueron nuevamente redimensionadas a 720x720 píxeles para que pudieran ser procesadas sin problemas.
