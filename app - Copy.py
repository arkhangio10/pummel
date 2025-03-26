
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__)

# Funciones de procesamiento
def calcular_glcm(imagen):
    glcm = graycomatrix(imagen, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contraste = graycoprops(glcm, 'contrast')[0, 0]
    energia = graycoprops(glcm, 'energy')[0, 0]
    homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
    correlacion = graycoprops(glcm, 'correlation')[0, 0]
    return contraste, energia, homogeneidad, correlacion

def procesar_imagen(imagen_gris):
    # Umbralización sin Otsu
    _, umbral = cv2.threshold(imagen_gris, 120, 255, cv2.THRESH_BINARY)
    total_pixeles = umbral.size
    area_desprendida = np.sum(umbral == 0)  # Área blanca, representa el área desprendida
    porcentaje_desprendido = (area_desprendida / total_pixeles) * 100
    porcentaje_adherido = 100 - porcentaje_desprendido  # Complemento del área desprendida

    # Cálculo del criterio de caída de píxeles
    # Cuenta el número de transiciones de negro a blanco (cambios de adherido a desprendido)
    transiciones = np.sum((umbral[:-1, :] == 0) & (umbral[1:, :] == 255)) + \
                   np.sum((umbral[:, :-1] == 0) & (umbral[:, 1:] == 255))
    criterio_caida = transiciones / total_pixeles  # Normalización por el total de píxeles

    # Gradiente
    gradiente_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)
    magnitud_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)
    promedio_gradiente = np.mean(magnitud_gradiente)
    
    # GLCM
    contraste, energia, homogeneidad, correlacion = calcular_glcm(imagen_gris)

    return {
        "porcentaje_desprendido": porcentaje_desprendido,
        "criterio_caida": criterio_caida,
        "promedio_gradiente": promedio_gradiente,
        "contraste": contraste,
        "energia": energia,
        "homogeneidad": homogeneidad,
        "correlacion": correlacion
    }

def determinar_grado(datos):
    porcentaje_desprendido = datos["porcentaje_desprendido"]

    
    if porcentaje_desprendido >= 65:  # Grado 2
        return 2
    elif 50 <= porcentaje_desprendido < 65:  # Grado 3
        return 3
    elif 28 <= porcentaje_desprendido < 50:  # Grado 4
        return 4
    elif 23 <= porcentaje_desprendido < 28:  # Grado 5
        return 5
    elif 20 <= porcentaje_desprendido < 23:  # Grado 6
        return 6
    elif 12 <= porcentaje_desprendido < 20:  # Grado 7
        return 7
    elif 13 <= porcentaje_desprendido < 15:  # Grado 8
        return 8
    else:  # Porcentaje menor a 13, Grado 9
        return 9


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

   
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    datos = procesar_imagen(img)
    grado = determinar_grado(datos)

    _, umbral = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    processed_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    processed_image[umbral == 255] = [0, 0, 255]  # Rojo para área desprendida

    output_path = os.path.join("static", "processed_image.png")
    cv2.imwrite(output_path, processed_image)

    response = {
        "porcentaje_desprendido": datos["porcentaje_desprendido"],
        "criterio_caida": datos["criterio_caida"],
        "grado": grado,
        "processed_image_url": f"/{output_path}"
    }
    return jsonify(response)

if __name__ == '__main__':
    print("⚙️ Iniciando servidor Flask...")
    app.run(debug=True)




