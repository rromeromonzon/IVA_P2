# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 1: Deteccion de puntos de interes con Harris corner detector.

# AUTOR1: ROMERO MONZÓN, RAFAEL
# AUTOR2: LANDALUCE FERNÁNDEZ, ABEL
# PAREJA/TURNO: 13/NUMERO_TURNO

# librerias y paquetes por defecto
import numpy as np

from p2_tests import test_p2_tarea1

# Incluya aqui las librerias que necesite en su codigo
from scipy import ndimage
from scipy import signal
from scipy.ndimage import gaussian_filter
from skimage.feature import corner_peaks

def detectar_puntos_interes_harris(imagen, sigma = 1.0, k = 0.05, threshold_rel = 0.2):
    """
    # Esta funcion detecta puntos de interes en una imagen con el algoritmo de Harris.
    #
    # Argumentos de entrada:
    #   - imagen: numpy array con dimensiones [imagen_height, imagen_width].  
    #   - sigma: valor de tipo double o float que determina el factor de suavizado aplicado
    #   - k: valor de tipo double o float que determina la respuesta R de Harris
    #   - threshold_rel: valor de tipo double o float que define el umbral relativo aplicado sobre el valor maximo de R
    # Argumentos de salida
    #   - coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2] con las coordenadas 
    #                      de los puntos de interes detectados en la imagen. Cada punto de interes 
    #                      se encuentra en el formato [fila, columna] de tipo int64
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada sigma y k, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    """
    coords_esquinas = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)

   
    # Normalizar imagen a float en [0,1] manualmente 
    img = imagen.astype(np.float64)

    # Si la imagen es RGB, convertir a escala de grises mediante promedio (opcional).
    if img.ndim == 3:
        # promedio sobre canales (simple y permisible cuando no se puede usar utilidades externas)
        img = np.mean(img, axis=2)

    # Normalización manual: restar mínimo y dividir por rango si el rango > 0
    minv = img.min()
    maxv = img.max()
    if maxv > minv:
        img = (img - minv) / (maxv - minv)
    else:
        # imagen constante -> retorna array vacío
        return np.zeros((0, 2), dtype=np.int64)

    # Derivadas usando kernels de Sobel con convolve2d (mode='same') 
    # Kernel Sobel en X
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    # Kernel Sobel en Y (transpuesta)
    sobel_y = sobel_x.T

    # Usar boundary='symm' para evitar artefactos fuertes en los bordes
    Ix = signal.convolve2d(img, sobel_x, mode='same', boundary='symm')
    Iy = signal.convolve2d(img, sobel_y, mode='same', boundary='symm')

    # Productos de derivadas
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Suavizado con Gaussiano (ventana w(u,v) = 1x1 + smoothing por gaussian_filter)
    # Se pide usar gaussian_filter con mode='constant'
    Sxx = gaussian_filter(Ixx, sigma=sigma, mode='constant')
    Syy = gaussian_filter(Iyy, sigma=sigma, mode='constant')
    Sxy = gaussian_filter(Ixy, sigma=sigma, mode='constant')

    # Calcular R (respuesta de Harris)
    detM = (Sxx * Syy) - (Sxy ** 2)
    traceM = Sxx + Syy
    R = detM - k * (traceM ** 2)

    # Umbral relativo
    R_max = np.max(R)
    # Si R_max <= 0, no hay respuestas positivas (pero puede haber negativos); igual aplicamos umbral
    R_threshold = threshold_rel * R_max
    # Para permitir detectar esquinas en imágenes con R_max <= 0, podemos filtrar por R > R_threshold
    mask_threshold = R > R_threshold

    # Extracción de máximos locales con min_distance = 5 
    # peak_local_max devuelve coordenadas en (fila, columna)
    # Si no hay valores por encima del umbral, devolvemos array vacío
    if not np.any(mask_threshold):
        return np.zeros((0, 2), dtype=np.int64)

    # Aplicar corner_peaks sobre la respuesta R con threshold_abs y min_distance
    coords = corner_peaks(R, min_distance=5, threshold_rel=threshold_rel)

    # Asegurarse de que coords tenga dtype int64 y forma (N,2)
    if coords is None or coords.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    coords_esquinas = np.asarray(coords, dtype=np.int64)
    return coords_esquinas

if __name__ == "__main__":
    print("Practica 2 - Tarea 1 - Test autoevaluación\n")
    
    # print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=False,debug=False))) #analizar todos los casos sin pararse en errores
    print("Tests completados = " + str(test_p2_tarea1(disptime=1,stop_at_error=False,debug=False))) #analizar y visualizar todos los casos sin pararse en errores
    #print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=True,debug=False))) #analizar todos los casos y pararse en errores 
    #print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=True,debug=True))) #analizar todos los casos, pararse en errores y mostrar informacion