# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 2: Descripcion de puntos de interes mediante histogramas.

# AUTOR1: ROMERO MONZÓN, RAFAEL
# AUTOR2: LANDALUCE FERNÁNDEZ, ABEL
# PAREJA/TURNO: 13/NUMERO_TURNO

# librerias y paquetes por defecto
import numpy as np
# Incluya aqui las librerias que necesite en su codigo
# ...

def descripcion_puntos_interes(imagen, coords_esquinas, vtam = 8, nbins = 16, tipoDesc='hist'):
    """
    # Esta funcion describe puntos de interes de una imagen mediante histogramas, analizando 
    # vecindarios con dimensiones "vtam+1"x"vtam+1" centrados en las coordenadas de cada punto de interes
    #   
    # La descripcion obtenida depende del parametro 'tipoDesc'
    #   - Caso 'hist': histograma normalizado de valores de gris 
    #   - Caso 'mag-ori': histograma de orientaciones de gradiente
    #
    # En el caso de que existan puntos de interes en los bordes de la imagen, el descriptor no
    # se calcula y el punto de interes se elimina de la lista <new_coords_esquinas> que devuelve
    # esta funcion. Esta lista indica los puntos de interes para los cuales existe descriptor.
    #
    # Argumentos de entrada:
    #   - imagen: numpy array con dimensiones [imagen_height, imagen_width].        
    #   - coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2] con las coordenadas 
    #                      de los puntos de interes detectados en la imagen. Tipo int64
    #                      Cada punto de interes se encuentra en el formato [fila, columna]
    #   - vtam: valor de tipo entero que indica el tamaño del vecindario a considerar para
    #           calcular el descriptor correspondiente.
    #   - nbins: valor de tipo entero que indica el numero de niveles que tiene el histograma 
    #           para calcular el descriptor correspondiente.
    #   - tipoDesc: cadena de caracteres que indica el tipo de descriptor calculado
    #
    # Argumentos de salida
    #   - descriptores: numpy array con dimensiones [num_puntos_interes, nbins] con los descriptores 
    #                   de cada punto de interes (i.e. histograma de niveles de gris)
    #   - new_coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2], solamente con las coordenadas 
    #                      de los puntos de interes descritos. Tipo int64  <class 'numpy.ndarray'>
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada vtam y nbins, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    """
    # Iniciamos variables de salida
    descriptores = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)
    new_coords_esquinas = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)

    #incluya su codigo aqui
    # Antes de operar convierta la imagen al formato float en el rango [0,1]
    # Normalización manual (sin usar img_as_float)
    img = imagen.astype(np.float64)
    
    # Si la imagen es de color, convertir a escala de grises
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    
    # Normalizar al rango [0, 1] manualmente
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img - img_min

    # Identificar el descriptor con tipoDesc='hist'
    if tipoDesc == 'hist':
        # El descriptor considera un vecindario vtam+1 x vtam+1 centrado en la
        # coordenada de la esquina para la cual se extrae el descriptor
        
        # Listas para ir guardando descriptores y coordenadas válidas
        lista_desc = []
        lista_coords = []

        # Dimensiones imagen y radio de vecindario (para tamaño total vtam+1)
        alto, ancho = img.shape
        radio = vtam // 2

        # Recorrer cada punto de interés
        for coord in coords_esquinas:
            fila = int(coord[0])
            col = int(coord[1])

            # Comprobar que el vecindario completo cabe dentro de la imagen
            if (fila - radio >= 0 and fila + radio + 1 <= alto and
                col - radio >= 0 and col + radio + 1 <= ancho):

                # Extraer vecindario (vtam+1) x (vtam+1)
                patch = img[fila - radio:fila + radio + 1,
                            col - radio:col + radio + 1]

                # Aplanar y calcular histograma con nbins, intervalos homogéneos [a,b) en [0,1]
                vals = patch.flatten()
                hist, _ = np.histogram(vals, bins=nbins, range=(0, 1))

                # Normalizar (suma total = 1)
                total = np.sum(hist)
                if total > 0:
                    hist = hist.astype(np.float64) / total
                else:
                    hist = hist.astype(np.float64)

                lista_desc.append(hist)
                lista_coords.append([fila, col])

        # Transformar listas a arrays de numpy
        if len(lista_desc) > 0:
            descriptores = np.asarray(lista_desc)
            new_coords_esquinas = np.asarray(lista_coords, dtype=np.int64)
        else:
            descriptores = np.empty((0, nbins))
            new_coords_esquinas = np.empty((0, 2), dtype=np.int64)
        
    elif tipoDesc == 'mag-ori':
        # Calcula el gradiente de la imagen
        # Considerar nbins de cuantificación homogeneos entre 0 y 360 grados
        # En cada nivel de cuantificación
        pass
    

    

    return descriptores, new_coords_esquinas

if __name__ == "__main__":    
    print("Practica 2 - Tarea 2 - Test autoevaluación\n")                

    # Parche para compatibilidad con tests que usan fig.canvas.set_window_title
    try:
        import matplotlib
        # Asegurar backend TkAgg si es posible
        try:
            matplotlib.use('TkAgg')
        except Exception:
            pass
        try:
            import matplotlib.backends.backend_tkagg as mtk
            if not hasattr(mtk.FigureCanvasTkAgg, 'set_window_title'):
                def _set_window_title(self, title):
                    try:
                        # En versiones modernas se usa el manager
                        self.manager.set_window_title(title)
                    except Exception:
                        pass
                mtk.FigureCanvasTkAgg.set_window_title = _set_window_title
            # Activar modo interactivo para evitar bloqueos prolongados en pause()
            try:
                import matplotlib.pyplot as plt
                plt.ion()
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    # Importar tests después del parche
    from p2_tests import test_p2_tarea2

    ## tests descriptor tipo 'hist' (tarea 2a)
    # print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=False,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test
    print("Tests completados = " + str(test_p2_tarea2(disptime=0.0005,stop_at_error=False,debug=False,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test, mostrar imagenes con resultados (1 segundo)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test, pararse en errores y mostrar datos
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist',imgIdx = 3, poiIdx = 7))) #analizar solamente imagen #2 y esquina #7    

    ## tests descriptor tipo 'mag-ori' (tarea 2b)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=False,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test
    #print("Tests completados = " + str(test_p2_tarea2(disptime=0.1,stop_at_error=False,debug=False,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test, mostrar imagenes con resultados (1 segundo)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test, pararse en errores y mostrar datos
    #print("Tests completados = " + str(test_p2_tarea2(disptime=1,stop_at_error=True,debug=True,tipoDesc='mag-ori',imgIdx = 3,poiIdx = 7))) #analizar solamente imagen #1 y esquina #7       