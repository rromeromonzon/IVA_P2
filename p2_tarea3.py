# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 3:  Similitud y correspondencia de puntos de interes

# AUTOR1: ROMERO MONZÓN, RAFAEL
# AUTOR2: LANDALUCE FERNÁNDEZ, ABEL
# PAREJA/TURNO: 13/NUMERO_TURNO

# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea3

# Incluya aqui las librerias que necesite en su codigo
# ...

def correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2, tipoCorr='mindist',max_distancia=25):
    """
    # Esta funcion determina la correspondencias entre dos conjuntos de descriptores mediante
    # el calculo de la similitud entre los descriptores.
    #
    # El parametro 'tipoCorr' determina el criterio de similitud aplicado 
    # para establecer correspondencias entre pares de descriptores:
    #   - Criterio 'mindist': minima distancia euclidea entre descriptores 
    #                         menor que el umbral 'max_distancia'
    #  
    # Argumentos de entrada:
    #   - descriptores1: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
    #                    con los descriptores de los puntos de interes de la imagen 1.        
    #   - descriptores2: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
    #                    con los descriptores de los puntos de interes de la imagen 2.        
    #   - tipoCorr: cadena de caracteres que indica el tipo de criterio para establecer correspondencias
    #   - max_distancia: valor de tipo double o float utilizado por el criterio 'mindist' y 'nndr', 
    #                    que determina si se aceptan correspondencias entre descriptores 
    #                    con distancia minima menor que 'max_distancia' 
    #
    # Argumentos de salida
    #   - correspondencias: numpy array con dimensiones [numero_correspondencias, 2] de tipo int64 
    #                       que determina correspondencias entre descriptores de imagen 1 e imagen 2.
    #                       Por ejemplo: 
    #                       correspondencias[0,:]=[5,22] significa que el descriptor 5 de la imagen 1 
    #                                                  corresponde con el descriptor 22 de la imagen 2. 
    #                       correspondencias[1,:]=[6,23] significa que el descriptor 6 de la imagen 1 
    #                                                  corresponde con el descriptor 23 de la imagen 2.
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada tipoCorr y max_distancia, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    #
    # CONSIDERACIONES: 
    # 1) La funcion no debe permitir correspondencias de uno a varios descriptores. Es decir, 
    #   un descriptor de la imagen 1 no puede asignarse a multiples descriptores de la imagen 2 
    # 2) En el caso de que existan varios descriptores de la imagen 2 con la misma distancia minima 
    #    con algún descriptor de la imagen 1, seleccione el descriptor de la imagen 2 con 
    #    indice/posicion menor. Por ejemplo, si las correspondencias [5,22] y [5,23] tienen la misma
    #    distancia minima, seleccione [5,22] al ser el indice 22 menor que 23
    """    
    correspondencias = np.empty(shape=[0,2]) # iniciamos la variable de salida (numpy array)

    # Validaciones rápidas
    if descriptores_imagen1 is None or descriptores_imagen2 is None:
        return correspondencias
    if descriptores_imagen1.size == 0 or descriptores_imagen2.size == 0:
        return correspondencias

    if tipoCorr == 'mindist':
        # Para cada descriptor de la imagen 1, buscamos el más cercano en la imagen 2
        lista_corr = []
        usados_img2 = set()

        for i, d1 in enumerate(descriptores_imagen1):
            # Distancias euclídeas a todos los descriptores de la imagen 2
            diffs = descriptores_imagen2 - d1  # broadcasting sobre filas
            dists = np.linalg.norm(diffs, axis=1)

            # Recorremos candidatos en orden ascendente de distancia
            orden = np.argsort(dists)
            asignado = False
            for j in orden:
                dj = dists[j]
                if dj < max_distancia and j not in usados_img2:
                    lista_corr.append([i, int(j)])
                    usados_img2.add(int(j))
                    asignado = True
                    break
            # Si no encuentra candidato válido, no se añade correspondencia para i

        if len(lista_corr) > 0:
            correspondencias = np.asarray(lista_corr, dtype=np.int64)
        else:
            correspondencias = np.empty((0, 2), dtype=np.int64)

    return correspondencias

if __name__ == "__main__":
    print("Practica 2 - Tarea 3 - Test autoevaluación\n")               

    ## tests correspondencias tipo 'minDist' (tarea 3a)
    # print("Tests completados = " + str(test_p2_tarea3(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'hist' y ver errores
    #print("Tests completados = " + str(test_p2_tarea3(disptime=-1,stop_at_error=False,debug=False,tipoDesc='hist',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'hist'
    print("Tests completados = " + str(test_p2_tarea3(disptime=1,stop_at_error=False,debug=False,tipoDesc='mag-ori',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'mag-ori'