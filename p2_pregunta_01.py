# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta 01

# AUTOR1: ROMERO MONZÓN, RAFAEL
# AUTOR2: LANDALUCE FERNÁNDEZ, ABEL
# PAREJA/TURNO: 13/NUMERO_TURNO

import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from p2_tarea2 import descripcion_puntos_interes 


def main_visualizacion_descriptores():
    # 1. Cargar la imagen y definir el punto de interés
    imagen = data.camera()

    centro_fila = 150 
    centro_col = 200
    coords_esquinas = np.array([[centro_fila, centro_col]])
    
    # Definir las combinaciones a probar
    combinaciones = [
        (8, 16, 'A: vtam=8, nbins=16'),
        (16, 16, 'B: vtam=16, nbins=16'),
        (8, 32, 'C: vtam=8, nbins=32'),
        (16, 32, 'D: vtam=16, nbins=32')
    ]
    
    resultados = {}
    
    # 2. Calcular descriptores para cada combinación
    print(f"Calculando descriptores 'hist' en la coordenada ({centro_fila}, {centro_col})...")
    for vtam, nbins, etiqueta in combinaciones:
        # Llama a la función implementada en p2_tarea2
        desc, _ = descripcion_puntos_interes(
            imagen, coords_esquinas, vtam=vtam, nbins=nbins, tipoDesc='hist'
        )
        if desc.size > 0:
            resultados[etiqueta] = desc[0]
            print(f"Calculado: {etiqueta} (Longitud: {desc[0].shape[0]})")
        else:
            print(f"ATENCIÓN: {etiqueta} no se pudo calcular (punto fuera de bordes para vtam={vtam}).")

    # 3. Generar la gráfica para mostrar los cambios
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    fig.suptitle('Comparación de Descriptores de Histograma (Tipo "hist")')
    
    ancho_barra_16 = 0.4
    ancho_barra_32 = 0.4

    # --- a) Gráfica para nbins=16 (Panel Superior) ---
    ax0 = axes[0]
    desc_A = resultados['A: vtam=8, nbins=16']
    desc_B = resultados['B: vtam=16, nbins=16']
    
    x_bins_16 = np.arange(16)
    
    # Barras desplazadas para comparación clara
    ax0.bar(x_bins_16 - ancho_barra_16/2, desc_A, width=ancho_barra_16, label='vtam=8 (9x9)', align='center')
    ax0.bar(x_bins_16 + ancho_barra_16/2, desc_B, width=ancho_barra_16, label='vtam=16 (17x17)', align='center')
    
    ax0.set_title('nbins = 16 (Resolución Baja)')
    ax0.set_ylabel('Frecuencia Normalizada')
    ax0.set_xticks(x_bins_16)
    ax0.set_xticklabels(x_bins_16)
    ax0.legend()

    # --- b) Gráfica para nbins=32 (Panel Inferior) ---
    ax1 = axes[1]
    desc_C = resultados['C: vtam=8, nbins=32']
    desc_D = resultados['D: vtam=16, nbins=32']

    x_bins_32 = np.arange(32)
    
    # CORRECCIÓN: Usar ax1.bar con desplazamiento para visualizar el histograma
    ax1.bar(x_bins_32 - ancho_barra_32/2, desc_C, width=ancho_barra_32, label='vtam=8 (9x9)', align='center')
    ax1.bar(x_bins_32 + ancho_barra_32/2, desc_D, width=ancho_barra_32, label='vtam=16 (17x17)', align='center')

    ax1.set_title('nbins = 32 (Resolución Alta)')
    ax1.set_ylabel('Frecuencia Normalizada')
    ax1.set_xlabel('Bin (Nivel de Gris)')
    ax1.set_xticks(x_bins_32[::4]) # Mostrar cada 4 bins
    
    ax1.legend()
    
    plt.tight_layout()
    plt.show()

# Ejecutar la función principal
if __name__ == '__main__':
    main_visualizacion_descriptores()