# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta 01

# AUTOR1: ROMERO MONZÓN, RAFAEL
# AUTOR2: LANDALUCE FERNÁNDEZ, ABEL
# PAREJA/TURNO: 13/NUMERO_TURNO

# =========================================================================
# FUNCIONES DE TAREA (T1, T2, T3)
# =========================================================================

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import corner_peaks
import matplotlib.pyplot as plt
import os

from p2_tarea2 import descripcion_puntos_interes 
from p2_tarea1 import detectar_puntos_interes_harris
from p2_tarea3 import correspondencias_puntos_interes

def plot_correspondences(img1, coords1, img2, coords2, corr, title, color='r'):
    if img1.ndim == 3: img1_gray = np.mean(img1, axis=2)
    else: img1_gray = img1
    if img2.ndim == 3: img2_gray = np.mean(img2, axis=2)
    else: img2_gray = img2

    # Concatenar imágenes horizontalmente
    H1, W1 = img1_gray.shape
    H2, W2 = img2_gray.shape
    
    combined_img = np.zeros((max(H1, H2), W1 + W2), dtype=np.float64)
    combined_img[:H1, :W1] = img1_gray
    combined_img[:H2, W1:] = img2_gray

    plt.figure(figsize=(15, 7))
    plt.imshow(combined_img, cmap='gray')
    plt.title(title)
    plt.axis('off')

    # Dibujar líneas de correspondencia
    for i1, i2 in corr:
        r1, c1 = coords1[i1]
        r2, c2 = coords2[i2]
        c2_shifted = c2 + W1
        
        plt.plot([c1, c2_shifted], [r1, r2], color=color, linestyle='-', linewidth=0.8, marker='o', markersize=3)
    
    plt.show()


def analisis_p2_3(img_dir='img', nbins=16, max_distancia=0.2):
    """
    Aplica Detección, Descripción y Correspondencia a pares de imágenes 
    y analiza las diferencias entre descriptores 'hist' y 'mag-ori' usando los pares de test.
    """
    
    image_pairs = [
        ('EGaudi_1.jpg', 'EGaudi_2.jpg', 'Cambio de Perspectiva y Distorsión'),
        ('Mount_Rushmore1.jpg', 'Mount_Rushmore2.jpg', 'Cambio de Perspectiva y Escala'),
        ('NotreDame1.jpg', 'NotreDame2.jpg', 'Cambio de Rotación y Punto de Vista')
    ]
    
    print("--- Análisis P2.3: Correspondencias entre pares de imágenes de Monumentos ---")
    
    for img1_name, img2_name, title in image_pairs:
        try:
            # Cargar Imágenes 
            img1_path = os.path.join(img_dir, img1_name)
            img2_path = os.path.join(img_dir, img2_name)
            
            img1 = plt.imread(img1_path)
            img2 = plt.imread(img2_path)

            print(f"\n## Imágenes: {img1_name} & {img2_name} ({title})")

            # Detección de Puntos de Interés (Tarea 1)
            coords1 = detectar_puntos_interes_harris(img1)
            coords2 = detectar_puntos_interes_harris(img2)
            
            print(f"Esquinas detectadas en Img1: {len(coords1)}, en Img2: {len(coords2)}")
                
            print("\n Descriptor 'hist' (Intensidad)")
            desc1_hist, new_coords1_hist = descripcion_puntos_interes(img1, coords1, nbins=nbins, tipoDesc='hist')
            desc2_hist, new_coords2_hist = descripcion_puntos_interes(img2, coords2, nbins=nbins, tipoDesc='hist')
            
            corr_hist = correspondencias_puntos_interes(desc1_hist, desc2_hist, max_distancia=max_distancia)
            
            print(f"  Correspondencias 'hist' encontradas: {len(corr_hist)}")
            plot_correspondences(img1, new_coords1_hist, img2, new_coords2_hist, corr_hist, 
                                 f"{title} 'hist' ({len(corr_hist)}/{len(desc1_hist)})", color='blue')

            print("\n Descriptor 'mag-ori' (Gradiente)")
            desc1_magori, new_coords1_magori = descripcion_puntos_interes(img1, coords1, nbins=nbins, tipoDesc='mag-ori')
            desc2_magori, new_coords2_magori = descripcion_puntos_interes(img2, coords2, nbins=nbins, tipoDesc='mag-ori')

            corr_magori = correspondencias_puntos_interes(desc1_magori, desc2_magori, max_distancia=max_distancia)
            
            print(f"  Correspondencias 'mag-ori' encontradas: {len(corr_magori)}")
            plot_correspondences(img1, new_coords1_magori, img2, new_coords2_magori, corr_magori, 
                                 f"{title} 'mag-ori' ({len(corr_magori)}/{len(desc1_magori)})", color='red')
        
        except FileNotFoundError:
            print(f"\n[ERROR] Archivos no encontrados para el par {img1_name} y {img2_name}. Asegúrate de que estén en la carpeta '{img_dir}'.")
        except Exception as e:
            print(f"\n[ERROR] Ocurrió un error al procesar el par {img1_name} y {img2_name}: {e}")

if __name__ == '__main__':
    analisis_p2_3(max_distancia=0.2)