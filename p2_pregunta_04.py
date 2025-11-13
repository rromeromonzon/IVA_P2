"""
P2 - Pregunta 04: Ejemplos Visuales del Criterio NNDR
Muestra comparaciones visuales entre MinDist y NNDR usando imágenes de skimage.data
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import data
from skimage.util import img_as_ubyte
from skimage.color import gray2rgb
import cv2

# Parche para compatibilidad con matplotlib
if not hasattr(FigureCanvasTkAgg, 'set_window_title'):
    FigureCanvasTkAgg.set_window_title = lambda self, title: self.manager.set_window_title(title)

plt.ion()

# Importar funciones de las tareas
from p2_tarea1 import detectar_puntos_interes_harris
from p2_tarea2 import descripcion_puntos_interes
from p2_tarea3 import correspondencias_puntos_interes


def crear_tablero_ajedrez(n=8, tamaño_casilla=10):
    """
    Crea un tablero de ajedrez NxN con casillas de tamaño especificado
    
    Args:
        n: número de casillas por lado (default 8)
        tamaño_casilla: tamaño en píxeles de cada casilla (default 10)
    
    Returns:
        imagen RGB del tablero
    """
    # Crear patrón de tablero
    tablero = np.zeros((n * tamaño_casilla, n * tamaño_casilla), dtype=np.uint8)
    
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                tablero[i*tamaño_casilla:(i+1)*tamaño_casilla, 
                       j*tamaño_casilla:(j+1)*tamaño_casilla] = 255
    
    # Convertir a RGB
    tablero_rgb = gray2rgb(tablero)
    return tablero_rgb


def cargar_imagen_skimage(nombre):
    """
    Carga una imagen de skimage.data
    
    Args:
        nombre: nombre de la imagen ('Astronaut', 'Cofee', 'Coins', 'Rocket')
    
    Returns:
        imagen RGB en formato uint8
    """
    if nombre == 'Astronaut':
        return data.astronaut()
    elif nombre == 'Cofee' or nombre == 'Coffee':
        return data.coffee()
    elif nombre == 'Coins':
        # Convertir a RGB (es grayscale)
        return gray2rgb(data.coins())
    elif nombre == 'Rocket':
        return data.rocket()
    else:
        raise ValueError(f"Imagen '{nombre}' no reconocida")


def cargar_imagen_archivo(ruta):
    """Carga una imagen desde un archivo y la convierte a RGB"""
    img = cv2.imread(ruta)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def aplicar_transformacion(img, tipo='original'):
    """
    Aplica una transformación a la imagen para crear una segunda vista
    
    Args:
        img: imagen original
        tipo: tipo de transformación ('original', 'rotation_cw', 'rotation_ccw', 'flip_horizontal')
    
    Returns:
        imagen transformada
    """
    h, w = img.shape[:2]
    
    if tipo == 'original':
        return img.copy()
    elif tipo == 'rotation_cw':
        # Rotar 4 grados sentido horario y ampliar el canvas
        angle = 4
        center = (w // 2, h // 2)
        
        # Calcular el nuevo tamaño necesario para contener toda la imagen rotada
        angle_rad = np.radians(angle)
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        
        # Obtener matriz de rotación
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Ajustar la traslación para centrar la imagen en el nuevo canvas
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Aplicar la transformación con el nuevo tamaño (SIN recortar)
        return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))
    elif tipo == 'rotation_ccw':
        # Rotar 4 grados sentido antihorario y ampliar el canvas
        angle = -4
        center = (w // 2, h // 2)
        
        # Calcular el nuevo tamaño necesario
        angle_rad = np.radians(abs(angle))
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        
        # Obtener matriz de rotación
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Ajustar la traslación
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))
    elif tipo == 'flip_horizontal':
        # Invertir horizontalmente y rotar 4 grados
        img_flipped = cv2.flip(img, 1)
        
        # Aplicar rotación de 4 grados
        angle = 4
        center = (w // 2, h // 2)
        
        # Calcular el nuevo tamaño necesario
        angle_rad = np.radians(angle)
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        
        # Obtener matriz de rotación
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Ajustar la traslación
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        return cv2.warpAffine(img_flipped, M, (new_w, new_h), borderValue=(255, 255, 255))
    else:
        return img.copy()


def mostrar_correspondencias_comparacion(img1, coords1, img2, coords2, 
                                        corr_mindist, corr_nndr, titulo_base):
    """
    Muestra una comparación lado a lado de MinDist vs NNDR
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h_max = max(h1, h2)
    
    # Crear figura con 2 subplots verticales (tamaño maximizado para ocupar toda la pantalla)
    fig = plt.figure(figsize=(14, 8))
    
    # Colores consistentes para comparación
    np.random.seed(42)
    
    # --- SUBPLOT 1: MinDist (arriba) ---
    ax1 = plt.subplot(2, 1, 1)
    img_combined1 = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    img_combined1[:h1, :w1] = img1
    img_combined1[:h2, w1:] = img2
    ax1.imshow(img_combined1)
    
    print(f"    Dibujando {len(corr_mindist)} correspondencias MinDist...")
    # Dibujar correspondencias MinDist
    colors_mindist = np.random.rand(len(corr_mindist), 3)
    lines_drawn = 0
    for idx, (i, j) in enumerate(corr_mindist):
        y1, x1_coord = coords1[i]
        y2, x2_coord = coords2[j]
        x2_offset = x2_coord + w1
        
        # Verificar que las coordenadas estén dentro de los límites
        if 0 <= x1_coord < w1 and 0 <= y1 < h1 and 0 <= x2_coord < w2 and 0 <= y2 < h2:
            # Líneas más gruesas y opacas
            ax1.plot([x1_coord, x2_offset], [y1, y2], '-', 
                    color=colors_mindist[idx], linewidth=2.5, alpha=0.9)
            # Puntos más grandes y con borde más grueso
            ax1.plot(x1_coord, y1, 'o', color=colors_mindist[idx], 
                    markersize=8, markeredgecolor='yellow', markeredgewidth=2)
            ax1.plot(x2_offset, y2, 'o', color=colors_mindist[idx], 
                    markersize=8, markeredgecolor='yellow', markeredgewidth=2)
            lines_drawn += 1
        else:
            print(f"      ⚠️  Correspondencia {idx} fuera de límites: ({x1_coord},{y1}) -> ({x2_coord},{y2})")
    
    print(f"      ✓ {lines_drawn} líneas dibujadas (de {len(corr_mindist)} correspondencias)")
    
    ax1.set_title('MinDist', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # --- SUBPLOT 2: NNDR (abajo) ---
    ax2 = plt.subplot(2, 1, 2)
    img_combined2 = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    img_combined2[:h1, :w1] = img1
    img_combined2[:h2, w1:] = img2
    ax2.imshow(img_combined2)
    
    print(f"    Dibujando {len(corr_nndr)} correspondencias NNDR...")
    # Dibujar correspondencias NNDR
    np.random.seed(42)  # Resetear para obtener colores similares
    colors_nndr = np.random.rand(len(corr_mindist), 3)  # Misma cantidad para mapeo
    
    # Crear mapeo de correspondencias NNDR para usar colores consistentes
    nndr_set = set(map(tuple, corr_nndr.tolist()))
    
    lines_drawn_nndr = 0
    for idx, (i, j) in enumerate(corr_mindist):
        if (i, j) in nndr_set:
            y1, x1_coord = coords1[i]
            y2, x2_coord = coords2[j]
            x2_offset = x2_coord + w1
            
            # Verificar que las coordenadas estén dentro de los límites
            if 0 <= x1_coord < w1 and 0 <= y1 < h1 and 0 <= x2_coord < w2 and 0 <= y2 < h2:
                # Correspondencia aceptada por NNDR (mismo color, más grueso)
                ax2.plot([x1_coord, x2_offset], [y1, y2], '-', 
                        color=colors_nndr[idx], linewidth=2.5, alpha=0.9)
                ax2.plot(x1_coord, y1, 'o', color=colors_nndr[idx], 
                        markersize=8, markeredgecolor='yellow', markeredgewidth=2)
                ax2.plot(x2_offset, y2, 'o', color=colors_nndr[idx], 
                        markersize=8, markeredgecolor='yellow', markeredgewidth=2)
                lines_drawn_nndr += 1
                print(f"      NNDR [{i:3d} -> {j:3d}]: punto1=({x1_coord:4.0f},{y1:4.0f}) punto2=({x2_coord:4.0f},{y2:4.0f})")
            else:
                print(f"      ⚠️  Correspondencia NNDR {idx} fuera de límites: ({x1_coord},{y1}) -> ({x2_coord},{y2})")
    
    print(f"      ✓ {lines_drawn_nndr} líneas dibujadas (de {len(corr_nndr)} correspondencias NNDR)")
    
    ax2.set_title('NNDR (umbral=0.75)', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # Calcular porcentaje de retención
    retencion = (len(corr_nndr) / len(corr_mindist) * 100) if len(corr_mindist) > 0 else 0
    
    # Título general con porcentaje de retención
    fig.suptitle(f'{titulo_base} (Retención: {retencion:.1f}%)', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Mostrar la figura inmediatamente y esperar a que se cierre
    print(f"\n  📊 Mostrando figura. Cierra la ventana para continuar...")
    plt.show(block=True)  # Bloquear hasta que se cierre la ventana
    
    return fig


def analizar_par_imagenes(nombre_img, transformacion, nombre_par, tipoDesc='hist', nbins=16, vtam=4, es_archivo=False):
    """
    Analiza un par de imágenes (original y transformada) y genera visualización comparativa
    
    Args:
        nombre_img: nombre de la imagen de skimage ('Astronaut', 'Cofee') o ruta de archivo
        transformacion: tipo de transformación ('rotation')
        nombre_par: nombre descriptivo del par
        tipoDesc: tipo de descriptor
        nbins: número de bins para histogramas
        vtam: tamaño de ventana para descriptores
        es_archivo: True si nombre_img es una ruta de archivo, False si es de skimage
    """
    print(f"\n{'='*70}")
    print(f"ANALIZANDO: {nombre_par}")
    print(f"{'='*70}")
    
    # Cargar imagen original
    if es_archivo:
        print(f"  Cargando imagen desde archivo '{nombre_img}' y aplicando transformación '{transformacion}'...")
        img1 = cargar_imagen_archivo(nombre_img)
    else:
        print(f"  Cargando imagen '{nombre_img}' y aplicando transformación '{transformacion}'...")
        img1 = cargar_imagen_skimage(nombre_img)
    
    if img1 is None:
        print(f"  ✗ Error al cargar imagen")
        return None
    
    # Crear imagen transformada
    img2 = aplicar_transformacion(img1, transformacion)
    
    print(f"    Imagen original: {img1.shape}")
    print(f"    Imagen transformada: {img2.shape}")
    
    # Extraer puntos de interés
    print(f"\n  Extrayendo puntos de interés (Harris)...")
    coords1 = detectar_puntos_interes_harris(img1, threshold_rel=0.15)
    coords2 = detectar_puntos_interes_harris(img2, threshold_rel=0.15)
    print(f"    Imagen 1: {len(coords1)} puntos")
    print(f"    Imagen 2: {len(coords2)} puntos")
    
    # Describir puntos
    print(f"\n  Calculando descriptores (tipo='{tipoDesc}', nbins={nbins}, vtam={vtam})...")
    desc1, coords1_valid = descripcion_puntos_interes(
        img1, coords1, tipoDesc=tipoDesc, nbins=nbins, vtam=vtam
    )
    desc2, coords2_valid = descripcion_puntos_interes(
        img2, coords2, tipoDesc=tipoDesc, nbins=nbins, vtam=vtam
    )
    print(f"    Imagen 1: {len(desc1)} descriptores válidos")
    print(f"    Imagen 2: {len(desc2)} descriptores válidos")
    
    # Correspondencias MinDist
    print(f"\n  Calculando correspondencias MinDist...")
    corr_mindist = correspondencias_puntos_interes(
        desc1, desc2, tipoCorr='mindist', max_distancia=25
    )
    print(f"    Correspondencias MinDist: {len(corr_mindist)}")
    
    # Correspondencias NNDR
    print(f"\n  Calculando correspondencias NNDR (umbral=0.75)...")
    corr_nndr = correspondencias_puntos_interes(
        desc1, desc2, tipoCorr='nndr', max_distancia=25
    )
    print(f"    Correspondencias NNDR: {len(corr_nndr)}")
    
    # Análisis
    filtradas = len(corr_mindist) - len(corr_nndr)
    porcentaje = 100 * len(corr_nndr) / max(1, len(corr_mindist))
    
    print(f"\n  📊 RESUMEN:")
    print(f"    • MinDist:  {len(corr_mindist)} correspondencias")
    print(f"    • NNDR:     {len(corr_nndr)} correspondencias")
    print(f"    • Filtradas: {filtradas} correspondencias ambiguas")
    print(f"    • Retenido:  {porcentaje:.1f}%")
    
    # Calcular ratios de algunas correspondencias NNDR para mostrar
    if len(corr_nndr) > 0:
        print(f"\n  🔍 Ejemplos de ratios en correspondencias NNDR:")
        # Mostrar hasta 5 ejemplos
        n_ejemplos = min(5, len(corr_nndr))
        for idx in range(n_ejemplos):
            i, j = corr_nndr[idx]
            d1 = desc1[i]
            dists = np.linalg.norm(desc2 - d1, axis=1)
            dists_sorted = np.sort(dists)
            if len(dists_sorted) >= 2:
                ratio = dists_sorted[0] / dists_sorted[1]
                print(f"    [{i:3d} -> {j:3d}]: ratio={ratio:.3f} "
                      f"(dist1={dists_sorted[0]:.3f}, dist2={dists_sorted[1]:.3f})")
    
    # Generar visualización
    print(f"\n  Generando visualización...")
    fig = mostrar_correspondencias_comparacion(
        img1, coords1_valid, img2, coords2_valid,
        corr_mindist, corr_nndr, nombre_par
    )
    
    return fig, {
        'nombre': nombre_par,
        'mindist': len(corr_mindist),
        'nndr': len(corr_nndr),
        'filtradas': filtradas,
        'porcentaje': porcentaje
    }


def main():
    """
    Función principal - analiza pares de imágenes con diferentes transformaciones
    Usa solo imágenes de skimage.data
    """
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "EJEMPLOS VISUALES CRITERIO NNDR - P2 PREGUNTA 04" + " " * 10 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Usando solo imágenes de skimage.data")
    print("  • Transformaciones: Rotación +4°, Rotación -4°, Inversión horizontal + Rotación +4°")
    print()
    
    # Lista de imágenes a procesar - SOLO skimage (las mejores)
    imagenes = [
        ('skimage', 'Astronaut', 'Astronaut'),
        ('skimage', 'Cofee', 'Cofee'),
        ('skimage', 'Coins', 'Coins'),
        ('skimage', 'Rocket', 'Rocket'),
    ]
    
    # Transformaciones a aplicar
    transformaciones = [
        ('rotation_cw', 'Rotación +4°'),
        ('rotation_ccw', 'Rotación -4°'),
        ('flip_horizontal', 'Inversión horizontal + Rotación +4°'),
    ]
    
    figuras = []
    resultados = []
    
    # Procesar cada imagen con cada transformación
    for fuente, nombre_img, nombre_base in imagenes:
        es_archivo = fuente != 'skimage'
        img_path = fuente if es_archivo else nombre_img
        
        for transf_tipo, transf_nombre in transformaciones:
            try:
                nombre_completo = f"{nombre_base} - {transf_nombre}"
                
                resultado = analizar_par_imagenes(
                    img_path, 
                    transf_tipo,
                    nombre_completo,
                    es_archivo=es_archivo
                )
                if resultado is not None:
                    fig, info = resultado
                    figuras.append(fig)
                    resultados.append(info)
            except Exception as e:
                print(f"\n✗ Error procesando '{nombre_completo}': {e}")
                import traceback
                traceback.print_exc()
    
    # Resumen global
    print(f"\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "RESUMEN GLOBAL" + " " * 34 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    print(f"{'Escena':<50} {'MinDist':>10} {'NNDR':>10} {'Filtradas':>10} {'Retenido':>10}")
    print("-" * 90)
    
    for info in resultados:
        print(f"{info['nombre']:<50} {info['mindist']:>10} {info['nndr']:>10} "
              f"{info['filtradas']:>10} {info['porcentaje']:>9.1f}%")
    
    if len(resultados) > 0:
        avg_total = np.mean([r['porcentaje'] for r in resultados])
        total_mindist = sum([r['mindist'] for r in resultados])
        total_nndr = sum([r['nndr'] for r in resultados])
        total_filtradas = sum([r['filtradas'] for r in resultados])
        
        print("-" * 90)
        print(f"{'TOTALES':<50} {total_mindist:>10} {total_nndr:>10} {total_filtradas:>10}")
        print(f"{'PROMEDIO DE RETENCIÓN':<50} {'':<10} {'':<10} {'':<10} {avg_total:>9.1f}%")
    
    print()
    print("INTERPRETACIÓN:")
    print("  • NNDR filtra correspondencias donde el segundo vecino más cercano")
    print("    está demasiado cerca del primero (ratio < 0.75)")
    print("  • Esto elimina matches ambiguos y mejora la calidad")
    print("  • Las correspondencias retenidas son más confiables")
    print()
    print("TRANSFORMACIONES APLICADAS:")
    print("  • Rotación +4°: Sentido horario alrededor del centro")
    print("  • Rotación -4°: Sentido antihorario alrededor del centro")
    print("  • Inversión horizontal + Rotación +4°: Reflejo + rotación")
    print()
    print("IMÁGENES USADAS:")
    print("  • skimage.data: Astronaut, Cofee, Coins, Rocket")
    print()
    
    # Mantener ventanas abiertas
    print(f"\n{'='*70}")
    print(f"Total de figuras mostradas: {len(figuras)}")
    print("Todas las figuras han sido cerradas.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
