import os
import sys
import glob
import shutil
import random
import pygame
import numpy as np
import pandas as pd
from skimage import io, color, morphology, measure, img_as_ubyte
from skimage.morphology import remove_small_objects
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

# Funciones auxiliares para el juego Memorama
def display_grid(screen, memorama_array, matched_indices, font, grid_size=10, card_size=80, padding=10):
    screen.fill((30, 30, 30))  # Fondo oscuro
    for idx, card in enumerate(memorama_array):
        row = idx // grid_size
        col = idx % grid_size
        x = padding + col * (card_size + padding)
        y = padding + row * (card_size + padding)
        rect = pygame.Rect(x, y, card_size, card_size)
        if matched_indices[idx]:
            pygame.draw.rect(screen, (0, 200, 0), rect)  # Verde para pares encontrados
            label = font.render("X", True, (255, 255, 255))
            screen.blit(label, (x + card_size//2 - label.get_width()//2, y + card_size//2 - label.get_height()//2))
        else:
            pygame.draw.rect(screen, (200, 200, 200), rect)  # Gris para cartas ocultas
            label = font.render(str(idx+1), True, (0, 0, 0))
            screen.blit(label, (x + card_size//2 - label.get_width()//2, y + card_size//2 - label.get_height()//2))
    pygame.display.flip()

def get_player_selection(memorama_array, matched_indices, font, screen, grid_size=10, card_size=80, padding=10):
    selected = []
    while len(selected) < 2:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                col = pos[0] // (card_size + padding)
                row = pos[1] // (card_size + padding)
                idx = row * grid_size + col
                if 0 <= idx < len(memorama_array) and not matched_indices[idx] and idx not in selected:
                    selected.append(idx)
                    # Mostrar selección
                    screen.fill((30, 30, 30))
                    display_grid(screen, memorama_array, matched_indices, font, grid_size, card_size, padding)
                    for s in selected:
                        row_s = s // grid_size
                        col_s = s % grid_size
                        x = padding + col_s * (card_size + padding)
                        y = padding + row_s * (card_size + padding)
                        rect = pygame.Rect(x, y, card_size, card_size)
                        pygame.draw.rect(screen, (0, 0, 255), rect)  # Azul para selección
                        label = font.render(memorama_array[s], True, (255, 255, 255))
                        screen.blit(label, (x + card_size//2 - label.get_width()//2, y + card_size//2 - label.get_height()//2))
                    pygame.display.flip()
        pygame.time.wait(100)
    return selected[0], selected[1]

def get_image_class_counts(resultados_totales, image_name):
    subset = resultados_totales[resultados_totales['Nombre_Imagen'] == image_name]
    clases = subset['Etiqueta'].unique()
    counts = {}
    for clase in clases:
        counts[clase] = (subset['Etiqueta'] == clase).sum()
    return counts

def display_selected_images(folder_path, img1, img2):
    img1_path = os.path.join(folder_path, img1)
    img2_path = os.path.join(folder_path, img2)
    
    if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
        print('Una o ambas imágenes seleccionadas no existen en la ruta especificada.')
        return
    
    img1_display = io.imread(img1_path)
    img2_display = io.imread(img2_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img1_display)
    axes[0].set_title(f'Imagen 1: {img1}')
    axes[0].axis('off')
    
    axes[1].imshow(img2_display)
    axes[1].set_title(f'Imagen 2: {img2}')
    axes[1].axis('off')
    
    plt.show()
    plt.pause(3)  # Mostrar por 3 segundos
    plt.close(fig)

# Ruta de la carpeta que contiene las imágenes
folder_path = r'C:\Users\albsa\Desktop\Vision Artificial\DBVision'

# Verificar si la carpeta existe
if not os.path.isdir(folder_path):
    raise FileNotFoundError(f'La carpeta especificada no existe: {folder_path}')

# Obtener la lista de archivos de imagen en la carpeta
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))

# Verificar si hay imágenes en la carpeta
if not image_files:
    raise FileNotFoundError(f'No se encontraron imágenes en la carpeta: {folder_path}')

# Definir el tamaño mínimo de área para considerar un objeto
MinArea = 500  # Número mínimo de píxeles

# Inicializar una tabla para almacenar los resultados de todas las imágenes
resultados_totales = pd.DataFrame()

# Crear carpetas para guardar las imágenes modificadas y tablas de características
modified_folder = os.path.join(folder_path, 'Imagenes_Modificadas_Python')
features_folder = os.path.join(folder_path, 'Caracteristicas_Python')
os.makedirs(modified_folder, exist_ok=True)
os.makedirs(features_folder, exist_ok=True)

# Iterar sobre cada imagen en la carpeta
for i, filepath in enumerate(image_files, 1):
    filename = os.path.basename(filepath)
    print(f'Procesando imagen {i} de {len(image_files)}: {filename}')
    
    # Leer la imagen
    img = io.imread(filepath)
    
    # Verificar si la imagen es RGB
    if img.ndim != 3 or img.shape[2] != 3:
        print(f'Advertencia: La imagen {filename} no es RGB. Se omitirá.')
        continue  # Saltar a la siguiente imagen
    
    # Identificar los píxeles que son puramente blancos (R=255, G=255, B=255)
    mask_white = np.all(img == 255, axis=2)
    
    # Crear una copia de la imagen para modificar
    img_modified = img.copy()
    
    # Convertir los píxeles blancos a negros
    img_modified[mask_white] = [0, 0, 0]
    
    # Crear una máscara de objetos: píxeles que no son completamente negros
    mask_objects = np.any(img_modified != 0, axis=2)
    
    # Convertir la máscara a una imagen binaria
    binary_image = mask_objects
    
    # Rellenar huecos dentro de los objetos para una mejor detección
    binary_filled = morphology.remove_small_holes(binary_image, area_threshold=MinArea)
    
    # Aplicar el filtrado para eliminar objetos pequeños
    binary_filtered = remove_small_objects(binary_filled, min_size=MinArea)
    
    # Etiquetar los componentes conectados
    labeled_image = measure.label(binary_filtered, connectivity=2)
    properties = measure.regionprops(labeled_image)
    
    # Contar el número de objetos
    num_objects = len(properties)
    
    print(f'Número de objetos detectados (≥ {MinArea} píxeles): {num_objects}')
    
    # Si no se detectaron objetos, continuar con la siguiente imagen
    if num_objects == 0:
        print(f'Advertencia: No se detectaron objetos en la imagen {filename}.')
        continue
    
    # Inicializar arreglos para almacenar las características
    objeto_num = np.arange(1, num_objects + 1)
    cantidad_pixeles = np.zeros(num_objects)
    promedio_r = np.zeros(num_objects)
    promedio_g = np.zeros(num_objects)
    promedio_b = np.zeros(num_objects)
    promedio_rgb = np.zeros(num_objects)
    
    # Extraer los canales R, G, B de la imagen modificada
    R_channel = img_modified[:, :, 0].astype(float)
    G_channel = img_modified[:, :, 1].astype(float)
    B_channel = img_modified[:, :, 2].astype(float)
    
    # Iterar sobre cada objeto para calcular las características
    for j, prop in enumerate(properties):
        # Obtener los índices de píxeles del objeto actual
        coords = prop.coords
        cantidad_pixeles[j] = len(coords)
        
        # Promedio de cada canal de color
        promedio_r[j] = R_channel[coords[:, 0], coords[:, 1]].mean()
        promedio_g[j] = G_channel[coords[:, 0], coords[:, 1]].mean()
        promedio_b[j] = B_channel[coords[:, 0], coords[:, 1]].mean()
        
        # Promedio RGB combinado
        promedio_rgb[j] = (promedio_r[j] + promedio_g[j] + promedio_b[j]) / 3
    
    # Crear una DataFrame con las características de la imagen actual
    caracteristicas_imagen = pd.DataFrame({
        'Objeto': objeto_num,
        'Cantidad_Pixeles': cantidad_pixeles,
        'Promedio_R': promedio_r,
        'Promedio_G': promedio_g,
        'Promedio_B': promedio_b,
        'Promedio_RGB': promedio_rgb,
        'Nombre_Imagen': filename
    })
    
    # Añadir las características de la imagen actual a la tabla total
    resultados_totales = pd.concat([resultados_totales, caracteristicas_imagen], ignore_index=True)
    
    # Guardar la tabla de características como un archivo .csv
    csv_filename = os.path.join(features_folder, f'{filename}_Caracteristicas.csv')
    caracteristicas_imagen.to_csv(csv_filename, index=False)
    
    # Guardar la imagen modificada en la carpeta de imágenes modificadas
    modified_filename = os.path.join(modified_folder, f'Modificado_{filename}')
    io.imsave(modified_filename, img_modified)
    
    print(f'Imagen procesada y guardada: {modified_filename}')
    print('-' * 50)

# Visualización de Resultados Totales
print('Resultados Consolidados de Todas las Imágenes:')
print(resultados_totales)

# Guardar la tabla consolidada como un archivo .csv
consolidated_csv = os.path.join(features_folder, 'Resultados_Totales.csv')
resultados_totales.to_csv(consolidated_csv, index=False)
print(f'Resultados consolidados guardados en: {consolidated_csv}')

# --- Inicio de la Nueva Funcionalidad: Clasificador k-means ---

# Paso 1: Preparar los Datos para k-means
# Seleccionar las características para la clasificación
features = resultados_totales[['Promedio_R', 'Promedio_G', 'Promedio_B', 'Promedio_RGB']].values

# Normalizar las características para que todas tengan igual peso
features_normalized = normalize(features)

# Paso 2: Aplicar k-means para Clasificar los Objetos
k = 5  # Número de clases
random_state = 1  # Para reproducibilidad
kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=5)
idx = kmeans.fit_predict(features_normalized)
C = kmeans.cluster_centers_

# Añadir los índices de cluster a la tabla
resultados_totales['Cluster'] = idx

# Paso 3: Identificar Imágenes Representativas
# Buscamos la primera imagen que contenga únicamente objetos de cada clase

etiquetas_clases = [None] * k

for class_num in range(k):
    # Obtener los nombres únicos de las imágenes
    images = resultados_totales['Nombre_Imagen'].unique()
    
    imagen_encontrada = False
    for filename in images:
        datos_imagen = resultados_totales[resultados_totales['Nombre_Imagen'] == filename]
        
        # Verificar si todos los objetos en esta imagen pertenecen a 'class_num'
        if (datos_imagen['Cluster'] == class_num).all():
            # Si es así, esta es la imagen representativa para la clase 'class_num'
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            
            # Mostrar la imagen
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.title(f'Clase {class_num +1} - {filename}')
            plt.axis('off')
            plt.show(block=False)
            
            # Pedir al usuario que ingrese la etiqueta para esta clase
            etiqueta = input(f'¿Qué etiqueta recibe la Clase {class_num +1}? ')
            if not etiqueta:
                etiqueta = f'Clase_{class_num +1}'
            etiquetas_clases[class_num] = etiqueta
            
            # Cerrar la figura de la imagen
            plt.close()
            
            imagen_encontrada = True
            break  # Salir del bucle de imágenes para pasar a la siguiente clase
    
    # Si no se encontró una imagen que contenga solo objetos de 'class_num'
    if not imagen_encontrada:
        print(f'Advertencia: No se encontró una imagen con únicamente objetos de la Clase {class_num +1}.')
        etiqueta = input(f'Ingrese una etiqueta para la Clase {class_num +1} (Asignación por defecto: Clase_{class_num +1}): ')
        if not etiqueta:
            etiqueta = f'Clase_{class_num +1}'
            print(f'Se asignó la etiqueta por defecto: {etiqueta}')
        etiquetas_clases[class_num] = etiqueta

# Paso 4: Asignar Etiquetas a Todos los Objetos y Actualizar la Tabla

# Crear una columna de etiquetas basada en los clusters
resultados_totales['Etiqueta'] = resultados_totales['Cluster'].apply(lambda x: etiquetas_clases[x])

# Guardar la tabla actualizada
updated_csv = os.path.join(features_folder, 'Resultados_Totales_Con_Etiquetas.csv')
resultados_totales.to_csv(updated_csv, index=False)
print(f'Las etiquetas han sido asignadas y la tabla ha sido actualizada: {updated_csv}')

# Opcional: Guardar la tabla consolidada visual con etiquetas
# Para simplificar, se guardará como una imagen usando pandas y matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')
table_data = resultados_totales.values
column_labels = resultados_totales.columns
table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.savefig(os.path.join(features_folder, 'Resultados_Totales_Con_Etiquetas.png'), bbox_inches='tight')
plt.close()
print('Tabla consolidada con etiquetas guardada como imagen.')

print(f'Proceso de clasificación y etiquetado completado. Los resultados actualizados se han guardado en:\n{features_folder}')

# --- Inicio de la Nueva Funcionalidad: Memorama ---

# Preparar los datos necesarios para el memorama
# Seleccionar las 25 imágenes aleatorias y crear el arreglo de 50 elementos

# Obtener una lista única de imágenes
unique_images = resultados_totales['Nombre_Imagen'].unique()

# Verificar que hay al menos 25 imágenes para seleccionar
if len(unique_images) < 25:
    raise ValueError('No hay suficientes imágenes únicas para crear el memorama. Se requieren al menos 25 imágenes únicas.')

# Seleccionar 25 imágenes al azar
selected_images = random.sample(list(unique_images), 25)

# Crear el arreglo de 50 elementos con pares
memorama_array = selected_images * 2
random.shuffle(memorama_array)

# Inicializar variables para el juego
matched_indices = [False] * 50  # Indicadores de pares encontrados
player_scores = [0, 0]  # [Jugador1, Jugador2]
current_player = 0  # 0 para Jugador 1, 1 para Jugador 2

print('\n--- Bienvenido al Memorama ---')

# Inicializar Pygame
pygame.init()
screen_width = 900
screen_height = 900
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Memorama')
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# Función para mostrar las imágenes seleccionadas (adaptada para Pygame)
def show_selected_images(img1, img2, duration=3):
    img1_path = os.path.join(folder_path, img1)
    img2_path = os.path.join(folder_path, img2)
    
    if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
        print('Una o ambas imágenes seleccionadas no existen en la ruta especificada.')
        return
    
    img1_display = pygame.image.load(img1_path)
    img2_display = pygame.image.load(img2_path)
    
    img1_display = pygame.transform.scale(img1_display, (400, 400))
    img2_display = pygame.transform.scale(img2_display, (400, 400))
    
    screen.fill((30, 30, 30))
    screen.blit(img1_display, (50, 200))
    screen.blit(img2_display, (450, 200))
    pygame.display.flip()
    
    pygame.time.delay(duration * 1000)  # Pausar por 'duration' segundos

# Función para obtener los conteos de clases de una imagen
def get_image_class_counts_pygame(resultados_totales, image_name):
    subset = resultados_totales[resultados_totales['Nombre_Imagen'] == image_name]
    clases = subset['Etiqueta'].unique()
    counts = {}
    for clase in clases:
        counts[clase] = (subset['Etiqueta'] == clase).sum()
    return counts

# Iniciar el juego
running = True
while running:
    display_grid(screen, memorama_array, matched_indices, font)
    
    # Mostrar puntajes actuales
    score_text = f'Puntajes:\nJugador 1: {player_scores[0]}\nJugador 2: {player_scores[1]}'
    lines = score_text.split('\n')
    for i, line in enumerate(lines):
        label = font.render(line, True, (255, 255, 255))
        screen.blit(label, (750, 50 + i * 30))
    
    pygame.display.flip()
    
    # Check for quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
    
    # Solicitar selección de dos números al jugador actual
    print(f'\nTurno del Jugador {current_player +1}')
    idx1, idx2 = get_player_selection(memorama_array, matched_indices, font, screen)
    
    img1 = memorama_array[idx1]
    img2 = memorama_array[idx2]
    
    # Mostrar las imágenes seleccionadas
    show_selected_images(img1, img2)
    
    # Obtener los conteos de clases
    counts1 = get_image_class_counts_pygame(resultados_totales, img1)
    counts2 = get_image_class_counts_pygame(resultados_totales, img2)
    
    print(f'\nImagen {idx1 +1}: {img1}')
    print(counts1)
    print(f'Imagen {idx2 +1}: {img2}')
    print(counts2)
    
    # Verificar si las dos imágenes son iguales en términos de objetos por clase
    if counts1 == counts2:
        print('¡Pareja encontrada!')
        matched_indices[idx1] = True
        matched_indices[idx2] = True
        player_scores[current_player] += 1
    else:
        print('No es una pareja. Cambia el turno.')
        current_player = 1 - current_player  # Cambiar de jugador
    
    # Verificar si el juego ha terminado
    if all(matched_indices):
        print('\n--- Juego Finalizado ---')
        print(f'Puntajes Finales:\nJugador 1: {player_scores[0]}\nJugador 2: {player_scores[1]}')
        if player_scores[0] > player_scores[1]:
            print('¡Jugador 1 gana!')
        elif player_scores[1] > player_scores[0]:
            print('¡Jugador 2 gana!')
        else:
            print('¡Es un empate!')
        running = False

# Cerrar Pygame
pygame.quit()
