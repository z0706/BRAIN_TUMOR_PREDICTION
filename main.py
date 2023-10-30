import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

data_path_training = 'C:/Users/rosar/Downloads/TUMORS/training'  # Ruta a la carpeta 'training'
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
images = []  # Guarda todas las imágenes
labels = []  # Guarda los números asignados a cada categoría

for category in categories:
    path = os.path.join(data_path_training, category)
    class_num = categories.index(category)  # Asignar un número a cada categoría

    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Leer la imagen
        new_height, new_width = 100, 100
        resized_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        images.append(resized_img)  # Utilizar resized_img en lugar de img_array
        labels.append(class_num)  # Agregar el número de categoría como etiqueta

data_path_testing = 'C:/Users/rosar/Downloads/TUMORS/Testing'  # Ruta a la carpeta 'training'
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
images = []  # Guarda todas las imágenes
labels = []  # Guarda los números asignados a cada categoría

for category in categories:
    path = os.path.join(data_path_testing, category)
    class_num = categories.index(category)  # Asignar un número a cada categoría
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Leer la imagen
        new_height, new_width = 100, 100
        resized_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        images.append(resized_img)  # Utilizar resized_img en lugar de img_array
        labels.append(class_num)  # Agregar el número de categoría como etiqueta

images = np.array(images)
labels = np.array(labels)




