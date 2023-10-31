import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

dataset = 'C:/Users/zoama/Downloads/BRAIN_TUMOR_PREDICTION' #cargamos el dataset

# Asignamos las categorias que se encuentran dentro de las dos carpetas: training y testing
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Almacenamos las imágenes y las etiquetas correspondientes (números asignados a cada categoría) de los datos del training.
images = []  # Guarda todas las imágenes
labels = []  # Guarda los números asignados a cada categoría

# Cargar imágenes de training
for category in categories:
    path = os.path.join(dataset, 'training', category)  # Ruta a la carpeta de training
    class_num = categories.index(category)  # Asignar un número a cada categoría

    # OpenCV (cv2) para cargar la imagen en escala de grises, cambiar su tamaño a 100x100 píxeles y agregarla a la lista
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Leer la imagen
        new_height, new_width = 100, 100
        resized_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        images.append(resized_img)
        labels.append(class_num)


images = np.array(images)  # training
labels = np.array(labels)  # training

# dividimos los datos en conjuntos de training y validación - CROSS-TRAIN VALIDATION
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar los valores de píxeles al rango [0, 1] con el fin de que los valores de píxeles estén en una escala adecuada
X_train = X_train / 255.0
X_val = X_val / 255.0

# Se dividen todos los valores de píxeles por 255.0, lo que garantiza que los valores estén en el rango [0, 1].
# La razón detrás de esta normalización es que las imágenes generalmente se representan en formato de 8 bits.
# Donde 0 representa el color negro y 255 representa el color blanco.
# Al dividir por 255, se escala este rango a [0, 1], lo que hace que el entrenamiento sea más efectivo.



# Arquitectura del modelo elegída: CNN
# Consta de 5 capas: Conv2D / MaxPooling2D / Flatten / Dense / Dense de salida
# Arquitectura básica de CNN que puede servir como punto de partida

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 clases: glioma, meningioma, notumor, pituitary
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Optimizador : optimizer - 'adam' es un optimizador popular que funciona bien en la mayoría de los casos.
# loss: evalúa cuán bien se está desempeñando el modelo.
# metrics:  evaluar el rendimiento del modelo durante y después del entrenamiento. En este caso, estás utilizando la métrica 'accuracy'

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

#epochs: la cantidad de veces que el modelo verá todo el conjunto de training durante el training.
# el modelo realizará 10 pasadas completas a través del conjunto de entrenamiento.

# Podriamos evaluar el modelo en el testing antes del evaluarlo en training

# 143 indica que durante cada época, se procesan 143 lotes de datos de entrenamiento.
# Esto es parte del proceso de entrenamiento por lotes, que es común en el aprendizaje profundo para acelerar el entrenamiento
# y hacer un uso eficiente de los recursos computacionales.

# El modelo se guarda en el formato de Keras
model.save('modelo_tumor_cerebral.keras')

# Evaluar el modelo en el conjunto training
# train_loss, train_accuracy = model.evaluate(X_train, y_train)
# print(f'Pérdida en el conjunto de entrenamiento: {train_loss}')
# print(f'Precisión en el conjunto de entrenamiento: {train_accuracy}')





