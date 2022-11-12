import numpy as np
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from scipy.fftpack import fft
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow import keras
from keras import models, Sequential,layers,optimizers,metrics,losses
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from keras import optimizers
from keras.layers import Dropout, Dense, Flatten, Conv2D, Activation, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator

#Cargar Modelo:

modelo_cargado = tf.keras.models.load_model('E:/Conocimiento/Física/Notas/9no Semestre/Redes Neuronales/Tarea 2 - Reconocimiento Facial/Tarea2/FaceRecogPlus.h5')
print('Modelo Cargado: ', modelo_cargado)

#Procesamiento de imagenes

ih, iw = 192, 192 #tamano de la imagen
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales
train_dir = ('E:/Conocimiento/Física/Notas/9no Semestre/Redes Neuronales/Tarea 2 - Reconocimiento Facial/Data/Train/') #directorio de entrenamiento
test_dir = ('E:/Conocimiento/Física/Notas/9no Semestre/Redes Neuronales/Tarea 2 - Reconocimiento Facial/Data/Test/') #directorio de prueba

num_class = 2 #cuantas clases
batch_size = 50 #batch para hacer cada entrenamiento. Lee 50 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria

gentrain = ImageDataGenerator(rescale=1. / 255.) #indica que reescale cada canal con valor entre 0 y 1.

train_data = gentrain.flow_from_directory(train_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')
gentest = ImageDataGenerator(rescale=1. / 255)

test_data = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')

print('train data: ', train_data)
print('Len Train Data: ', len(train_data))
print('test data: ', test_data)
print('Len Test Data: ', len(test_data))

#Modelo

input_shape=(192, 192, 3)

model = models.Sequential()
model.add(modelo_cargado.layers[0])
model.add(modelo_cargado.layers[1])
model.add(modelo_cargado.layers[2])
model.add(modelo_cargado.layers[3])
model.add(modelo_cargado.layers[4])
model.add(modelo_cargado.layers[5])
model.add(modelo_cargado.layers[6])
model.add(modelo_cargado.layers[7])
model.add(modelo_cargado.layers[8])
model.add(modelo_cargado.layers[9])
model.add(modelo_cargado.layers[10])
model.add(modelo_cargado.layers[11])
model.add(modelo_cargado.layers[12])
model.add(modelo_cargado.layers[13])
model.add(modelo_cargado.layers[14])
model.add(modelo_cargado.layers[15])
model.add(modelo_cargado.layers[16])
model.add(modelo_cargado.layers[17])
model.add(layers.Dense(100))
model.add(layers.Dense(1))
model.add(Activation('sigmoid'))

for layer in model.layers[:18]:
    layer.trainable = False
    
model.summary()

# steps_per_epoch = len(train_data)//batch_size
# print('steps_per_epoch: ', steps_per_epoch)
# validation_steps = len(test_data)//batch_size
# print('validation_steps: ', validation_steps)

#Compilación del modelo

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=tf.keras.metrics.BinaryAccuracy(),
              run_eagerly=True)
history = model.fit(train_data,
                    #steps_per_epoch = steps_per_epoch,
                    epochs=50,
                    verbose=1,
                    #validation_steps = validation_steps,
                    validation_data=test_data)          

history = history.history
plt.plot(history['binary_accuracy'])
plt.plot(history['val_binary_accuracy'], c = 'green')
plt.title('Precisión VS Val-Precisión')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'],c ='red')

plt.title('loss vs val-loss')
plt.show()

score = model.evaluate(test_data, verbose=0)
print(score)

pred = model.predict(test_data) 
print(pred) 


def predict_one(model):  
    class_names = ['Bowie', 'Otro']
    image_batch, classes_batch = next(test_data)
    predicted_batch = model.predict(image_batch)
    for k in range(0,image_batch.shape[0]):
      image = image_batch[k]
      pred = predicted_batch[k]
      the_pred = np.argmax(pred)
      predicted = class_names[the_pred]
      val_pred = max(pred)
      the_class = np.argmax(classes_batch[k])
      value = class_names[np.argmax(classes_batch[k])]
      plt.figure(k)
      isTrue = (the_pred == the_class)
      plt.title(str(isTrue) + ' - class: ' + value + ' - ' + 'predicted: ' + predicted + '[' + str(val_pred) + ']')
      plt.imshow(image)

predict_one(model)  
