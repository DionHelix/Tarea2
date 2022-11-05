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

# with open('list_attr_celeba.txt', 'r') as f:
#     print("skipping : " + f.readline())
#     print("skipping headers : " + f.readline())
#     with open('attr_celeba_prepared.txt', 'w') as newf:
#         for line in f:
#             new_line = ' '.join(line.split())
#             newf.write(new_line)
#             newf.write('\n')

##### Leer la Data #####

df =  pd.read_csv('attr_celeba_prepared.txt', sep=' ', header=None)
df =  df.replace(-1, 0)
files = tf.data.Dataset.from_tensor_slices(df[0])
atributos = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, atributos))
print('data: ', data)

pic_dir = ('E:/Conocimiento/Física/Otro/Códigos muy pesados para Drive (Si no llora)/CelebA/archive/img_align_celeba/')

def process_file(file_name, atributos):
    image = tf.io.read_file(pic_dir + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, atributos

labeled_images = data.map(process_file).batch(128)

print('labeled images:  ', labeled_images)

train_data = labeled_images.skip(800)
test_data = labeled_images.take(200)
print('test data: ', test_data)
print('train data: ', train_data)

input_shape=(192, 192, 3)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(100))
model.add(layers.Dense(40))
model.add(Activation('sigmoid'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=tf.keras.metrics.BinaryAccuracy())
history = model.fit(train_data,
                    steps_per_epoch=400,
                    epochs=25,
                    verbose=1,
          validation_data=test_data)          

history = history.history
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'], c = 'green')
plt.title('Precisión VS Val-Precisión')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'],c ='red')

plt.title('loss vs val-loss')
plt.show()