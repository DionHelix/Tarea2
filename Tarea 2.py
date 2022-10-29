import numpy as np
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from scipy.fftpack import fft
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow import keras
from keras import models, Sequential,layers,optimizers,metrics,losses
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D, Conv2D, Activation ,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
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

labeled_images = data.map(process_file).batch(10)

print('labeled images:  ', labeled_images)

test_data = labeled_images.take(50)
train_data = labeled_images.skip(50)
print('test data: ', test_data)
print('train data: ', train_data)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, 192, 192, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy,
              metrics=['accuracy'])
history = model.fit(train_data, epochs=10, 
                    validation_data=test_data)