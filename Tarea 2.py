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
from keras import Sequential,layers,optimizers,metrics,losses
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D, Conv2D, Activation ,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils

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

labeled_images = data.map(process_file)

print('labeled images:  ', labeled_images)

#for image, attri in labeled_images.take(2):
#    plt.imshow(image)
#    plt.show()
#    print(image.shape)

##### Separación de la Data ######

test_data = labeled_images.take(50)
train_data = labeled_images.skip(50)
print('test data: ', test_data)
print('train data: ', train_data)

#### Modelo ####

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=(192,192,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']) 
model.summary()
history = model.fit(train_data, batch_size = 30, epochs = 10, verbose=1, validation_data = test_data)