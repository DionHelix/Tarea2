import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from scipy.fftpack import fft
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow import keras
from keras import Sequential,layers,optimizers,metrics,losses
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils

df =  pd.read_csv("E://Conocimiento/Física/Otro/Códigos muy pesados para Drive (Si no llora)/CelebA/archive/list_attr_celeba.csv")
files = tf.data.Dataset.from_tensor_slices(df[0:1])
atributos = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, atributos))
pic_dir = ("E://Conocimiento/Física/Otro/Códigos muy pesados para Drive (Si no llora)/CelebA/archive/img_align_celeba")
print(df.shape, df.head())

def process_file(file_name, atributos):
    image = tf.io.read_file(pic_dir + file_name)
    image = tf.image.decode_jpg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, atributos

labeled_images = data.map(process_file)

print(labeled_images)