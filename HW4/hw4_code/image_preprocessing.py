from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers


def data_preprocessing(IMG_SIZE=32):
    '''
    In this function you are going to build data preprocessing layers using tf.keras
    First, resize your image to consistent shape
    Second, standardize pixel values to [0,1]
    return tf.keras.Sequential object containing the above mentioned preprocessing layers
    '''
    # HINT :You can resize your images with tf.keras.layers.Resizing,
    # You can rescale pixel values with tf.keras.layers.Rescaling
    resize_and_rescale = tf.keras.Sequential([layers.Resizing(IMG_SIZE, IMG_SIZE),layers.Rescaling(1./255)])
    return resize_and_rescale
    # raise NotImplementedError
    

def data_augmentation():
    '''
    In this function you are going to build data augmentation layers using tf.keras
    First, add random horizontal and vertical flip
    Second, add random rotation
    return tf.keras.Sequential object containing the above mentioned augmentation layers
    '''
    data_aug = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),layers.RandomRotation(0.2)])
    return data_aug
    # raise NotImplementedError


    

