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
    
    raise NotImplementedError
    

def data_augmentation():
    '''
    In this function you are going to build data augmentation layers using tf.keras
    First, add random horizontal and vertical flip
    Second, add random rotation
    return tf.keras.Sequential object containing the above mentioned augmentation layers
    '''
    
    raise NotImplementedError


    

