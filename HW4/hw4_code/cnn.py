from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN(object):
    def __init__(self):
        # change these to appropriate values

        self.batch_size = 60
        self.epochs = 20
        self.init_lr= 1e-3 #learning rate

        # No need to modify these
        self.model = self.create_net()

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''
        model = Sequential()
        #add model layers
        model.add(Conv2D(8, kernel_size=3, padding="same", 
                        kernel_regularizer=regularizers.l2(1e-4), 
                        activation='selu', input_shape=(32,32,3)))
        model.add(LeakyReLU(alpha = 0.3))
        model.add(Conv2D(32, kernel_size=3, padding="same", 
                        kernel_regularizer=regularizers.l2(1e-4), 
                        activation='selu'))
        model.add(LeakyReLU(alpha = 0.3))
        model.add(MaxPooling2D(pool_size=2, padding='valid'))
        model.add(Dropout(0.3))
        model.add(Conv2D(32, kernel_size=3, padding="same", 
                        kernel_regularizer=regularizers.l2(1e-4), 
                        activation='selu'))
        model.add(LeakyReLU(alpha = 0.3))
        model.add(Conv2D(64, kernel_size=3, padding="same", 
                        kernel_regularizer=regularizers.l2(1e-4), 
                        activation='selu'))
        model.add(LeakyReLU(alpha = 0.3))
        model.add(MaxPooling2D(pool_size=2, padding='valid'))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(256, activation='selu'))
        model.add(LeakyReLU(alpha = 0.3))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='selu'))
        model.add(LeakyReLU(alpha = 0.3))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='selu'))
        model.add(Activation('selu'))
        return model
        # raise NotImplementedError

    
    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model.
        '''
        self.model = model
        model.compile(tf.keras.optimizers.Adam(learning_rate=self.init_lr), 
                    loss = tf.keras.losses.MeanSquaredError(), 
                    metrics = [tf.keras.metrics.CategoricalAccuracy()])
        return model
        # raise NotImplementedError
