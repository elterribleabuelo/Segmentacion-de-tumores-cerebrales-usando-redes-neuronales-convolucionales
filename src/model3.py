import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.layers import (Conv2D, 
                                     BatchNormalization, 
                                     Activation, 
                                     MaxPool2D, 
                                     UpSampling2D,
                                     Input,
                                     Conv3D,
                                     UpSampling3D)

from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), 
                   epsilon=0.00001):


    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))
    
    return dice_loss


def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), 
                     epsilon=0.00001):
    

    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean((dice_numerator)/(dice_denominator))
    
    return dice_coefficient
    
    
def batchnorm_relu(inputs):
     x = BatchNormalization()(inputs)
     x = Activation("relu")(x)
     return x

def residual_block(inputs, num_filters, strides=1):
     x = batchnorm_relu(inputs)
     x = Conv3D(num_filters, 3, padding="same", strides=strides)(x)
     x = batchnorm_relu(x)
     x = Conv3D(num_filters, 3, padding="same", strides=1)(x)
     s = Conv3D(num_filters, 1, padding="same", strides=strides)(inputs)
     x = x + s
     return x

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling3D((2, 2, 2))(inputs) 
    x = concatenate([x, skip_features],axis = 1) 
    x = residual_block(x, num_filters, strides=1) 
    return x

def resunet_model_3d(loss_function, input_shape=(4, 160, 160, 16), n_labels=3,
                     initial_learning_rate=0.00001,
                     metrics=[],
                     activation_name="sigmoid"):
    
    #################### PASO 1: CAPA DE ENTRADA #############################
    
    inputs = Input(input_shape)
    
    #################### PASO 2: INICIO ENCODER #############################

    ### BLOQUE 1 #### 
    x = Conv3D(64, 3, padding="same", strides=1)(inputs) 
    x = batchnorm_relu(x) 
    x = Conv3D(64, 3, padding="same", strides=1)(x) 
    s = Conv3D(64, 1, padding="same")(inputs) 
    s1 = x + s

    ### BLOQUE 2 #### 
    s2 = residual_block(s1, 128, strides=2)
    
    ### BLOQUE 3 #### 
    s3 = residual_block(s2, 256, strides=2)

    #################### PASO 3: CAPA INTERMEDIA #############################
    
    b = residual_block(s3, 512, strides=2)
    
    #################### PASO 4: INICIO DECODER #########################
    
    ### BLOQUE 1 #### 
    x = decoder_block(b, s3, 256)
    
    ### BLOQUE 2 #### 
    x = decoder_block(x, s2, 128) 
    
    ### BLOQUE 3 #### 
    x = decoder_block(x, s1, 64)

    ################### PASO 5: CAPA DE CLASIFICACION ################
    
    outputs = Conv3D(n_labels, 1, padding="same", activation = activation_name)(x)

    ################### PASO 6: CREAMOS EL OBJETO MODELO ################
    
    model = Model(inputs, outputs)
    
    ################### PASO 7: COMPILAMOS EL MODELO ################

    model.compile(optimizer = Adam(lr = initial_learning_rate), loss = loss_function, metrics = metrics)

    return model