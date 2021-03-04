import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,
)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

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


def unet_model_3d(loss_function, input_shape=(4, 160, 160, 16),
                  pool_size=(2, 2, 2), n_labels=3,
                  initial_learning_rate=0.00001,
                  deconvolution=False,metrics=[],
                  activation_name="sigmoid"):
    
    #################### PASO 1: CAPA DE ENTRADA #############################
     
    inputs = Input(input_shape)
    
    #################### PASO 2: INICIO ENCODER #############################
    
    #### Bloque 1 ####
    
    conv1 = Conv3D(32, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    #### Bloque 2 ####
    
    conv2 = Conv3D(64,  (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    #### Bloque 3 ####
     
    conv3 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
     #### Bloque 4 ####
    #conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    #################### FIN ENCODER #################################
    
    #################### PASO 3: INICIO CAPA INTERMEDIA ######################
    
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop5 = Dropout(0.5)(conv5)
    
    #################### FIN CAPA INTERMEDIA #########################
    
    #################### PASO 4: INICIO DECODER #########################
    
    #### Bloque 1 ####
    
    up6 = UpSampling3D(size = (2, 2, 2))(conv4)
    merge6 = concatenate([conv3,up6], axis = 1)
    conv6 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    #### Bloque 2 ####
    
    up7 = UpSampling3D(size = (2, 2, 2))(conv6)
    merge7 = concatenate([conv2,up7], axis = 1)
    conv7 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    #### Bloque 3 ####
    
    up8 = UpSampling3D(size = (2, 2, 2))(conv7)
    merge8 = concatenate([conv1,up8], axis = 1)
    conv8 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    #################### FIN DECODER #########################
     
     
    #### Bloque 4 ####
    #up9 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #merge9 = concatenate([conv1,up9], axis = 3)
    #conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    ################### PASO 5: CAPA DE CLASIFICACION ################
    
    conv9 = Conv3D(n_labels, (1, 1, 1), activation = activation_name, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #conv10 = Conv3D(1, 1, activation = 'sigmoid')(conv9)
    
    ################### PASO 6: CREAMOS EL OBJETO MODELO ################
    
    model = Model(inputs, conv9)
    
    ################### PASO 7: COMPILAMOS EL MODELO ################

    model.compile(optimizer = Adam(lr = initial_learning_rate), loss = loss_function, metrics = metrics)
    
    return model
