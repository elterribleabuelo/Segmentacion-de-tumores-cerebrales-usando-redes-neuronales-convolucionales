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
    BatchNormalization,
    Dropout
)
from tensorflow.keras.layers import concatenate,Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

### BLIQUE DE CONVOLUCION ####
### CONV->BATCHNORM->ACTIVATION->CONV->BATCHNORM->ACTIVATION->ADD ####

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

def conv_block(input_mat,num_filters,kernel_size,batch_norm):
  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_mat)
  
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)

  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
  
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
    
  X = Add()([input_mat,X])
  
  return X

#### MODELO V-NET ###

def vnet_model_3d(loss_function, input_shape=(4, 160, 160, 16),
                  pool_size=(2, 2, 2), n_labels=3,
                  initial_learning_rate=0.00001,
                  deconvolution=False,metrics=[],
                  activation_name="sigmoid",dropout = 0.2,
                  batch_norm = True):
    
    ###### CAPA DE ENTRADA ######
    inputs = Input(input_shape)
    
    
    ###### ENCODER ######
    c1 = Conv3D(16,kernel_size = (5,5,5) , strides = (1,1,1) , padding='same', kernel_initializer = 'he_normal')(inputs)
    #8
    c2 = Conv3D(32,kernel_size = (2,2,2) , strides = (2,2,2) , padding = 'same',kernel_initializer = 'he_normal' )(c1)
    #16
    c3 = conv_block(c2 , 32,5,True)
    #16
    p3 = Conv3D(64,kernel_size = (2,2,2) , strides = (2,2,2), padding = 'same',kernel_initializer = 'he_normal')(c3)
    #32
    p3 = Dropout(dropout)(p3)
    c4 = conv_block(p3, 64,5,True)
    #32
    p4 = Conv3D(128,kernel_size = (2,2,2) , strides = (2,2,2) , padding='same', kernel_initializer = 'he_normal')(c4)
    #64
    p4 = Dropout(dropout)(p4)
    c5 = conv_block(p4, 128,5,True)
    #64
    p6 = Conv3D(256,kernel_size = (2,2,2) , strides = (2,2,2) , padding='same', kernel_initializer = 'he_normal')(c5)
    #128
    p6 = Dropout(dropout)(p6)
    
    
    ###### CAPA INTERMEDIA ######
    p7 = conv_block(p6,256,5,True)
    #128
    
    
    ###### DECODER ######
    u6 = Conv3DTranspose(128, (2,2,2), strides=(2, 2, 2), padding='same')(p7)
    #64
    merge6 = concatenate([u6,c5],axis = 1)
    c7 = conv_block(merge6,256,5,True)
    #128
    c7 = Dropout(dropout)(c7)
    
    u7 = Conv3DTranspose(64,(2,2,2),strides = (2,2,2) , padding= 'same')(c7)
    #32
    merge8 = concatenate([u7,c4],axis = 1)
    c8 = conv_block(merge8,128,5,True)
    #64
    c8 = Dropout(dropout)(c8)
    
    u9 = Conv3DTranspose(32,(2,2,2),strides = (2,2,2) , padding= 'same')(c8)
    #16
    merge9 = concatenate([u9,c3],axis = 1)
    c9 = conv_block(merge9,64,5,True)
    #32
    c9 = Dropout(dropout)(c9)
    
    u10 = Conv3DTranspose(16,(2,2,2),strides = (2,2,2) , padding= 'same')(c9)
    #8
    merge10 = concatenate([u10,c1],axis = 1)
    
    c10 = Conv3D(32,kernel_size = (5,5,5),strides = (1,1,1) , padding = 'same')(merge10)
    #16
    c10 = Dropout(dropout)(c10)
    c10 = Add()([c10,merge10])
    
    ###### CAPA DE CLASIFICACIÓN ######
    outputs = Conv3D(n_labels, (1,1,1), activation = activation_name)(c10)
    
    ###### CREAMOS EL MODELO ######
    model = Model(inputs, outputs=outputs)
    
    ###### COMPILAMOS EL MODELO ######
    model.compile(optimizer = Adam(lr = initial_learning_rate), loss = loss_function, metrics = metrics)
    
    return model