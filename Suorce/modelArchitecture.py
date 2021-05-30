# -*- coding: utf-8 -*-
"""
Created on Fri May 28 21:28:29 2021

@author: TUSHAR JAIN

"""

import tensorflow as tf
import config
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, RepeatVector,Reshape,Dense, Flatten, Input, Concatenate


def build_model(ks=(3,3),act='sigmoid',learning_rate=1e-2):
    
  # Input Layer
  input_lvl = Input(shape = (config.HEIGHT,config.WIDTH,1))
  
  # Initial Shared Network of Low - Level Features
  low_lvl = Conv2D(64 ,kernel_size=ks,strides=(2,2),activation=act,padding='SAME')(input_lvl)
  low_lvl = layers.BatchNormalization()(low_lvl)
  low_lvl = Conv2D(128,kernel_size=ks,strides=(1,1),activation=act,padding='SAME')(low_lvl) 
  low_lvl = layers.BatchNormalization()(low_lvl)
  low_lvl = Conv2D(128,kernel_size=ks,strides=(2,2),activation=act,padding='SAME')(low_lvl) 
  low_lvl = layers.BatchNormalization()(low_lvl)
  low_lvl = Conv2D(256,kernel_size=ks,strides=(1,1),activation=act,padding='SAME')(low_lvl) 
  low_lvl = layers.BatchNormalization()(low_lvl)
  low_lvl = Conv2D(256,kernel_size=ks,strides=(2,2),activation=act,padding='SAME')(low_lvl)
  low_lvl = layers.BatchNormalization()(low_lvl)
  low_lvl = Conv2D(512,kernel_size=ks,strides=(1,1),activation=act,padding='SAME')(low_lvl)
  low_lvl = layers.BatchNormalization()(low_lvl)

  # Path one for  Mid-Level Features Network
  mid_lvl = Conv2D(512,kernel_size=ks,strides=(1,1),activation=act,padding='SAME')(low_lvl)
  mid_lvl = layers.BatchNormalization()(mid_lvl)
  mid_lvl = Conv2D(256,kernel_size=ks,strides=(1,1),activation=act,padding='SAME')(mid_lvl)
  mid_lvl = layers.BatchNormalization()(mid_lvl)

  # Path two for Global Features Network
  global_lvl = Conv2D(512,kernel_size=ks,strides=(2,2),activation=act,padding='SAME')(low_lvl)
  global_lvl = layers.BatchNormalization()(global_lvl)
  global_lvl = Conv2D(512,kernel_size=ks,strides=(1,1),activation=act,padding='SAME')(global_lvl)
  global_lvl = layers.BatchNormalization()(global_lvl)
  global_lvl = Conv2D(512,kernel_size=ks,strides=(2,2),activation=act,padding='SAME')(global_lvl)
  global_lvl = layers.BatchNormalization()(global_lvl)
  global_lvl = Conv2D(512,kernel_size=ks,strides=(1,1),activation=act,padding='SAME')(global_lvl)
  global_lvl = layers.BatchNormalization()(global_lvl)
  global_lvl = Flatten()(global_lvl) 
  global_lvl = Dense(1024,activation=act)(global_lvl)
  global_lvl = Dense(512 ,activation=act)(global_lvl)
  global_lvl = Dense(256 ,activation=act)(global_lvl)
  

  # Fusing the output of above two paths
  fusion_lvl = RepeatVector(mid_lvl.shape[1] * mid_lvl.shape[1])(global_lvl) 
  fusion_lvl = Reshape(([mid_lvl.shape[1],mid_lvl.shape[1]  , 256]))(fusion_lvl)
  fusion_lvl = Concatenate( axis=3)([mid_lvl, fusion_lvl]) 
  fusion_lvl = Conv2D(256, kernel_size=ks,strides =(1, 1), activation=act,padding='SAME')(fusion_lvl)

  # Colorization Network
  # Instead of UpSampling Layers I am using 2D Convolutional Transpose or deconv for upscaling the images 
  color_lvl = Conv2DTranspose(128,kernel_size = ks,strides = (1,1),padding='SAME',activation=act)(fusion_lvl)
  color_lvl = layers.BatchNormalization()(color_lvl)
  color_lvl = Conv2DTranspose(64,kernel_size = ks,strides = (2,2),padding='SAME',activation=act)(color_lvl)
  color_lvl = layers.BatchNormalization()(color_lvl)
  color_lvl = Conv2DTranspose(64,kernel_size = ks,strides = (1,1),padding='SAME',activation=act)(color_lvl)
  color_lvl = layers.BatchNormalization()(color_lvl)
  color_lvl = Conv2DTranspose(32,kernel_size = ks,strides = (2,2),padding='SAME',activation=act)(color_lvl)
  color_lvl = layers.BatchNormalization()(color_lvl)
  
  # I added the below mentioned two lines when I trained the model for 100 X 100 sized images
  # Ignore if you are using 256 X 256
  # color_lvl = Conv2D(32,kernel_size = ks,strides = (1,1),padding='VALID',activation=act)(color_lvl)
  # color_lvl = layers.BatchNormalization()(color_lvl)

  # Output Layer
  output_lvl = Conv2DTranspose(2,kernel_size=ks,strides=(2,2),padding='SAME',activation='sigmoid')(color_lvl)


  # Model Parameters
  model = Model(inputs = input_lvl, outputs = output_lvl)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(
      loss = config.LOSS_FUNCTION,
      optimizer = optimizer,
      metrics = ['accuracy',
          tf.keras.metrics.CosineSimilarity()
          ])
  return model