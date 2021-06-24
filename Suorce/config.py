# -*- coding: utf-8 -*-
"""
Created on Fri May 28 21:30:45 2021

@author: TUSHAR JAIN

"""

import tensorflow as tf

"""# Input Image Dimension"""

# Change the dimension below if you want to change the input dimension
# if you are changing the dimension than be carefull to tweak the model
# architecture in model.py

HEIGHT = 256
WIDTH  = 256

"""# Model"""

# Don't Change kernel size if you are not sure about what it is
# basically this is the Size of Convolution kernels
# In the original implementation of 'Let There be Colour' kernel size was kept
# as (3,3) so I have left it as it was in original

KERNEL_SIZE = (3,3) 

# I tried using multiple kinds of activation functions but found that sigmoid
# turns out to be a better choice initially I started with tanh activation than
# noticed that output images are becoming more red also since tanh has a range
# of -1 to 1 it is not good for final output layer if your input images are 
# scaled between 0 to 1
  
ACTIVATION_FUNCTION = 'sigmoid'

# Loss Functions that I tried were MSE , MASE, and MAE. I found that MSE is 
# better that or equal to all other in results So I have kept it below although
# MASE will also produce similar results 

LOSS_FUNCTION = tf.keras.losses.MeanSquaredError()

LEARNING_RATE = 1e-3

"""# Paths"""

TRAIN_DIR_PATH   = ""
TEST_DIR_PATH    = ""
VAL_DIR_PATH     = ""
SAVE_MODEL_PATH  = ""
SAVE_OUTPUT_PATH = ""
LOAD_MODEL_PATH  = ""
SAVE_CSV_PATH    = ""
LOAD_CSV_PATH    = ""

"""# other """

NUMBER_OF_TRAINING_EXAMPLES = 500
NUMBER_OF_TEST_EXAMPLES     = 40
NUMBER_OF_VAL_EXAMPLES      = 50
NUMBER_OF_EPOCHS            = 100
STEPS_PER_EPOCHS            = 100
VALIDATION_STEPS            = 20
BATCH_SIZE                  = 50
