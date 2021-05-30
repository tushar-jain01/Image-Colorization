# -*- coding: utf-8 -*-
"""
Created on Sat May 29 11:10:13 2021

@author: TUSHAR JAIN
"""

import os
import config
import tools
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import modelArchitecture as net


train_data = tools.loadImagesToArray(
    dir_path   = config.TRAIN_DIR_PATH,
    num_of_img = config.NUMBER_OF_TRAINING_EXAMPLES,
    search_inside = True)

val_data = tools.loadImagesToArray(
    dir_path   = config.VAL_DIR_PATH,
    num_of_img = config.NUMBER_OF_VAL_EXAMPLES,
    search_inside = True)


imgDataGen = tools.DataGenerator()
valimgDataGen = tools.DataGenerator()

"""# Training """
colorize_model = net.build_model(
    ks = config.KERNEL_SIZE,
    act=config.ACTIVATION_FUNCTION,
    learning_rate=config.LEARNING_RATE)

# Load Previously trained model
colorize_model.load_weights(config.LOAD_MODEL_PATH)

log = tf.keras.callbacks.CSVLogger(config.SAVE_CSV_PATH,append=True, separator=',')
callbacks = [log]

history = colorize_model.fit(
    tools.BatchGenerator(train_data, imgDataGen,config.BATCH_SIZE),
    validation_data = tools.BatchGenerator(val_data, valimgDataGen),
    steps_per_epoch =config.STEPS_PER_EPOCHS,
    epoch=config.NUMBER_OF_EPOCHS,
    callbacks=callbacks)


#Plotting and saving the history
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()


colorize_model.save(config.SAVE_MODEL_PATH)

"""# Testing """
test_images = tools.loadImagesToArray(
    dir_path = config.TEST_DIR_PATH,
    num_of_img = config.NUMBER_OF_TEST_EXAMPLES,
    search_inside=True)

gray = tools.RGB2GRAY(test_images,True)
gray2 = tools.RGB2GRAY(test_images)
pred = colorize_model.predict(gray)

for i in range(config.NUMBER_OF_TEST_EXAMPLES):
    output = tools.Lab2RGB(gray[i],pred[i])
    path =  config.SAVE_OUTPUT_PATH+os.sep+"img_"+str(i)
    tools.compare_results(test_images[i],gray2[i],output.reshape(test_images[i].shape),save_results=True,save_as=path)