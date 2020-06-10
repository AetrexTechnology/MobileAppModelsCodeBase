#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import os
import os.path
import glob
import numpy as np
import pandas as pd
import cv2
# import seaborn as sns
import random
import matplotlib.pyplot as plt
import json
from tensorflow.keras.layers import Flatten, Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, \
    GlobalAveragePooling2D, Dropout, Add, ReLU, Concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist

# In[3]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# In[4]:


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.8))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

# In[5]:


METRIC_ACCURACY = tf.keras.metrics.RootMeanSquaredError()

# In[10]:


with tf.summary.create_FileWriter('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='RootMeanSquaredError')])

# In[ ]:

input1 = Input(shape=(None, None, 3))
input2 = Input(shape=(18))


# need to change from model to our model architecture
def train_test_model(hparams):
    base_model = VGG16(include_top=False, input_tensor=input1)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    intrins_flat = Flatten()(input2)

    merge = Concatenate()([x, intrins_flat])

    left = Dense(hparams[HP_NUM_UNITS])(merge)
    left = BatchNormalization()(left)
    left = ReLU()(left)
    left - Dropout(hparams[HP_DROPOUT])(left)
    left = Dense(hparams[HP_NUM_UNITS])(left)
    left = BatchNormalization()(left)
    left = ReLU()(left)
    left = Dropout(hparams[HP_DROPOUT])(left)
    left = Dense(hparams[HP_NUM_UNITS])(left)
    left = BatchNormalization()(left)
    left = ReLU()(left)
    left = Dropout(hparams[HP_DROPOUT])(left)
    left_length = Dense(units=1, activation='linear', name='left_length')(left)

    right = Dense(hparams[HP_NUM_UNITS])(merge)
    right = BatchNormalization()(right)
    right = ReLU()(right)
    right = Dropout(hparams[HP_DROPOUT])(right)
    right = Dense(hparams[HP_NUM_UNITS])(right)
    right = BatchNormalization()(right)
    right = ReLU()(right)
    right = Dropout(hparams[HP_DROPOUT])(right)
    right = Dense(hparams[HP_NUM_UNITS])(right)
    right = BatchNormalization()(right)
    right = ReLU()(right)
    right = Dropout(hparams[HP_DROPOUT])(right)

    right_length = Dense(units=1, activation='linear', name='right_length')(right)

    model = Model(inputs=[input1, input2], outputs=[left_length, right_length])
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss={'left_length': 'mse', 'right_length': 'mse'},
        metrics={'left_length': tf.keras.metrics.RootMeanSquaredError(),
                       'right_length': tf.keras.metrics.RootMeanSquaredError()},
    )

    model.fit(x_train, y_train, epochs=1)  # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy


# In[ ]:


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


# In[ ]:


session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1
