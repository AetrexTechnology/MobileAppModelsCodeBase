#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path
import glob
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import random
import matplotlib.pyplot as plt
import json
from tensorflow.keras.layers import Flatten, Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D,     GlobalAveragePooling2D, Dropout, Add, ReLU, Concatenate
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


directory = '/home/ubuntu/Aetrex/home/sonu/sonu/Albie_Ml/regression_modeldata2/'

with open('/home/ubuntu/Aetrex/groundtruth2.json') as json_file:
    gtdata = json.load(json_file)
subdirs = os.listdir(directory)

finalX_img = []
finalX_num = []
finalyLeft = []
finalyRight = []

for i in range(len(subdirs)):
    dir2pull = directory + '/' + subdirs[i] + '/'
    print(dir2pull)

    lefty = gtdata[subdirs[i]]['leftlength']
    righty = gtdata[subdirs[i]]['rightlength']

    for file in glob.glob(dir2pull + '*.csv'):
        data2read = file.replace("\\", '/')
        # print(data2read)
        # print(data2read)
        singledata = pd.read_csv(data2read, header=None)
        singledata = np.array(singledata)

        dataarray = []

        for j in range(singledata.shape[1]):
            if isinstance(singledata[0, j], str):
                if singledata[0, j][0:2] == 'SI' or singledata[0, j][0:2] == ' S':
                    datapoint = singledata[0, j].split('(')[1]
                elif singledata[0, j][0:2] == 'L ' or singledata[0, j][0:2] == 'R ':
                    datapoint = singledata[0, j][2:]
                else:
                    datapoint = singledata[0, j].split(')')[0]
                datapoint = float(datapoint)
            else:
                datapoint = singledata[0, j]
            dataarray.append(datapoint)
        procdata = np.array(dataarray)
        # print(procdata.shape)

        if procdata.shape[0] != 60:
            continue

        finalX_num.append(procdata)
        finalyLeft.append(lefty + random.uniform(-0.5, 0.5))
        finalyRight.append(righty + random.uniform(-0.5, 0.5))

        imgpath = data2read.split('.cs')[0]

        if os.path.isfile(imgpath):
            image = cv2.imread(imgpath)
        else:
            image = cv2.imread(imgpath + '.jpg')

        # print(image.shape)
        image = cv2.resize(image, (224, 224))

        finalX_img.append(image)

# print(len(finalX))

finalX_img = np.array(finalX_img)
print(finalX_img.shape)

finalX_num = np.array(finalX_num)
print(finalX_num.shape)

finalyLeft = np.array(finalyLeft)
finalyRight = np.array(finalyRight)


def shuffle_in_unison_four(a, b, c, d):
    n_elem = a.shape[0]
    indeces = np.random.choice(n_elem, size=n_elem, replace=False)
    return a[indeces], b[indeces], c[indeces], d[indeces]


# In[3]:


# def createinputForModel():


X_img, X_num, yL, yR = shuffle_in_unison_four(finalX_img, finalX_num, finalyLeft, finalyRight)

# In[19]:


# In[4]:


X_img_train = X_img[:120]
X_img_test = X_img[120:]
X_num_train = X_num[:120]
X_num_test = X_num[120:]
yL_train = yL[:120]
yL_test = yL[120:]
yR_train = yR[:120]
yR_test = yR[120:]

X_img_train = X_img
X_num_train = X_num
yL_train = yL
yR_train = yR

# In[20]:


# In[5]:


yL_train.shape
yL_test.shape


# In[6]:


#X_num_train.shape

# In[21]:


# X_num_train = X_num_train.reshape(X_num_train.shape[0], X_num_train.shape[1], 1)
# # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_num_test = X_num_test.reshape(X_num_test.shape[0], X_num_test.shape[1], 1)

# In[22]:


# In[7]:


print(X_num_train.shape, X_num_test.shape)

# In[23]:


# In[21]:


yL_train = yL_train.reshape(142,1)
yL_test = yL_test.reshape(3,1)


# In[22]:


yR_train = yR_train.reshape(142,1)
yR_test = yR_test.reshape(3,1)


# In[23]:


root_logdir = os.path.join(os.curdir, "my_logs")


# In[24]:


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
run_logdir

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# In[ ]:


# In[25]:


# In[25]:


input1 = Input(shape=(None, None, 3))
input2 = Input(shape=(60))

# In[26]:


# In[26]:


input1.shape

# In[27]:


# In[27]:


input2.shape

# In[28]:


# In[28]:


base_model = VGG16(include_top=False, input_tensor=input1)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)

intrins_flat = Flatten()(input2)

merge = Concatenate()([x, intrins_flat])

left = Dense(units=512)(merge)
left = BatchNormalization()(left)
left = ReLU()(left)
left - Dropout(0.5)(left)
left = Dense(units=256)(left)
left = BatchNormalization()(left)
left = ReLU()(left)
left = Dropout(0.5)(left)
left = Dense(units=128)(left)
left = BatchNormalization()(left)
left = ReLU()(left)
left = Dropout(0.5)(left)
left_length = Dense(units=1, activation='linear', name='left_length')(left)

right = Dense(units=512)(merge)
right = BatchNormalization()(right)
right = ReLU()(right)
right = Dropout(0.5)(right)
right = Dense(units=256)(right)
right = BatchNormalization()(right)
right = ReLU()(right)
right = Dropout(0.5)(right)
right = Dense(units=128)(right)
right = BatchNormalization()(right)
right = ReLU()(right)
right = Dropout(0.5)(right)

right_length = Dense(units=1, activation='linear', name='right_length')(right)

model = Model(inputs=[input1, input2], outputs=[left_length, right_length])
for layer in base_model.layers:
    layer.trainable = False

model.summary()

checkPoint = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=False, period=5)

# In[29]:


# In[29]:


print(X_img_train.shape)

# image_train = X_img_train / 255


# In[30]:


print(X_num_train.shape)


# In[31]:


yL_train.shape


# In[32]:


# yL_train = np.expand_dims(yL_train, axis=1)
# yR_train = np.expand_dims(yR_train, axis=1)

# yL_train = np.expand_dims(yL_train, axis=2)
# yR_train = np.expand_dims(yR_train, axis=2)


print(X_img_train.shape)
print(X_num_train.shape)
print(yL_train.shape)
print(yR_train.shape)


# In[33]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model.compile(loss={'left_length': 'mse', 'right_length': 'mse'}, optimizer='adam',
              metrics={'left_length': tf.keras.metrics.RootMeanSquaredError(),
                       'right_length': tf.keras.metrics.RootMeanSquaredError()})

# history = model.fit(X_train, y_train, batch_size = 50, validation_split = 0.2, epochs = 100, verbose = 0)

model.fit([X_img_train, X_num_train], [yL_train, yR_train], batch_size=2, validation_split=0.1, epochs=3, verbose=1,
          callbacks=[tensorboard_cb, checkPoint])
# model.fit([X_CNN1, X_CNN2], y)

print('-----done--------')


# In[ ]:




