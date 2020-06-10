# %%

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
from tensorflow.keras.layers import Flatten, Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, \
    GlobalAveragePooling2D, Dropout, Add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow import keras

# %%

directory = '/Users/vaneesh_k/Downloads/home 3/sonu/sonu/Albie_Ml/regression_modeldata2'
# directory = '/Users/vaneesh_k/PycharmProjects/Albie_ML/regression_modeldata1'

with open('/Users/vaneesh_k/Downloads/groundtruth.json') as json_file:
# with open('/Users/vaneesh_k/PycharmProjects/Albie_ML/groundtruth.json') as json_file:
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
        if image.shape[0] != 1792:
            image = cv2.resize(image,(828,1792))

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


# def createinputForModel():


X_img, X_num, yL, yR = shuffle_in_unison_four(finalX_img, finalX_num, finalyLeft, finalyRight)

# %%

X_img_train = X_img[:100]
X_img_test = X_img[100:]
X_num_train = X_num[:100]
X_num_test = X_num[100:]
yL_train = yL[:100]
yL_test = yL[100:]
yR_train = yR[:100]
yR_test = yR[100:]

# %%

X_num_train.shape

# %%

X_num_train = X_num_train.reshape(X_num_train.shape[0], X_num_train.shape[1], 1)
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_num_test = X_num_test.reshape(X_num_test.shape[0], X_num_test.shape[1], 1)

# %%

print(X_num_train.shape, X_num_test.shape)

# %%

root_logdir = os.path.join(os.curdir, "my_logs")


# %%
# Setting tensorflow graphs

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
run_logdir

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# %%


# %%

input1 = Input(shape=(None, None, 3))
input2 = Input(shape=(None, 1))

# %%

input1.shape

# %%

input2.shape

# %%

base_model = VGG16(include_top=False, input_tensor=input1)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)

merge = Add()([x, input2])

left = Dense(units=1024, activation='relu')(merge)
left = Dense(units=512, activation='relu')(left)
left = Dense(units=128, activation='relu')(left)
left_length = Dense(units=1, activation='linear', name='left_length')(left)

right = Dense(units=1024, activation='relu')(merge)
right = Dense(units=512, activation='relu')(right)
right = Dense(units=128, activation='relu')(right)

right_length = Dense(units=1, activation='linear', name='right_length')(right)

model = Model(inputs=[input1, input2], outputs=[left_length, right_length])
for layer in base_model.layers:
    layer.trainable = False

model.summary()

# %%

model.compile(loss={'left_length': 'mse', 'right_length': 'mse'}, optimizer='sgd',
              metrics={'left_length': 'mse', 'right_length': 'mse'})

# history = model.fit(X_train, y_train, batch_size = 50, validation_split = 0.2, epochs = 100, verbose = 0)

model.fit([X_img_train, X_num_train], [yL_train, yR_train], batch_size=32, validation_split=0.2, epochs=100, verbose=1,
          callbacks=[tensorboard_cb])
# model.fit([X_CNN1, X_CNN2], y)

print('-----done--------')

# %%
#
# %load_ext tensorboard
# %tensorboard --logdir=./my_logs --port=6006
# %%
