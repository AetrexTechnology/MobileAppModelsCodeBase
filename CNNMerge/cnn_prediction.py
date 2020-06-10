#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing import image
import numpy as np
# from tensorflow import keras
import tensorflow as tf
import keras
import os
import os.path
import glob
import numpy as np
import pandas as pd
import cv2
import json
# import seaborn as sns
import random
import matplotlib.pyplot as plt
import json, glob
from keras import backend as K
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

# In[2]:
rgb_weights = [0.2989, 0.5870, 0.1140]


def createMask(imagePath):
    image = cv2.imread(imagePath)
    boundaries = [
        # ([100, 0, 0], [255, 70, 79]),
        ([0, 0, 100], [40, 40, 255])
    ]
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        max_y = getPixelLocation(output) + 10
        crop_image = image[max_y - image.shape[1]:max_y, :, :]
        return crop_image


def getPixelLocation(mask):
    grayscale_image = np.dot(mask[..., :3], rgb_weights)
    output2 = np.where(grayscale_image != 0)
    max_y = output2[0].max()
    return max_y


def create_model():
    input1 = Input(shape=(None, None, 3))
    input2 = Input(shape=(1))

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

    return model


print(tf.__version__)

# In[3]:


directory = '/Users/vaneesh_k/Documents/stuff/tesingDataForPrediction'

with open('/Users/vaneesh_k/PycharmProjects/Albie_ML/testingdata1/groundtruth1.json') as json_file:
    gtdata = json.load(json_file)
subdirs = os.listdir(directory)

finalX_img = []
finalX_num = []
finalyLeft = []
finalyRight = []

for i in range(len(subdirs)):
    if subdirs[i][-4:] == 'json':
        continue

    dir2pull = directory + '/' + subdirs[i] + '/'
    print(dir2pull)

    lefty = gtdata[subdirs[i]]['leftlength']
    righty = gtdata[subdirs[i]]['rightlength']

    for file in glob.glob(dir2pull + '*.csv'):
        data2read = file.replace("\\", '/')
        print(data2read)
        # print(data2read)
        singledata = pd.read_csv(data2read, header=None)
        singledata = np.array(singledata)

        if singledata.shape[1] != 60:
            continue

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
            image = createMask(imgpath)
        else:
            image = createMask(imgpath + '.jpg')

        image = cv2.resize(image, (224, 224))

        finalX_img.append(image / 255)

# print(len(finalX))

finalX_img = np.array(finalX_img)
print(finalX_img.shape)

finalX_num = np.array(finalX_num)
print(finalX_num.shape)

finalyLeft = np.array(finalyLeft)
finalyRight = np.array(finalyRight)

# In[4]:


finalX_num = pd.DataFrame(finalX_num).drop(
    [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32,33,34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
     44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59], axis=1).to_numpy()

# In[5]:


finalX_num.shape


# In[6]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


from tensorflow.keras.initializers import glorot_uniform

dependencies = {
    # 'auc_roc': root_mean_squared_error,
    # 'left_length': root_mean_squared_error,
    # 'right_length': root_mean_squared_error,
    'root_mean_squared_error': root_mean_squared_error
    # 'GlorotUniform': glorot_uniform()
}

for file in glob.glob('//Users/vaneesh_k/PycharmProjects/Albie_ML/weights/distance_camera_kumar/weights00002000.h5'):
    modelfile2test = file

# In[7]:


# load the model we saved
model = create_model()
model.load_weights(modelfile2test)

# In[8]:


finalX_num.shape

# In[9]:


yleft_pred = []
yright_pred = []


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


for i in range(len(finalX_img)):

    test_image = finalX_img[i, :, :, :]
    test_image = np.array(test_image, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    ar_data = finalX_num[i, :]
    ar_data = np.array(ar_data, dtype=np.float32)
    ar_data = np.expand_dims(ar_data, axis=0)
    # ar_data = np.expand_dims(ar_data,axis=2)
    # print(ar_data.shape)

    result = model.predict([test_image, ar_data])
    print(result)
    leftpred = np.mean(result[0])
    rightpred = np.mean(result[1])

    yleft_pred.append(leftpred)
    yright_pred.append(rightpred)

print('Left Foot RMSE: {}'.format(rmse(finalyLeft, np.array(yleft_pred))))
print('Right Foot RMSE: {}'.format(rmse(finalyRight, np.array(yright_pred))))

# In[ ]:


# In[ ]:
