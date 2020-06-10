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
import seaborn as sns
import random
import matplotlib.pyplot as plt
import json
from keras import backend as K
from tensorflow.keras.layers import Flatten, Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, \
    GlobalAveragePooling2D, Dropout, Add, ReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from keras import backend as K
from tensorflow import keras
import tensorflow as tf

print(tf.__version__)

directory = './testingdata1/'

with open(directory + '/groundtruth1.json') as json_file:
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
        # print(data2read)
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
            image = cv2.imread(imgpath)
        else:
            image = cv2.imread(imgpath + '.jpg')

        # print(image.shape)
        if image.shape[0] != 1792:
            image = cv2.resize(image, (828, 1792))

        finalX_img.append(image)

# print(len(finalX))

finalX_img = np.array(finalX_img)
# print(finalX_img.shape)

finalX_num = np.array(finalX_num)
# print(finalX_num.shape)

finalyLeft = np.array(finalyLeft)
finalyRight = np.array(finalyRight)


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

for file in glob.glob('./*.h5'):
    modelfile2test = file
print(modelfile2test)


# load the model we saved

def create_model():
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


def train():
    model = create_model()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    checkPoint = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=250)
    model.compile(loss={'left_length': 'mse', 'right_length': 'mse'}, optimizer='adam',
                  metrics={'left_length': root_mean_squared_error, 'right_length': root_mean_squared_error})

    model.fit([X_img_train, X_num_train], [yL_train, yR_train], batch_size=2, validation_split=0.1, epochs=10000,
              verbose=1,
              callbacks=[tensorboard_cb, checkPoint])


def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)


model = tf.keras.models.load_model(modelfile2test, custom_objects=dependencies)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


yleft_pred = []
yright_pred = []

for i in range(len(finalX_img)):
    test_image = finalX_img[i, :, :, :]
    test_image = np.array(test_image, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    ar_data = finalX_num[i, :]
    ar_data = np.array(ar_data, dtype=np.float32)
    ar_data = np.expand_dims(ar_data, axis=0)
    ar_data = np.expand_dims(ar_data, axis=2)
    # print(ar_data.shape)

    result = model.predict([test_image, ar_data])
    # print(np.squeeze(result[0]).shape)
    leftpred = np.mean(np.squeeze(result[0]))
    rightpred = np.mean(np.squeeze(result[1]))

    yleft_pred.append(leftpred)
    yright_pred.append(rightpred)

print('Left Foot RMSE: {}'.format(rmse(finalyLeft, np.array(yleft_pred))))
print('Right Foot RMSE: {}'.format(rmse(finalyRight, np.array(yright_pred))))
