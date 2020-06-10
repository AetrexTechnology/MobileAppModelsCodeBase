import coremltools
import os
import os.path
import glob
import numpy as np
import json
import random
import pandas as pd
import cv2



directory = './testingdata1'

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



model = coremltools.models.MLModel('/Users/vaneesh_k/PycharmProjects/Albie_ML/cnn_99_2.mlmodel')

for i in range(len(finalX_img)):
    test_image = finalX_img[i, :, :, :]
    test_image = np.array(test_image, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    ar_data = finalX_num[i, :]
    ar_data = np.array(ar_data, dtype=np.float32)
    ar_data = np.expand_dims(ar_data, axis=0)
    # ar_data = np.expand_dims(ar_data, axis=2)
    # print(ar_data.shape)
    result = model.predict({'input_1':test_image,'input_2':ar_data})
    # result = model.predict([test_image, ar_data])

print('Done')