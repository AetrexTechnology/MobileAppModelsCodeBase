import coremltools
import glob
from keras.layers import Flatten, Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, \
    GlobalAveragePooling2D, Dropout, Add, ReLU, Concatenate
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
import tensorflow as tf

# tf.compat.v1.enable_eager_execution()
# tf.executing_eagerly()


def create_model():
    input1 = Input(shape=(224, 224,3))
    input2 = Input(shape=(1,18))
    
    base_model = VGG16(include_top=False, input_tensor=input1)


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Flatten()(x)

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

# for file in glob.glob('./*.h5'):
#     modelfile2test = file

print('start')
modelfile2test = '/Users/vaneesh_k/PycharmProjects/Albie_ML/weights/18_features/weights_382_00030000.h5'
# load the model we saved
model = create_model()
model.load_weights(modelfile2test)

coreml_model = coremltools.converters.keras.convert(model)
coreml_model.save('/Users/vaneesh_k/PycharmProjects/Albie_ML/mlmodels/cnn_18_features_382.mlmodel')
print('Done')






