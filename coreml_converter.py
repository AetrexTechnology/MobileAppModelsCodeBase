import coremltools
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.initializers import glorot_uniform


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


dependencies = {
    'GlorotUniform': glorot_uniform()
}
# load model
model = tf.keras.models.load_model('weights00000005.h5')

coreml_model = coremltools.converters.tensorflow.convert(model)
coreml_model.save('cnn_regression.mlmodel')
# print(model.get_spec())

print("--------Done---------")
