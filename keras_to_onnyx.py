import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import keras2onnx
import onnxruntime
import coremltools
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
import tfcoreml


# load keras model

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


dependencies = {
    # 'auc_roc': root_mean_squared_error,
    # 'left_length': root_mean_squared_error,
    # 'right_length': root_mean_squared_error,
    'root_mean_squared_error': root_mean_squared_error,
    'GlorotUniform': glorot_uniform
}
# load model
model = load_model('cnn_99_2.h5',custom_objects=dependencies)
# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

temp_model_file = 'model.onnx'
keras2onnx.save_model(onnx_model, temp_model_file)
sess = onnxruntime.InferenceSession(temp_model_file)
print('DONE')