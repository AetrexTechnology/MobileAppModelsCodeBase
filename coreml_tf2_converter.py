from tensorflow.keras.applications import MobileNet
import tfcoreml
import numpy as np

# keras_model = MobileNet(weights=None, input_shape=(224, 224, 3))
# keras_model.save('./savedmodel', save_format='tf')
# tf.saved_model.save(keras_model, './savedmodel')
# def root_mean_squared_error(y_true, y_pred):
#     return np.sqrt(np.mean(np.square(y_pred - y_true)))


model = tfcoreml.convert('weights00000005.h5',
                         mlmodel_path='./cnn_99_2.mlmodel',
                         input_name_shape_dict={'input_1': (1,1792, 828, 3), 'input_2': (1,60)},
                         output_feature_names=['Identity'],
                         minimum_ios_deployment_target='13')
print('------Done---------')
