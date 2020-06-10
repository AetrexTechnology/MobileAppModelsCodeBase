from keras.preprocessing import image
import numpy as np
from tensorflow import keras
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# dimensions of our images    -----   are these then grayscale (black and white)?
# img_width, img_height = 313, 220

# load the model we saved
model =  keras.models.load_model('/Users/vaneesh_k/Downloads/cnn_99_2.h5')

# Get test image ready
test_image = image.load_img('/Users/vaneesh_k/PycharmProjects/Albie_ML/regression_modeldata1/vaneesh/123829.jpg',custom_objects={'left_length': root_mean_squared_error,'right_length':root_mean_squared_error})
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
ar_data = [-1.0045258, -1.0045258, -0.5915225, -0.08197192, 0.083296426, 0.17892422, 0.0, -1.0045259, -0.43184212,
           -0.12142114, 0.23065814, -1.0045259, -0.24669495, -0.0816838, 0.22575317, 0.1793735, -0.18922983, 0.98191434,
           0.0060414104, -0.25371173, -0.013512621, -0.008755976, 0.99987036, -0.10162211, 0.98183984, 0.18912373,
           0.014925127,
           -0.20469286, 0.0, 0.0, 0.0, 0.99999994, -1.5546939, 0.38462126, -2.1457539, 1464.5238, 0.0, 958.8872, 0.0,
           1464.5238, 716.26074, 0.0, 0.0, 1.0, 1.5255456, 0.0, 0.00063830614, 0.0, 0.0, 2.034061, -0.0044989586, 0.0,
           0.0, 0.0, -0.99999976, -0.0009999998, 0.0, 0.0, -1.0, 0.0]
# test_image = test_image.reshape(img_width, img_height * 3)  # Ambiguity!
# Should this instead be: test_image.reshape(img_width, img_height, 3) ??

result = model.predict(test_image,ar_data)
print(result)
