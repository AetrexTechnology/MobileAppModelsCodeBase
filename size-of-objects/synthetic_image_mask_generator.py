# importing libraries
import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw

# weights for R,G,B for grayscale to rgb conversion (Weighted method or luminosity method)
rgb_weights = [0.2989, 0.5870, 0.1140]

# coins image directory
coin_path = '/Users/vaneesh_k/PycharmProjects/deep_lab/coin/images/syn_coin_img/coins_cropped/'
# foot images directory
feet_path = '/Users/vaneesh_k/PycharmProjects/deep_lab/coin/images/syn_coin_img/feet'
# output directory where generated images are saved
output_directory = '/Users/vaneesh_k/PycharmProjects/deep_lab/coin/images/syn_coin_img/syn_data_generated/'

coins_file_names = os.listdir(coin_path)  # contains all the images in the directory


def createMask(imagePath):
    '''

    :param imagePath: path for image
    :return: the location of y cordinate found according to the color found in the base-line (red,blue)
    '''
    image = cv2.imread(imagePath)
    boundaries = [
        # ([100, 0, 0], [255, 70, 79]),
        ([0, 0, 0], [255, 0, 20])
    ]
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        max_y = get_y_pixel_location_of_line(output) + 10
        crop_image = image[max_y - image.shape[1]:max_y, :, :]
        return max_y


def get_y_pixel_location_of_line(mask):
    '''

    :param mask: thresholded mask of the image
    :return: the y location of the base-line pixel found according to the rgb value searched for (red,blue)
    '''
    grayscale_image = np.dot(mask[..., :3], rgb_weights)
    output2 = np.where(grayscale_image != 0)
    max_y = output2[0].max()
    return max_y



def generate_syntetic_image():
    '''
    This function generates the images by putting coins on feet images randomly
    '''
    global im2, mask_im
    imagePath = (os.path.join(feet_path, file))
    coin_img = coin_path + random.choice(coins_file_names)
    im1 = Image.open(coin_img)
    # resize coin image
    im1 = im1.resize((50, 50))
    Image1copy = im1.copy()
    # open the image
    im2 = Image.open(imagePath)
    # resize foot image into one size
    im2 = im2.resize((828, 1792))
    mask_im = Image.new("L", im1.size, 0)
    a, b = im1.size
    draw = ImageDraw.Draw(mask_im)
    draw.ellipse((0, 0, a, b), fill=255)
    back_im = im2.copy()

    back_im.paste(Image1copy, (X, Y), mask_im)
    # save the image
    back_im.save(output_directory+ file)


def save_synthetic_image_mask():
    '''
    This function generates the mask for coin image generated synthetically and then saves the mask
    '''
    image_mask = Image.new("L", im2.size, 0)
    image_mask_np = np.array(image_mask)
    mask_im_np = np.array(mask_im)
    image_mask_np[Y:Y + 50, X:X + 50] = mask_im_np
    image_mask_image = Image.fromarray(image_mask_np)
    image_mask_image.save(output_directory + file[:-4] + '_mask.jpg')

# Here we loop via all images(jpg) in directory, generate their masks and synthetic image
for file in os.listdir(feet_path):
    if file.endswith(".jpg"):
        # generate te image and then saves it
        X = random.randrange(100, 750, 10)
        Y = random.randrange(900, 1350, 50)
        generate_syntetic_image()
        # code to generate a mask for training
        save_synthetic_image_mask()
        print('-----generating image - ', file)
