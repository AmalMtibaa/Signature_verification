# Here we used the concept of data Augmentation we didn 't use it in the final model but it's a concept that
# could be used in the future, the function one_image_augmentation will generate different aspects for the same

import numpy as np
from PIL import Image
from scipy.stats import mode

from data_visualisation import *
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

K.set_image_dim_ordering('th') #"th" format means that the convolutional kernels will have the shape (depth, input_depth, rows, cols)
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             zca_epsilon=1e-06,
                             rotation_range=30,  # modify this
                             width_shift_range=10,  # modify this 10
                             height_shift_range=10,  # modify this
                             brightness_range=None,
                             shear_range=0.0,
                             zoom_range=0.1,  # modify this
                             channel_shift_range=0,  # modify this
                             fill_mode='constant',  # specify this depending on the usecase
                             cval=1.0,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=None,  # modify this
                             preprocessing_function=None,
                             data_format=None,
                             validation_split=0.0,
                             dtype=None)


#Converting an image to black and white
def converting_black_and_white(image):
    # Find the mode (background color) and convert to black and white with a
    # threshold based on it.
    threshold = int(0.9 * mode(image.getdata())[0][0]) #0.8 is better then 0.9
    lut = [0] * threshold + [255] * (256 - threshold)
    image = image.point(lut)
    #display_one(image)
    return image


#Resize an image to the unified size
def resize_image(image, resoltion):
    image = image.resize((resoltion, resoltion), Image.BILINEAR)
    return image

#Center an image on a Square
def pad_image_square_center(image):
    new_size = max(image.size)
    new_image = Image.new(image.mode, (new_size, new_size), 'white')
    position = ((new_size - image.size[0]) // 2,
                (new_size - image.size[1]) // 2)
    new_image.paste(image, position)
    return new_image



def process_one_image(image,size=224, padding=True):
    image = image.convert('L')  # convert gray
    image=converting_black_and_white(image)
    if padding:
        image = pad_image_square_center(image)
    image = resize_image(image,size)

    return np.array(image.getdata()).reshape((size, size))/ 255

img_width=224
img_height=224
dim = (img_width, img_height)


def one_image_augmentation(path,aug_size,image_size):
    image = Image.open(path)
    image=process_one_image(image,padding=True)
    generated_images = []
    for i in range(aug_size):
        im_aug = datagen.random_transform(image.reshape(1,image_size, image_size))  # random_transform applies a random transformation to an image
        #display_one(im_aug.reshape(image_size,image_size),'Augmented '+ str(i) )
        generated_images.append(im_aug.reshape(image_size,image_size))
    return generated_images




def torgb(x) :
    x1 = np.zeros(x.shape + (3,))
    for i in range(224):
        for j in range(224):
            x1[i, j, :] = (x[i, j], x[i, j], x[i, j])
    return x1.reshape((1,)+x1.shape)

