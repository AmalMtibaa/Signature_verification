import numpy as np
from PIL import Image
from scipy.stats import  mode

def resize_image(image, resolution):
    image = image.resize((resolution, resolution), Image.BILINEAR)
    return image

#Center an image on a Square
def pad_image_square_center(image):
    new_size = max(image.size)
    new_image = Image.new(image.mode, (new_size, new_size), 'white')
    position = ((new_size - image.size[0]) // 2,
                (new_size - image.size[1]) // 2)
    new_image.paste(image, position)
    return new_image

def torgb(x) :
    x1 = np.zeros(x.shape + (3,))
    for i in range(224):
        for j in range(224):
            x1[i, j, :] = (x[i, j], x[i, j], x[i, j])
    return x1.reshape((1,)+x1.shape)

def converting_black_and_white(image):
    # Find the mode (background color) and convert to black and white with a
    # threshold based on it.
    threshold = int(0.8 * mode(image.getdata())[0][0]) #0.8 is better then 0.9
    lut = [0] * threshold + [255] * (256 - threshold)
    image = image.point(lut)
    #display_one(image)
    return image

def process_one_image(image,final_res=224, padding=True):
    image = image.convert('L')  # convert gray
    #image=converting_black_and_white(image)
    if padding:
        image = pad_image_square_center(image)
    image = resize_image(image,final_res)
    return np.array(image.getdata()).reshape((final_res, final_res))

def process_image_rgb(image,final_res=224, padding=True):
    image = image.convert('L')  # convert gray
    #image=converting_black_and_white(image)
    if padding:
        image = pad_image_square_center(image)
    image = resize_image(image,final_res)
    image= np.array(image.getdata()).reshape((final_res, final_res))
    return torgb(image)