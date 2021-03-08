import os

import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K

def load_and_preprocess_img(path, target_size=(224, 224)):
    image = load_img(path, target_size=target_size)
    # convert the image pixels to a numpy array
    image = img_to_array(image) 
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    return image

def load_model(summary=False):
    model = VGG16(include_top=False, weights='imagenet')
    if summary:
        model.summary()
    return model

def extract_features(model, img, print_layers=False):
    ''' Returns feat and mask with shape 512x14x14 each. '''

    if print_layers:
        print(f'Model layers: {[layer.name for layer in model.layers]}')

    # input placeholder
    model_input = model.input
    # the second last convolutional layer to extract local descriptors
    feat = model.get_layer('block5_conv2').output
    # the last convolutional layer to discover salient regions
    mask = model.get_layer('block5_conv3').output 
    # concatenate outputs
    outputs = [feat, mask]
    # evaluation function
    functor = K.function([model_input], outputs)
    # get the outputs of layers
    layer_outs = functor([img])
    # squeeze 
    layer_outs = [np.squeeze(elem) for elem in layer_outs]
    # swap axes
    layer_outs = [np.swapaxes(elem, 0, 2) for elem in layer_outs]

    return layer_outs

if __name__ == '__main__':
    IMAGES_DIR = 'example_images'
    model = load_model()
    img1 = load_and_preprocess_img(os.path.join(IMAGES_DIR, '000094.jpg'))
    feat, mask = extract_features(model, img1)
