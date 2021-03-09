import os

import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras import models

def load_and_preprocess_img(path, target_size=(224, 224)):
    image = load_img(path, target_size=target_size)
    # convert the image pixels to a numpy array
    image = img_to_array(image) 
    # insert a new axis to transform data to shape (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    # prepare the image for the VGG modelS
    image = preprocess_input(image)
    return image

def load_model(summary=False):
    model = VGG16(include_top=False, weights='imagenet')
    if summary:
        model.summary()
    return model

def extract_activations(model, img, print_layers=False):
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
    # create a model that takes input image and 
    # outputs the activations of selected layers
    activation_model = models.Model(inputs=model_input, outputs=outputs)
    activations = activation_model.predict(img)
    # squeeze 
    activations = [np.squeeze(elem) for elem in activations]
    # swap axes
    activations = [np.swapaxes(elem, 0, 2) for elem in activations]

    return activations

def main():
    images_dir = 'example_images'
    img1 = load_and_preprocess_img(os.path.join(images_dir, '000050.jpg'))
    img2 = load_and_preprocess_img(os.path.join(images_dir, '000094.jpg'))
    
    model = load_model()
    feat1, mask1 = extract_activations(model, img1)
    feat2, mask2 = extract_activations(model, img2)

    # TO DO: encode 
    # TO DO: compare

if __name__ == '__main__':
    main()

