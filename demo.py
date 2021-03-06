import os
import sys
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # filter out tensorflow messages

import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras import models
import tensorflow as tf

from encode_feat import encode_feat
from compare_two import compare_two

tf.get_logger().setLevel('ERROR') 

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

def command_line_args():
    parser = argparse.ArgumentParser(description=
    'Compare example image with images in the dataset and return possible location.')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--image', help='Example image name',
                                required=True)
    return parser.parse_args()

def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    model = load_model()

    query_images_dir = os.path.join(current_path, 'query_images')
    directory = os.fsencode(query_images_dir)
    args = command_line_args()
    example_image_name = args.image

    img1 = load_and_preprocess_img(os.path.join(current_path, 
                                    'example_images',
                                    example_image_name))
    feat1, mask1 = extract_activations(model, img1)
    encode1 = encode_feat(feat1, mask1)

    print('Possible places:')

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        img2 = load_and_preprocess_img(os.path.join(query_images_dir, file_name))
        feat2, mask2 = extract_activations(model, img2)

        encode2 = encode_feat(feat2, mask2)
        if compare_two(encode1, encode2) > 0.15:
            print(os.path.splitext(file_name)[0])


if __name__ == '__main__':
    main()

