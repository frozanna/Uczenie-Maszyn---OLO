import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

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
    model = VGG16(include_top=True, weights='imagenet')
    if summary:
        model.summary()
    return model

def extract_features(model, img):
    pass

if __name__ == '__main__':
    IMAGES_DIR = 'example_images'
    model = load_model()
    img1 = load_and_preprocess_img(os.path.join(IMAGES_DIR, '000094.jpg'))

    extract_features(model, img1)