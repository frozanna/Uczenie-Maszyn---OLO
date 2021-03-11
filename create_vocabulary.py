from os import listdir
from os.path import isfile, join
from encode_feat import encode_feat
from demo import load_and_preprocess_img, load_model, extract_activations
import numpy as np

def create_vocabulary(images):
    model = load_model()
    loaded = [load_and_preprocess_img(i) for i in images]
    extracted = [extract_activations(model, img) for img in loaded]
    encoded = np.array([encode_feat(e[0], e[1]) for e in extracted])
    reshaped = np.reshape(encoded, (-1, 512))
    return reshaped

def get_files_from_dir(dir):
    return [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]

def main():
    images_dir = 'example_images'

    images = get_files_from_dir(images_dir)

    print(np.shape(create_vocabulary(images)))


if __name__ == '__main__':
    main()