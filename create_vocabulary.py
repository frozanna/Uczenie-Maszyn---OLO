from os import listdir
from os.path import isfile, join
from encode_feat import encode_feat
from demo import load_and_preprocess_img, load_model, extract_activations
from clustering import get_clusters
import numpy as np
import pandas


def create_vocabulary(dir, vocabulary_size):
    images = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    model = load_model()
    loaded = [load_and_preprocess_img(i) for i in images]
    extracted = [extract_activations(model, img) for img in loaded]
    encoded = np.array([encode_feat(e[0], e[1]) for e in extracted])
    reshaped = np.reshape(encoded, (-1, 512))
    C, wordcnt = get_clusters(reshaped, vocabulary_size)
    return C, wordcnt


def main():
    images_dir = 'training_images'

    C, wordcnt = create_vocabulary(images_dir, 200)

    Cdf = pandas.DataFrame(data=C)

    Wdf = pandas.DataFrame(data=wordcnt)

    Cdf.to_csv("C.csv", header=None, index=None)
    Wdf.to_csv("wordcnt.csv", header=None, index=None)


if __name__ == '__main__':
    main()