import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd


def compare_two(test_feat, refer_feat):
    mask_channel = np.size(test_feat, 0)

    test_temp = np.dot(test_feat, refer_feat.T)

    col_max = np.amax(test_temp, axis=0)
    col_max_idx = np.argmax(test_temp, axis=0)

    row_max = np.amax(test_temp, axis=1)
    row_max_idx = np.argmax(test_temp, axis=1)

    mutual = 0

    C = pd.read_csv("build_vocabulary/C.csv", delimiter=',', header=None).values

    wordcnt = pd.read_csv("build_vocabulary/wordcnt.csv", delimiter=',', header=None).values

    wordcnt = np.squeeze(wordcnt)

    word_size = wordcnt.size

    totalimg = 401

    for inner in range(mask_channel):
        if col_max_idx[row_max_idx[inner]] != inner:
            row_max[inner] = 0
        else:
            if row_max[inner] != 0:
                mutual = mutual + 1

    nonzero = np.where(row_max != 0)
    test_nonzero = np.squeeze(test_feat[nonzero, :])
    refer_nonzero = np.squeeze(refer_feat[row_max_idx[nonzero], :])

    testdist = cdist(C, test_nonzero)
    referdist = cdist(C, refer_nonzero)

    t_min_idx = np.argmin(testdist, axis=0)

    r_min_idx = np.argmin(referdist, axis=0)

    row_max[nonzero] = (
        np.dot(
            np.dot(
                row_max[nonzero],
                np.log10(
                    np.divide(totalimg, wordcnt[t_min_idx])
                ).T),
            np.log10(
                np.divide(totalimg, wordcnt[r_min_idx])
            ).T
        )
    )

    return np.sum(row_max)/mask_channel