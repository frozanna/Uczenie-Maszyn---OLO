import numpy as np
import cv2
from skimage.measure import regionprops, label
from numpy.linalg import norm


def encode_feat(feat, mask):
    num_pool = 200

    allmean = []
    allbox = []
    whichmask = []

    for mask_idx in range(len(mask)):
        temp_mask = np.squeeze(mask[mask_idx, :, :])

        (thresh, binary_mask) = cv2.threshold(temp_mask, temp_mask.min(), 255, cv2.THRESH_BINARY)

        img_label = label(binary_mask)
        allprops = regionprops(img_label, intensity_image=temp_mask)

        for i, region in enumerate(allprops):
            allmean.append(region.mean_intensity)
            allbox.append(region.bbox)
            whichmask.append(i)

    sort_mean = np.sort(allmean)[::-1]

    threshold = sort_mean[num_pool]
    upper = np.where(allmean >= threshold)

    final_box = np.array(allbox)[upper]

    final_mask = np.array(whichmask)[upper]

    output = []

    for upper_idx in range(len(final_mask)):
        region = final_box[upper_idx, :]

        temp_mask = np.squeeze(mask[final_mask[upper_idx], :, :])

        pool_feat = feat[:, region[1]:(region[1]+region[3]-1), region[0]:(region[0]+region[2]-1)]

        pool_mask = temp_mask[region[1]:(region[1]+region[3]-1), region[0]:(region[0]+region[2]-1)]

        pool_feat = np.reshape(pool_feat, (pool_feat.shape[0], -1))

        flatten_pool = pool_mask.flatten()
        temp_norm = norm(flatten_pool)

        if temp_norm != 0:
            flatten_pool = flatten_pool / temp_norm

        pool_multi = pool_feat @ flatten_pool

        temp_norm = norm(pool_multi)
        if temp_norm != 0:
            pool_multi = np.divide(pool_multi, temp_norm)

        output.append(pool_multi) ## ??

    return output