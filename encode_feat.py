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

        # allprops = regionprops(binary_mask, temp_mask, 'MeanIntensity', 'BoundingBox');
        img_label = label(binary_mask) # ???
        allprops = regionprops(img_label, ['MeanIntensity', 'BoundingBox']) # ???

        for i, region in allprops:
            allmean.append(region.MeanIntensity)
            allbox.append(region.BoundingBox)
            whichmask.append(i)

    sort_mean = allmean.sort(reverse=True)

    threshold = sort_mean[num_pool]
    upper = np.where(allmean >= threshold)

    final_box = allbox[upper, :]

    final_mask = whichmask[upper]

    output = []

    for upper_idx in range(len(final_mask)):
        region = final_box[upper_idx, :]

        temp_mask = np.squeeze(mask[final_mask[upper_idx], :, :])

        pool_feat = feat[:, region[2]:(region[2]+region[4]-1), region[1]:(region[1]+region[3]-1)]

        pool_mask = temp_mask[region[2]:(region[2]+region[4]-1), region[1]:(region[1]+region[3]-1)]

        # pool_feat = reshape(pool_feat,size(pool_feat,1),[]);
        pool_feat = np.reshape(pool_feat, (len(pool_feat), ??? ))

        # flatten_pool = pool_mask(:)
        flatten_pool = pool_mask.T # ???

        flatten_pool = flatten_pool / norm(flatten_pool)

        pool_multi = pool_feat * flatten_pool

        temp_norm = norm(pool_multi)
        if temp_norm != 0:
            pool_multi = np.divide(pool_multi, temp_norm)

        # output = [output pool_multi]
        output.append(pool_multi) ## ??