
from scipy.stats import skew, moment
from skimage.feature import hog
import numpy as np
from skimage import exposure, img_as_float
import pdb

def get_feat_vec(d):
    feat = []
    assert(d.dtype ==  np.float64)
    assert(d.shape == (32,32))
    assert(d.max() <= 1.0)

    fd, hog_image = hog(d, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(1, 1), visualize=True)
    hi = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    feat.append(np.mean(d.reshape(-1)))
    feat.append(np.var(d.reshape(-1)))
    feat.append(skew(d.reshape(-1)))
    feat.append(skew(d.reshape(-1)))
    feat.append(moment(d.reshape(-1)))
    feat.append(hi.reshape(-1))

    return np.hstack(feat)
