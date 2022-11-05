import cv2
import numpy as np

from pygo.Webcam import Webcam
from pygo.utils.image import *
from skimage import segmentation

from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def seg_slic(img):
    #NORMAL
    # convert to gray

    labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
    g = graph.rag_mean_color(img, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)

    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
    return out


def seg_chan(img):
    gimage = inverse_gaussian_gradient(img)

    # Initial level set
    init_ls = np.zeros(img.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    ls = morphological_geodesic_active_contour(gimage, iterations=230,
                                               init_level_set=init_ls,
                                               smoothing=1, balloon=-1,
                                               threshold=0.69)
    return ls

if __name__ == "__main__":
    clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    ipt = Webcam()
    while True:
        img = ipt.read()
        cv2.imshow("original", img)
        img_c = img.copy()
        hsv = toHSVImage(  img.copy())
        yuv = toYUVImage(  img.copy())
        cmyk = toCMYKImage(img.copy())
        gray = toGrayImage(img.copy())

        #cv2.imshow('rgb', seg_slic(img))
        #cv2.imshow('hsv', seg_slic(hsv))
        #cv2.imshow('cmyk', seg_slic(cmyk[:,:,3]))
        cv2.imshow('chan_cmyk', seg_chan(cmyk[:,:,3]))
        cv2.waitKey(1)

