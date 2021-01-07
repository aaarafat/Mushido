
import cv2
import os
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, rgb2hsv, rgba2rgb
from skimage import img_as_ubyte
from skimage.transform import rotate

import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin, binary_opening
from skimage.measure import find_contours
import skimage.draw as draw
import skimage.filters as filters
import skimage.exposure as ex
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb


# Edges
from skimage.filters import sobel_h, sobel, sobel_v, roberts, prewitt


# Show the figures / plots inside the notebook


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def get_dimensions(w, h):
    area = w * h
    if (area < 400000):
        return 15, 0
    elif (999999 > area > 400000):
        return 31, 0
    else:
        aspect = w / h
        nw = np.sqrt(750000*aspect)
        return 35, int(nw)


def load_images_from_folder(folder):
    images = []
    files = []
    print(folder)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            files.append(filename)
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return images, files


def integral_image(img):
    rows, cols = img.shape
    int_img = np.zeros(img.shape)
    int_img[0][0] = img[0][0]
    for r in range(1, rows):
        int_img[r][0] = int_img[r-1][0] + img[r][0]
    for c in range(1, cols):
        int_img[0][c] = int_img[0][c-1] + img[0][c]
    for r in range(1, rows):
        for c in range(1, cols):
            int_img[r][c] = img[r][c]+int_img[r][c-1] + \
                int_img[r-1][c] - int_img[r-1][c-1]
    return int_img


def thresholding_segment(img, window, t=10):
    rows, cols = img.shape
    output = np.zeros(img.shape, dtype="uint8")
    int_img = integral_image(img)
    s = int(window/2)
    p_img = np.pad(img, s, "constant")
    p_int = np.pad(int_img, s, 'edge')
    for r in range(s+1, rows+s):
        for c in range(s+1, cols+s):
            x1 = c-s
            x2 = c+s
            y1 = r-s
            y2 = r+s
            count = (x2-x1)*(y2-y1)
            sum = p_int[y2, x2] - p_int[y2, x1-1] - \
                p_int[y1-1, x2] + p_int[y1-1, x1-1]
            if(img[r-s][c-s]*count) <= (sum*(100-t)/100):
                output[r-s][c-s] = 0
            else:
                output[r-s][c-s] = 255
    return output
