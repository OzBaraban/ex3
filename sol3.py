from scipy import ndimage
import scipy
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    bulids a gaussian pyramid as shown in class
    :param im - matrix represents an image:
    :param max_levels - the maximal number of levels in the resulting pyramid:
    :param filter_size - an odd scalar that represents a squared filter:
    :return pyr - a standard python array where each element of the array is a grayscale image:
            filter-vec - row vector of shape (1, filter_size) used for the pyramid construction:
    """
    pyr = [im]
    filter_vec = gaussian_helper(filter_size)
    for level in range(max_levels-1):
        if im.shape[0] <= 16 or im.shape[1] <= 16:
            return pyr, filter_vec
        filtered_im = scipy.ndimage.filters.convolve(im, filter_vec)
        filtered_im = scipy.ndimage.filters.convolve(filtered_im, filter_vec.T)
        im = filtered_im[::2, ::2]
        pyr.append(im)
    return pyr, filter_vec


def gaussian_helper(filter_size):
    """
    calculates the gaussian filter using convolution
    :param filter_size:
    :return normalised np.array which represents the gaussian filter:
    """
    filter_size = max(2, filter_size)
    base = np.array([1, 1])
    if filter_size == 1:
        base = base/2
        return np.reshape(base, (1, filter_size))
    for i in range(1, filter_size-1):
        base = np.convolve(base, np.array([1, 1]))
    base = base / (2**(filter_size-1))
    return np.reshape(base, (1, filter_size))


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    bulids a laplacian pyramid from a given image, uses the build_gaussian_pyramid function before since a laplacian
    pyramid is based on gaussian pyramid
    :param im - a matrix represents an image:
    :param max_levels - the maximal number of levels in the resulting pyramid:
    :param filter_size - an odd scalar that represents a squared filter:
    :return pyr - a standard python array where each element of the array is a grayscale image:
            filter-vec - row vector of shape (1, filter_size) used for the pyramid construction:
    """
    g_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for idx in range(len(g_pyr)-1):
        pyr.append(g_pyr[idx]-expand(g_pyr[idx+1], filter_vec))
    pyr.append(g_pyr[-1])
    return pyr, filter_vec


def expand(im, filter_vec):
    """
    expands the image according to the method we saw in class
    :param im - a matrix represents an image we wnat to expand:
    :param filter_vec - the vector which represent the blur applied to the image:
    :return filtered_im - the expanded image after the expand algorithm:
    """
    rows, cols = im.shape
    new_rows, new_cols = rows*2, cols*2
    expanded_im = np.zeros([new_rows, new_cols])
    expanded_im[::2, ::2] = im
    filter_vec = np.array([filter_vec[0]*2])
    filtered_im = scipy.ndimage.filters.convolve(expanded_im, filter_vec)
    filtered_im = scipy.ndimage.filters.convolve(filtered_im, filter_vec.T)
    return filtered_im


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    a function which sums up all the layers of the laplacian pyramid
    :param lpyr - a standard python array where each element of the array is a grayscale image in a laplacian pyramid:
    :param filter_vec - the vector which represent the blur applied to the image:
    :param coeff -  a python list, the function multiply each level i of the laplacian pyramid by its corresponding
     coefficient coeff[i]:
    :return im - the result of the sum of the laplacian levels, which is an image:
    """
    levels = len(lpyr)
    lpyr_scaled = []
    #create a list of newly scaled levels of the laplacian pyramid
    for level in range(levels):
        scaled_im = coeff[level]*lpyr[level]
        lpyr_scaled.append(scaled_im)
    #taking the top of the pyramid element since it dont need to expand
    im = lpyr_scaled[-1]
    #summing all the layers in the pyrmaid
    for level in range(levels-2, -1, -1):
        im = lpyr_scaled[level] + expand(im, filter_vec)
    return im


def render_pyramid(pyr, levels):
    """
    creats the blackend area where images from the pyramids are going to be and places the images on top of it
    :param pyr - Gaussian or Laplacian pyramid - a standard python array where each element of the array is a grayscale
           image:
    :param levels - the number of levels to present in the result ≤ max_levels:
    :return blackend area with the images from the pyramid are in:
    """
    im = pyr[0]
    rows, cols = np.shape(im)
    #formula for the sum of the geomatric progression
    mod_im_cols = int(cols*(0.5**levels-1)/-0.5)
    canvas = np.zeros([rows, mod_im_cols])
    start_idx = 0
    level = 0
    for image in pyr:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        rows, cols = np.shape(image)
        image = image.astype(np.float64)
        canvas[:rows, start_idx:start_idx+cols] = image
        start_idx += cols
        level += 1
        if level == levels:
            break
    return canvas


def display_pyramid(pyr, levels):
    """
    presents all the images in a given pyramid.
    :param pyr - r a Gaussian or Laplacian pyramid as defined in the previous functions:
    :param levels - is the number of levels(include the original image) to present which is <= to max_levels in the
           pyramids :
    :return - None:
    """
    canvas = render_pyramid(pyr, levels)
    plt.imshow(canvas, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blends 2 images into 1 using their laplacian pyramid and a mask and its gaussian pyramid
    :param im1- input grayscale images to be blended:
    :param im2 - input grayscale images to be blended:
    :param mask - boolean mask containing True and False representing which parts of im1 and im2 should appear in the
                  resulting im_blend:
    :param max_levels - max_levels parameter used when generating the Gaussian and Laplacian pyramids:
    :param filter_size_im -  an odd scalar that represents a squared filter used in the construction of the Laplacian
                             pyramids of im1 and im2.:
    :param filter_size_mask - an odd scalar that represents a squared filter which used in the construction of the
                              Gaussian pyramid of mask:
    :return im - the blended image:
    """
    mask = mask.astype(dtype=np.float64)
    l1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, filter_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gm, filter_vec_g = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    lout = []
    for k in range(len(l1)):
        black = np.ones(gm[k].shape)
        lk = (gm[k] * l2[k]) + (np.subtract(black, gm[k])*l1[k])
        lout.append(lk)
    im = laplacian_to_image(lout, filter_vec, [1]*len(lout))
    im = np.clip(im, 0, 1)
    return im


def blending_example1():
    """
    blends and display the blended image as an example
    """
    mask = read_image(relpath("externals/mask2.jpg"), 1)
    mask = np.round(mask)
    mask = mask.astype(np.bool)
    im1 = read_image(relpath("externals/back2.jpg"), 2)
    im2 = read_image(relpath("externals/2nd layer2.jpg"), 2)
    final_image = display(im1,im2,mask)
    return im1, im2, mask, final_image


def blending_example2():
    """
    blends and display the blended image as an example
    """
    mask = read_image(relpath("externals/mask2_final.jpg"), 1)
    mask = np.round(mask)
    mask = mask.astype(np.bool)
    im1 = read_image(relpath("externals/bridge2_final.jpg"), 2)
    im2 = read_image(relpath("externals/train2_final.jpg"), 2)
    final_image = display(im1, im2, mask)
    return im1, im2, mask, final_image


def display(im1, im2, mask):
    """
    function which displays 2 images, the mask. and the blended image
    :param im1:
    :param im2:
    :param mask:
    :return final_image - the blended image:
    """
    final_image = np.zeros(im1.shape).astype(np.float64)
    red_channel = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 10, 12, 8)
    final_image[:, :, 0] = red_channel
    green_channel = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 10, 12, 8)
    final_image[:, :, 1] = green_channel
    blue_channel = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 10, 12, 8)
    final_image[:, :, 2] = blue_channel
    plt.subplot(221)
    plt.imshow(im1, cmap='gray')
    plt.subplot(222)
    plt.imshow(im2, cmap='gray')
    plt.subplot(223)
    plt.imshow(mask, cmap='gray')
    plt.subplot(224)
    plt.imshow(final_image, cmap='gray')
    plt.show()
    return final_image


def read_image(filename, representation):
    """
    reading the image
    :param filename - path to image:
    :param representation - int:
    :return picture in grayscale or rgb according to the input
    """
    im = imread(filename)
    if representation == 1:  # If the user specified they need grayscale image,
        if len(im.shape) == 3:  # AND the image is not grayscale yet
            im = rgb2gray(im)  # convert to grayscale (**Assuming its RGB and not a different format**)
    im_float = im.astype(np.float64)  # Convert the image type to one we can work with.
    if im_float.max() > 1:  # If image values are out of bound, normalize them.
        im_float = im_float / 255
    return im_float


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)
