import math
import glob

import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import convolve2d

HOUGH_SPACE_SIZE = 512
LINE_MIN_LENGTH = 64
NON_MAX_SUPPRESSION_DIST = 5

# TODO: pick the required threshold
LINE_EDGE_THRESHOLD = 500


def imshow_with_colorbar(index, img):
    plt.figure(index)
    ax = plt.subplot(111)
    im = ax.imshow(img)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def convolve(img, kernel):
    # TODO:     done
    # write convolve operation
    # (the output array may have negative values)
    # for result comparison:
    # return convolve2d(img, kernel, mode='valid')
    (iH, iW) = img.shape[:2]
    (kH, kernel_width) = kernel.shape[:2]
    pad = (kernel_width - 1) // 2
    image_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                      cv2.BORDER_REPLICATE)
    image_padded[pad:-pad, pad:-pad] = img
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image_padded[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # perform the convolution by
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum(axis=(0, 1))

            output[y - pad, x - pad] = k

    return output


def threshold(img, threshold_value):
    # TODO:
    # write threshold operation
    # funciton should return numpy array with 1 in pixels that are higher or equal to threshold, otherwise - zero
    return img > threshold_value


def draw_line_in_hough_space(h_space, y, x):
    # TODO:
    # I propose to use polar coordinates instead of proposed y = a*x + b.
    # They are way easier for implementation.
    # The idea is the same. Short documentation: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    for angle_index in range(HOUGH_SPACE_SIZE):
        angle = angle_index / HOUGH_SPACE_SIZE * (2 * math.pi)
        r_index = x * np.cos(angle) + y * np.sin(angle)
        r_index = int(r_index)
        if r_index > 0:
            h_space[r_index, angle_index] += 1


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D


def count_lines(img, is_debug):
    if is_debug: imshow_with_colorbar(1, img)

    # TODO: done
    # pick the kernel for convolve operation, you need to find edges
    # sobel kernel
    k = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # calculate convolve operation

    new_image_x = convolve(img, k)
    new_image_y = convolve(img, k.T)

    img_c = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    img_c *= 255.0 / img_c.max()
    gaussian_k = (gaussian_kernel(3))

    img_c = convolve(img_c, gaussian_k)
    if is_debug: imshow_with_colorbar(2, img_c)

    # TODO:
    # apply thresholding (the result of the threshold should be array with zeros and ones)
    img_thr = np.array(threshold(img_c, LINE_EDGE_THRESHOLD)).astype(int)  # LINE_EDGE_THRESHOLD

    if is_debug: imshow_with_colorbar(3, img_thr)
    #
    h_space = np.zeros((HOUGH_SPACE_SIZE, HOUGH_SPACE_SIZE), dtype=np.int)

    # for each coordinate ...
    for y in range(img_thr.shape[0]):
        for x in range(img_thr.shape[1]):
            # if there is edge ...
            if img_thr[y, x] != 0:
                draw_line_in_hough_space(h_space, y, x)
    if is_debug: imshow_with_colorbar(4, h_space)
    #
    # apply threshold for hough space.
    # TODO:
    # pick the threshold to cut-off smaller lines
    # h_space_mask = threshold(h_space, ...)
    h_space_mask = 1 * (h_space > 50)
    if is_debug: imshow_with_colorbar(5, h_space_mask)

    # get indices of non-zero elements
    y_arr, x_arr = np.nonzero(h_space_mask)

    # calculate the lines number,
    # here is simple non maximum suppresion algorithm.
    # it counts only one point in the distance NON_MAX_SUPPRESSION_DIST
    lines_num = 0
    for i in range(len(y_arr)):
        has_neighbour = False
        for j in range(i):
            yi, xi = y_arr[i], x_arr[i]
            yj, xj = y_arr[j], x_arr[j]
            dy = abs(yi - yj)
            dx = abs(xi - xj)
            if dy <= NON_MAX_SUPPRESSION_DIST and \
                    (dx <= NON_MAX_SUPPRESSION_DIST or \
                     dx >= HOUGH_SPACE_SIZE - NON_MAX_SUPPRESSION_DIST):  # if x axis represents the angle, than check the distance if the points points that are near 0 degree (for example, distance between 1 deg. and 359 deg. is 2 deg.)
                has_neighbour = True
                break

        if not has_neighbour:
            lines_num += 1

    if is_debug: print('lines number: %d' % lines_num); plt.show()
    return lines_num


if __name__ == '__main__':
    fpath_arr = glob.glob('./images/hw1/*.png')
    fpath_arr.sort()

    print(fpath_arr)

    for img_fpath in fpath_arr:
        img = cv2.imread(img_fpath, cv2.IMREAD_GRAYSCALE)
        print('Number of lines for %s: %d' % (img_fpath, count_lines(img, False)))
