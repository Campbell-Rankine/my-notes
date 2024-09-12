# import packages
import cv2 as cv
import numpy as np
from typing import Optional, List
from timeit import default_timer as timer
from numba import cuda
import math

# logging init
import logging


def pad_image(img: np.ndarray, kernel: list[int]) -> np.ndarray:
    # Apply padding to the image
    p_h = int((kernel[0] - 1) / 2)
    p_w = int((kernel[1] - 1) / 2)
    padded_image = np.zeros((img.shape[0] + (2 * p_h), img.shape[1] + (2 * p_w)))
    padded_image[
        p_h : padded_image.shape[0] - p_h, p_w : padded_image.shape[1] - p_w
    ] = img
    return padded_image


def get_image_xgrad(
    outputH: np.ndarray, img: np.ndarray, padded_image: np.ndarray, kernel: list[int]
):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                outputH[i][j] = abs(
                    np.sum(
                        np.array([1, 0, -1]).T
                        @ (
                            np.array([1, 2, 1])
                            @ padded_image[i : i + kernel[0], j : j + kernel[1]]
                        )
                    )
                )  # Decomposability of sobel filters
            except ValueError:  ###was useful in border handling
                print(i, j)
    return outputH


def get_image_ygrad(
    outputV: np.ndarray, img: np.ndarray, padded_image: np.ndarray, kernel: list[int]
):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                outputV[i][j] = abs(
                    np.sum(
                        np.array([1, 2, 1]).T
                        @ (
                            np.array([-1, 0, 1])
                            @ padded_image[i : i + kernel[0], j : j + kernel[1]]
                        )
                    )
                )  # Decomposability of sobel filters
            except ValueError:  ###was useful in border handling
                print(i, j)
    return outputV


def sobel_filter_cpu(
    img: np.ndarray, kernel: Optional[List[int]] = [16, 16]
) -> np.ndarray:
    outputH = np.zeros(img.shape)
    outputV = np.zeros(
        img.shape
    )  # We have 3 outputs, this function will display histograms for Horizontal and vertical
    output = np.zeros(img.shape)

    ###Smooth images###
    img = cv.GaussianBlur(
        img, (16, 16), np.std(img) / 4
    )  # The amount of gaussian blur can directly determine the sharpness of the sobel filter edge

    ###Unsure of if this is useful for edge detection but since we apply the same convolution algorithm:###
    ###Convert to padded image###
    padded_image = pad_image(img, kernel)

    # Convolve with filters Horizontal
    outputH = get_image_xgrad(outputH, img, padded_image, kernel)
    outputV = get_image_ygrad(outputV, img, padded_image, kernel)

    assert outputH.shape == img.shape and outputV.shape == img.shape  # quality of life

    ###Normalize###
    outputHN = (outputH - np.min(outputH)) / (np.max(outputH) - np.min(outputH))
    outputVN = (outputV - np.min(outputV)) / (np.max(outputV) - np.min(outputV))

    print("Horizontal Range: " + str((np.min(outputHN), np.max(outputHN))))
    print("Vertical Range: " + str((np.min(outputVN), np.max(outputVN))))

    ###Merge horizontal and vertical###
    output = np.sqrt(outputH**2 + outputV**2)

    outputN = (output - np.min(output)) / (np.max(output) - np.min(output))

    return [outputHN * 255, outputVN * 255, outputN * 255]
