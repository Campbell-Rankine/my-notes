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
    img: np.ndarray, kernel: Optional[List[int]] = [3, 3]
) -> np.ndarray:
    outputH = np.zeros(img.shape)
    outputV = np.zeros(
        img.shape
    )  # We have 3 outputs, this function will display histograms for Horizontal and vertical
    output = np.zeros(img.shape)

    ###Smooth images###
    img = cv.GaussianBlur(
        img, (7, 7), np.std(img) / 4
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


@cuda.jit
def sobel_filter(input_image, output_image):
    # Apply sobel filter inplace on the output image.
    x, y = cuda.grid(2)

    if x < input_image.shape[0] - 2 and y < input_image.shape[1] - 2:
        # get Gx
        Gx = (
            input_image[x, y]
            - input_image[x + 2, y]
            + 2 * input_image[x, y + 1]
            - 2 * input_image[x + 2, y + 1]
            + input_image[x, y + 2]
            - input_image[x + 2, y + 2]
        )
        # get Gy
        Gy = (
            input_image[x, y]
            - input_image[x, y + 2]
            + 2 * input_image[x + 1, y]
            - 2 * input_image[x + 1, y + 2]
            + input_image[x + 2, y]
            - input_image[x + 2, y + 2]
        )

        # in place op
        output_image[x + 1, y + 1] = math.sqrt(Gx**2 + Gy**2)


def cuda_sobel(np_image: np.ndarray):
    np_image = cv.GaussianBlur(
        np_image, (7, 7), np.std(np_image) / 4
    )  # Remove image noise

    # alloc
    cuda_im = cuda.to_device(np_image)
    output_image = np.zeros_like(np_image)
    threads_per_block = (16, 16)

    # calculate dims
    blockspergrid_x = (
        np_image.shape[0] + threads_per_block[0] - 1
    ) // threads_per_block[0]
    blockspergrid_y = (
        np_image.shape[1] + threads_per_block[1] - 1
    ) // threads_per_block[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # apply sobel filter
    sobel_filter[blockspergrid, threads_per_block](cuda_im, output_image)
    output_image = cuda_im.copy_to_host()
    return output_image


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.info("Starting CPU Sobel Filter calc")

    path_to_ims = "../images"

    start = timer()
    # CPU block
    im1 = cv.imread(f"{path_to_ims}/image1.jpg")
    # convert to grayscale
    gim1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2 = np.zeros_like(im1)
    im2[:, :, 0] = (
        gim1  # To display the image correctly you'll need to send the grey channel to each of the three image channels
    )
    im2[:, :, 1] = gim1
    im2[:, :, 2] = gim1
    [horizontal, vertical, neutral] = sobel_filter_cpu(gim1)

    end = timer()
    logging.info(f"CPU Sobel Filter calculation took: {round(end-start, 2)}s")

    # cuda
    logging.info("Starting CUDA Sobel Filter calc")

    start = timer()
    output = cuda_sobel(gim1)
    end = timer()
    logging.info(f"CUDA Sobel Filter calculation took: {round(end-start, 2)}s")
