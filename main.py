import cv2 as cv
import numpy as np
import rawpy
import imageio
import matplotlib.pyplot as plot

def white_balance(image_array, matrix_wb):
    white_balance = np.zeros((2, 2), dtype=np.double)
    white_balance[0][0] = matrix_wb[0] / matrix_wb[1]
    white_balance[0][1] = white_balance[1][0] = matrix_wb[1] / matrix_wb[1]
    white_balance[1][1] = matrix_wb[2] / matrix_wb[1]
    white_balance = np.tile(white_balance, (image_array.shape[0] // 2, image_array.shape[1] // 2))
    image_result = np.clip(image_array * white_balance, 0, 1)
    return image_result


def demosaic_downsampling(image_array):
    image_result = np.empty((image_array.shape[0] // 2, image_array.shape[1] // 2, 3), dtype=np.double)
    image_result[:, :, 0] = image_array[0::2, 0::2]
    image_result[:, :, 1] = (image_array[0::2, 1::2] + image_array[1::2, 0::2]) / 2
    image_result[:, :, 2] = image_array[1::2, 1::2]
    return image_result


def transfrom_to_sRGB(image_array, matrix_to_sRGB):
    image_result = np.empty((image_array.shape[0], image_array.shape[1], image_array.shape[2]), dtype=np.double)
    image_result[:, :, 0] = image_array[:, :, 0] * matrix_to_sRGB[0][0] + image_array[:, :, 1] * matrix_to_sRGB[0][1] + \
                            image_array[:, :, 2] * matrix_to_sRGB[0][2]
    image_result[:, :, 1] = image_array[:, :, 0] * matrix_to_sRGB[1][0] + image_array[:, :, 1] * matrix_to_sRGB[1][1] + \
                            image_array[:, :, 2] * matrix_to_sRGB[1][2]
    image_result[:, :, 2] = image_array[:, :, 0] * matrix_to_sRGB[2][0] + image_array[:, :, 1] * matrix_to_sRGB[2][1] + \
                            image_array[:, :, 2] * matrix_to_sRGB[2][2]
    return image_result


path = 'example.CR2'
with rawpy.imread(path) as raw:
    postprocess_rawpy = raw.postprocess()
    print(f'number of colors:             {raw.num_colors}')
    print(f'color description:            {raw.color_desc}')
    print(f'white level:                  {raw.white_level}')
    print('XYZ to RGB conversion matrix:')
    print(raw.rgb_xyz_matrix)
    print(f'camera white balance:         {raw.camera_whitebalance}')

    raw_image = raw.raw_image_visible
    black_level = np.min(raw_image)
    raw_image -= black_level
    raw_image = raw_image / (raw.white_level - black_level)

    device_wb = raw.camera_whitebalance
    image_wb = white_balance(raw_image, device_wb)

    image_demosaic = demosaic_downsampling(image_wb)

    matrix_XYZ_to_device = np.array(raw.rgb_xyz_matrix[0:3, 0:3], dtype=np.double)
    matrix_sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                                   [0.2126729, 0.7151522, 0.0721750],
                                   [0.0193339, 0.1191920, 0.9503041]], dtype=np.double)
    matrix_sRGB_to_device = np.dot(matrix_XYZ_to_device, matrix_sRGB_to_XYZ)
    normalization = np.tile(np.sum(matrix_sRGB_to_device, 1), (3, 1)).transpose()
    matrix_sRGB_to_device = matrix_sRGB_to_device / normalization
    matrix_device_to_sRGB = np.linalg.inv(matrix_sRGB_to_device)

    image_sRGB = transfrom_to_sRGB(image_demosaic, matrix_device_to_sRGB)

    image_sRGB = np.clip(image_sRGB, 0, 1)
    image_sRGB = image_sRGB ** (1 / 2.2)
    image_final = np.clip(image_sRGB, 0, 1)

    plot.axis('off')
    plot.imshow(image_final)
    plot.show()

imageio.imsave('final_srgb.png', image_final)
imageio.imsave('postprocess_rawpy.png', postprocess_rawpy)
