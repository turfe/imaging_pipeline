import cv2 as cv
import numpy as np
import rawpy
import imageio
import matplotlib.pyplot as plot
import exifread
import math
from PIL import Image, ImageFilter


def device_wb(image_array, matrix_wb):
    white_balance = np.zeros((2, 2), dtype=np.double)
    white_balance[1][0] = matrix_wb[0] / matrix_wb[1]
    white_balance[0][0] = white_balance[1][1] = 1
    white_balance[0][1] = matrix_wb[2] / matrix_wb[1]
    white_balance = np.tile(white_balance, (image_array.shape[0] // 2, image_array.shape[1] // 2))
    image_result = np.clip(image_array * white_balance, 0, 1)
    return image_result


def device_wb_demosaic(image_array, matrix_wb):
    image_array[:, :, 0] *= (matrix_wb[0] / matrix_wb[1])
    image_array[:, :, 2] *= (matrix_wb[2] / matrix_wb[1])
    return image_array


def awb_gray_world(image_array):
    R_mean = np.mean(image_array[:, :, 0])
    G_mean = np.mean(image_array[:, :, 1])
    B_mean = np.mean(image_array[:, :, 2])
    image_array[:, :, 0] *= (G_mean / R_mean)
    image_array[:, :, 2] *= (G_mean / B_mean)
    return image_array


def awb_white_patch(image_array, raw_sensor):
    R_max = np.amax(raw_sensor[1::2, 0::2])
    G_max = max(np.amax(raw_sensor[0::2, 0::2]), np.amax(raw_sensor[1::2, 1::2]))
    B_max = np.amax(raw_sensor[0::2, 1::2])
    print(R_max, G_max, B_max)
    image_array[:, :, 0] *= (G_max / R_max)
    image_array[:, :, 2] *= (G_max / B_max)
    return image_array


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


def demosaic_downsampling(image_array):
    image_result = np.empty((image_array.shape[0] // 2, image_array.shape[1] // 2, 3), dtype=np.double)
    image_result[:, :, 0] = image_array[1::2, 0::2]
    image_result[:, :, 1] = (image_array[0::2, 0::2] + image_array[1::2, 1::2]) / 2
    image_result[:, :, 2] = image_array[0::2, 1::2]
    return image_result


def edge_interpolation(image_array, image_output, i, j, threshold, color):
    if color == 1:
        if abs(image_array[i - 1][j] - image_array[i + 1][j]) < threshold and \
                abs(image_array[i][j - 1] - image_array[i + 1][j]) < threshold:
            return (image_array[i - 1][j] + image_array[i + 1][j] +
                    image_array[i][j - 1] + image_array[i][j + 1]) / 4
        elif abs(image_array[i - 1][j] - image_array[i + 1][j]) < threshold:
            return (image_array[i][j - 1] + image_array[i][j + 1]) / 2
        else:
            return (image_array[i - 1][j] + image_array[i + 1][j]) / 2
    else:
        if abs(image_array[i - 1][j - 1] - image_array[i + 1][j + 1]) < threshold and \
                abs(image_array[i + 1][j - 1] - image_array[i - 1][j + 1]) < threshold:
            return (image_array[i - 1][j - 1] + image_array[i + 1][j + 1] +
                    image_array[i + 1][j - 1] + image_array[i - 11][j + 1]) / 4
        elif abs(image_array[i - 1][j - 1] - image_array[i + 1][j + 1]) < threshold:
            return (image_array[i + 1][j - 1] + image_array[i - 1][j + 1]) / 2
        else:
            return (image_array[i - 1][j - 1] + image_array[i + 1][j + 1]) / 2


def demosaic(image_array):
    image_result = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.double)
    map_filter = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
    map_filter[1::2, 0::2] = 0  # Red
    map_filter[0::2, 1::2] = 2  # Blue
    map_filter[0::2, 0::2] = map_filter[1::2, 1::2] = 1  # Green

    image_result[0::2, 0::2, 1] = image_array[0::2, 0::2]
    image_result[1::2, 1::2, 1] = image_array[1::2, 1::2]
    image_result[0::2, 1::2, 2] = image_array[0::2, 1::2]
    image_result[1::2, 0::2, 0] = image_array[1::2, 0::2]

    for i in range(1, image_result.shape[0] - 1):
        for j in range(1, image_result.shape[1] - 1):
            if map_filter[i][j] == 0:
                image_result[i][j][1] = edge_interpolation(image_array, image_result, i, j, 0.5, 1)
                image_result[i][j][2] = edge_interpolation(image_array, image_result, i, j, 0.5, 2)
            elif map_filter[i][j] == 2:
                image_result[i][j][1] = edge_interpolation(image_array, image_result, i, j, 0.5, 1)
                image_result[i][j][0] = edge_interpolation(image_array, image_result, i, j, 0.5, 0)
            else:
                if map_filter[i - 1][j] == 2:
                    image_result[i][j][0] = (image_array[i][j - 1] + image_array[i][j + 1]) / 2
                    image_result[i][j][2] = (image_array[i - 1][j] + image_array[i + 1][j]) / 2
                else:
                    image_result[i][j][2] = (image_array[i][j - 1] + image_array[i][j + 1]) / 2
                    image_result[i][j][0] = (image_array[i - 1][j] + image_array[i + 1][j]) / 2

    return image_result


def transform_to(image_array, transform_matrix):
    image_result = np.empty((image_array.shape[0], image_array.shape[1], image_array.shape[2]), dtype=np.double)
    image_result[:, :, 0] = image_array[:, :, 0] * transform_matrix[0][0] + image_array[:, :, 1] * transform_matrix[0][
        1] + \
                            image_array[:, :, 2] * transform_matrix[0][2]
    image_result[:, :, 1] = image_array[:, :, 0] * transform_matrix[1][0] + image_array[:, :, 1] * transform_matrix[1][
        1] + \
                            image_array[:, :, 2] * transform_matrix[1][2]
    image_result[:, :, 2] = image_array[:, :, 0] * transform_matrix[2][0] + image_array[:, :, 1] * transform_matrix[2][
        1] + \
                            image_array[:, :, 2] * transform_matrix[2][2]
    return image_result


path = 'example.CR2'

with open(path, 'rb') as raw_file:
    tags = exifread.process_file(raw_file)
    f_number = convert_to_float(str(dict(tags.items()).get('EXIF FNumber')))
    exp_time = convert_to_float(str(dict(tags.items()).get('EXIF ExposureTime')))
    ISO = str(dict(tags.items()).get('EXIF ISOSpeedRatings'))
    print(f'ISO:                          {ISO}')
    print(f'f-number:                     {f_number}')
    print(f'exposure time:                {exp_time}')
    exp_value = math.log((math.pow(f_number, 2) / exp_time), 2)
    print(f'exposure value:               {exp_value}')

with rawpy.imread(path) as raw:
    postprocess_rawpy = raw.postprocess()
    print(f'number of colors:             {raw.num_colors}')
    print(f'raw patern:                   {raw.raw_pattern.tolist()}')
    print(f'color description:            {raw.color_desc}')
    print(f'white level:                  {raw.white_level}')
    print('XYZ to RGB conversion matrix:')
    print(raw.rgb_xyz_matrix)
    print(f'device black level:           {raw.black_level_per_channel}')
    print(f'device white balance:         {raw.camera_whitebalance}')

    raw_image = raw.raw_image_visible
    black_level = np.min(raw.black_level_per_channel)
    raw_image -= black_level
    raw_image = raw_image / (raw.white_level - black_level)

    image_demosaic = demosaic(raw_image)

    image_dm = Image.fromarray((image_demosaic * 255).astype(np.uint8))
    pil_image_blur = image_dm.filter(ImageFilter.GaussianBlur(radius=1))
    image_blur = np.array(pil_image_blur) / 255
    image_diff = image_demosaic - image_blur
    image_nr = np.where(abs(image_diff) > 0.01, image_demosaic, image_blur)

    image_wb = device_wb_demosaic(image_nr, raw.camera_whitebalance)
    # image_wb = awb_gray_world(image_nr)
    # image_wb = awb_white_patch(image_nr, raw.raw_image_visible)

    image_nr = image_nr * (2 ** (exp_value / 10))

    matrix_XYZ_to_device = np.array(raw.rgb_xyz_matrix[0:3, 0:3], dtype=np.double)
    matrix_sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                                   [0.2126729, 0.7151522, 0.0721750],
                                   [0.0193339, 0.1191920, 0.9503041]], dtype=np.double)
    matrix_sRGB_to_device = np.dot(matrix_XYZ_to_device, matrix_sRGB_to_XYZ)
    normalization = np.tile(np.sum(matrix_sRGB_to_device, 1), (3, 1)).transpose()
    matrix_sRGB_to_device = matrix_sRGB_to_device / normalization
    matrix_device_to_sRGB = np.linalg.inv(matrix_sRGB_to_device)
    image_sRGB = transform_to(image_nr, matrix_device_to_sRGB)

    image_sRGB = np.clip(image_sRGB, 0, 1)

    image_sRGB = image_sRGB ** (1 / 2.2)
    image_final = np.clip(image_sRGB, 0, 1)

    plot.axis('off')
    plot.imshow(image_final)
    plot.show()

imageio.imsave('final_srgb.png', image_final)
imageio.imsave('postprocess_rawpy.png', postprocess_rawpy)
