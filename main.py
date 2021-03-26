import cv2 as cv
import numpy as np
import rawpy
import imageio


def func_wb(channel, percent):
    mi, ma = (np.percentile(channel, percent), np.percentile(channel, 100.0 - percent))
    channel = np.uint8(np.clip((channel - mi) * 255.0 / (ma - mi), 0, 255))
    return channel


def white_balancing_cv(image):
    balancer = cv.xphoto.createSimpleWB()
    # balancer.setSaturationThreshold(0.05)
    image_wb = balancer.balanceWhite(image)
    # cv.imshow('image after wb_cv', image_wb)
    return image_wb


def white_balancing_gimp(image):
    image_wb = np.dstack([func_wb(channel, 0.05) for channel in cv.split(image)])
    cv.imshow('image after wb_gimp', image_wb)
    return image_wb


def white_balancing_manual(image, wb_matrix):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                image[i][j][k] *= wb_matrix[k]/wb_matrix[1]


def demosaic_downgrade_4x(image_array):
    R = image_array[0::2, 0::2]
    G = np.clip(image_array[0::2, 1::2] // 2 + image_array[1::2, 0::2] // 2, 0, 2 ** 16 - 1)
    B = image_array[1::2, 1::2]
    downgrade_result = np.dstack((R, G, B))
    return downgrade_result


path = 'example.cr2'
with rawpy.imread(path) as raw:
    print(f'raw type:                     {raw.raw_type}')
    print(f'number of colors:             {raw.num_colors}')
    print(f'color description:            {raw.color_desc}')
    print(f'raw pattern:                  {raw.raw_pattern.tolist()}')
    print(f'white level:                  {raw.white_level}')
    print(f'color matrix:                 {raw.color_matrix.tolist()}')
    print(f'XYZ to RGB conversion matrix: {raw.rgb_xyz_matrix.tolist()}')
    print(f'camera white balance:         {raw.camera_whitebalance}')
    print(f'daylight white balance:       {raw.daylight_whitebalance}')
    print(f'tone-curve:       {raw.tone_curve}')
    raw_ar = raw.raw_image_visible
    # rgb = raw.postprocess()
    int14_max = 2 ** 14 - 1
    black_level = np.amin(raw_ar)
    raw_ar -= black_level
    raw_ar *= int(int14_max / (raw.white_level - black_level))
    raw_ar = np.clip(raw_ar, 0, int14_max)

    raw_ar *= 2 ** 2
    image_demosaic = demosaic_downgrade_4x(raw_ar)
    print(image_demosaic.shape)

    camera_wb_red = raw.camera_whitebalance[0] / raw.camera_whitebalance[1]
    camera_wb_blue = raw.camera_whitebalance[2] / raw.camera_whitebalance[1]
    # print(camera_wb_red, camera_wb_blue)
    # print(camera_wb_red1, camera_wb_blue1)
    image_demosaic[:, :, 0] = image_demosaic[:, :, 0] * camera_wb_red
    image_demosaic[:, :, 2] = image_demosaic[:, :, 2] * camera_wb_blue
    image_demosaic = np.clip(image_demosaic, 0, int14_max)
    # white_balancing_manual(image_demosaic, raw.daylight_whitebalance)
    # print(image_demosaic)

    matrix_sRGB = raw.rgb_xyz_matrix[0:3, 0:3]
    matrix_sRGB = np.round(matrix_sRGB * 255)
    image_demosaic = image_demosaic // 2 ** 8
    shape = image_demosaic.shape
    # print(shape)
    pixel_image = image_demosaic.reshape((-1, 3)).T
    pixel_image = np.dot(matrix_sRGB, pixel_image) // 255
    image = pixel_image.T.reshape(shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    gamma_curve = [(i / 255) ** (1 / 2.2) * 255 for i in range(256)]
    gamma_curve = np.array(gamma_curve, dtype=np.uint8)
    image_final = gamma_curve[image]

imageio.imsave('final_srgb.png', image_final)
# imageio.imsave('normal.png', rgb)
