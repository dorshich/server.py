import cv2 as cv
import os
from functools import reduce


def red(x, y):
    r = (x + y + abs(x - y)) / 2
    return r


def combine_images(images_path):
    images_arr = [cv.imread(os.path.join(images_path, image), 0) for image in os.listdir(images_path) if
                  image.endswith('bmp')]
    blended = reduce(red, images_arr)
    cv.imwrite(rf'C:\Data\blended_images\{os.path.basename(images_path)}.bmp', blended)
    print(os.path.basename(images_path))


if __name__ == "__main__":
    images_paths = r'C:\Data\Sheba'
    for images_path in os.listdir(images_paths):
        if images_path.isnumeric():
            imgs_path = os.path.join(images_paths, images_path)
            combine_images(imgs_path)
