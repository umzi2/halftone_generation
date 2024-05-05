import os

import numpy as np
from pepeline import read, save

from src.add_halftone import rgb_halftone, cmyk_halftone, gray_halftone

input_folder = "./test/INPUT"
cmyk_angle = [0, 15, 30, 45]
rgb_angle = [0, 15, 30]
dot_size = 7
img_list = os.listdir(input_folder)
for img_name in img_list:
    img_folder = os.path.join(input_folder, img_name)
    img = read(img_folder, mode=1, format=0)

    rgb_halftone_img = rgb_halftone(img, 7, rgb_angle)
    out_rgb_folder = os.path.join("./test/OUTPUT/rgb", img_name)
    save(rgb_halftone_img, out_rgb_folder)

    cmyk_halftone_img = cmyk_halftone(img, 7, cmyk_angle)
    out_cmyk_folder = os.path.join("./test/OUTPUT/cmyk", img_name)
    save(cmyk_halftone_img, out_cmyk_folder)

    img_gray = np.dot(img[..., :3], [0.114, 0.587, 0.299]).astype(np.float32)
    gray_halftone_img = gray_halftone(img_gray, 7)
    out_gray_folder = os.path.join("./test/OUTPUT/gray", img_name)
    save(gray_halftone_img, out_gray_folder)
