import numpy as np
from pepeline import screentone, cvt_color,CvtType

# Function to apply halftone effect on RGB image
def rgb_halftone(img: np.ndarray, dot_size: int, angle: list = [0, 0, 0]) -> np.ndarray:
    # Create a copy of the input RGB image
    rgb_img = img.copy()
    # Apply halftone effect to each channel of the RGB image separately
    rgb_img[:, :, 0] = screentone(rgb_img[:, :, 0], dot_size, angle[0])
    rgb_img[:, :, 1] = screentone(rgb_img[:, :, 1], dot_size, angle[1])
    rgb_img[:, :, 2] = screentone(rgb_img[:, :, 2], dot_size, angle[2])
    return rgb_img

# Function to apply CMYK halftone effect on RGB  image
def cmyk_halftone(img: np.ndarray, dot_size: int, angle: list = [0, 0, 0, 0]) -> np.ndarray:
    # Convert RGB image to CMYK
    cmyk_img = cvt_color(img,CvtType.RGB2CMYK)
    # Apply halftone effect to each channel of the CMYK image separately
    cmyk_img[:, :, 0] = screentone(cmyk_img[:, :, 0], dot_size, angle[0])
    cmyk_img[:, :, 1] = screentone(cmyk_img[:, :, 1], dot_size, angle[1])
    cmyk_img[:, :, 2] = screentone(cmyk_img[:, :, 2], dot_size, angle[2])
    cmyk_img[:, :, 3] = screentone(cmyk_img[:, :, 3], dot_size, angle[3])
    # Convert halftoned CMYK image back to RGB
    rgb_img = cvt_color(cmyk_img,CvtType.CMYK2RGB)
    return rgb_img

# Function to apply halftone effect on grayscale image
def gray_halftone(img: np.ndarray, dot_size: int) -> np.ndarray:
    # Apply halftone effect to the grayscale image
    return screentone(img, dot_size)
