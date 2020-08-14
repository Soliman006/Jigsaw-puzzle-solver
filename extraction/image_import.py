""" Image Import """
import cv2
import numpy as np
import os
from utils import imageResize


def retrieveExample(filename):
    """Imports one of the built-in example images."""
    absolute_path = os.path.join(os.getcwd(), 'datasets', filename)
    img = cv2.imread(absolute_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("could not find image")
    return img


def imgCapture(img_orig, settings):
    """Pre-processes an input image so it is ready for extraction."""
    # Create copies at different resolutions, one for computation and one for display
    img_comp = imageResize(img_orig, height=settings.compute_height)
    img_disp = imageResize(img_comp, height=settings.disp_height)

    # Extract shape of each image size:
    img_orig_h, img_orig_w, img_orig_channels = img_orig.shape
    img_comp_h, img_comp_w, img_comp_channels = img_comp.shape
    img_disp_h, img_disp_w, img_disp_channels = img_disp.shape

    # Create blank images for easy reference later:
    img_blank_orig = np.zeros([img_orig_h, img_orig_w, img_orig_channels], dtype=np.uint8)
    img_blank_comp = np.zeros([img_comp_h, img_comp_w, img_comp_channels], dtype=np.uint8)
    img_blank_disp = np.zeros([img_disp_h, img_disp_w, img_disp_channels], dtype=np.uint8)

    # Print out the shape of each image:
    if (settings.show_extraction_text):
        print("Original Image -", "width:", img_orig_w, "height:",
              img_orig_h, "channels:", img_orig_channels)
        print("Compute Image  -", "width:", img_comp_w, "height:",
              img_comp_h, "channels:", img_comp_channels)
        print("Display Image  -", "width:", img_disp_w, " height:",
              img_disp_h, " channels:", img_disp_channels)

    return img_orig, img_comp, img_disp, img_blank_orig, img_blank_comp, img_blank_disp
