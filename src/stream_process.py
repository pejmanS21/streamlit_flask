import numpy as np
import cv2
import os
from PIL import Image, ImageOps

reference_shape = (256, 256, 1)


"""
    Image that showed in sidebar menu of streamlit
"""
def stream_data(file, pre_process="Original", dim=256):
    """
        :file:  path that pointed to the uploaded image in streamlit
        :pre_process:   Original or DHE for image
        :dim:  size for image
    """
    image = Image.open(file)
    image = ImageOps.fit(image, (dim, dim))
    image = np.asarray(image)
    # pil to cv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # check channel
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compatible size
        if image.shape != reference_shape:
            image = cv2.resize(image, (dim, dim))

    # selected pre_process
    if pre_process == "DHE":
        image = cv2.equalizeHist(image)

    # reshape & normalize
    image = image.reshape(1, dim, dim, 1)
    image = image / 255.
    return image

"""
    process for image that will be sent to backend
"""
def load_data(file, pre_process="Original", dim=256):
    """
        :file:  path that pointed to the uploaded image in streamlit
        :pre_process:   Original or DHE for image
        :dim:  size for image
    """
    image = Image.open(file)
    image = ImageOps.fit(image, (dim, dim))
    image = np.asarray(image)
    # pil to cv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # check channel
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compatible size
        if image.shape != reference_shape:
            image = cv2.resize(image, (dim, dim))

    # selected pre_process
    if pre_process == "DHE":
        image = cv2.equalizeHist(image)

    return image