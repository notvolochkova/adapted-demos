import cv2
import numpy as np
from math import ceil, floor

def resize_image(img, expected_sz):
    
    #get the current height and width of this image
    im_height, im_width = img.shape[:2] 
    
    #CV_INTER_AREA if we downscale the image, CV_INTER_CUBIC if we upscale the image
    interpolation = cv2.INTER_AREA if (im_height > expected_sz[0] or im_width > expected_sz[0]) else cv2.INTER_CUBIC
    aspect_ratio = im_width / im_height
    
    #horizontal image
    if aspect_ratio > 1:
        new_width = expected_sz[0]
        new_height = floor(new_width / aspect_ratio)
        top = floor((expected_sz[0] - new_height) / 2)
        bottom = ceil((expected_sz[0] - new_height) / 2)
        left = 0
        right = 0
        
    #vertical image
    else:
        new_height = expected_sz[0]
        new_width = floor(new_height * aspect_ratio)
        left = floor((expected_sz[0] - new_width) / 2)
        right = ceil((expected_sz[0] - new_width) / 2)
        top = 0
        bottom = 0
        
    tmp = cv2.resize(img, (new_width, new_height), interpolation)
    #pad the image to create a 640x640 with some color of choice
    tmp = cv2.copyMakeBorder(tmp, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    #flip from BGR to RGB so it's not blued
    resized_img = tmp[:,:,::-1]
    
    return resized_img