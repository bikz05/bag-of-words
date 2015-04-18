#!/usr/local/bin/python2.7

import cv2, os
import matplotlib.pyplot as plt 

def imlist(path):
    """
    The function imlist returns all the names of the files in 
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

def imshow(im_title, im):
    ''' This is function to display the image'''
    plt.figure()  
    plt.title(im_title)
    plt.axis("off")
    if len(im.shape) == 2:
        plt.imshow(im, cmap = "gray")
    else:
        im_display = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        plt.imshow(im_display)
    plt.show()

def imreads(path):
    '''
    This reads all the images in a given folder and returns the results 
    '''
    images_path = imlist("/home/bikz05/Desktop/back_projection")
    images = []
    for image_path in images_path:
        images.append(cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR))
    return images

def show(image, name="Image"):
    '''
    Routine to display the image.
    '''
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
