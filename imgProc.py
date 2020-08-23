#  =============================================
#
#  Copyright KIMO, 2020
#  All Rights Reserved
#  UNPUBLISHED, LICENSED SOFTWARE.
#
#  CONFIDENTIAL AND PROPRIETARY INFORMATION
#  WHICH IS THE PROPERTY OF KIMO.
#
#  =============================================
#
#  AUTHOR: Ioannis Kansizoglou
#
#  =============================================

import cv2
import argparse
import re, math
import numpy as np
import matplotlib.pyplot as plt

class imgProc:
    # get grayscale image
    @staticmethod
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    @staticmethod
    def remove_noise(image):
        return cv2.medianBlur(image,5)
    
    # thresholding
    @staticmethod
    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # dilation
    @staticmethod
    def dilate(image,ker=(3,3)):
        kernel = np.ones(ker,np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
        
    # erosion
    @staticmethod
    def erode(image,ker=(3,3)):
        kernel = np.ones(ker,np.uint8)
        return cv2.erode(image, kernel, iterations = 1)

    # opening - erosion followed by dilation
    @staticmethod
    def opening(image,ker=(7,7)):
        kernel = np.ones(ker,np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # closing - dilation followed by erosion
    @staticmethod
    def closing(image,ker=(7,7)):
        kernel = np.ones(ker,np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # canny edge detection
    @staticmethod
    def canny(image,a=100,b=200):
        return cv2.Canny(image, a, b)

    # skew correction
    @staticmethod
    def deskew(image):
        coords = np.column_stack(np.where(image > 0))
        print(coords.shape)
        angle = cv2.minAreaRect(coords)
        print(angle)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # template matching
    @staticmethod
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # imshow in subplots
    @staticmethod
    def imshow(imgLst, figsize=(10,10), cmap='gray', cols=2):        
        
        if len(imgLst) == 1:                
            plt.figure(figsize=figsize)
            plt.imshow(imgLst[0], cmap=cmap)        
        else:
            plt.figure(figsize=figsize)
            rows = int(np.ceil(len(imgLst) / float(cols)))        
            for i in range(len(imgLst)):
                plt.subplot(rows, cols, i+1)
                plt.imshow(imgLst[i], cmap=cmap)        
            
        plt.show()
        pass

    # improve contrast & brightness 
    @staticmethod
    def testImageQuality(img):
        imgqe = imgProc.enhanceImageQuality(
            img,
            a=2, b=-10, meanDenom=1.0,
            L_factor=1.1, H_factor=0.9,
            colorFactor=0.5
        )
        imgProc.imshow([img, imgqe], figsize=(15,15))
        return imgqe