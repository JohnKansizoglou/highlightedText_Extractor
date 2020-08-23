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

from pytesseract import Output
import pytesseract
import argparse
import cv2
import matplotlib.pyplot as plt
import re, math, time
import numpy as np
from imgProc import imgProc
from skimage.filters import threshold_otsu, threshold_yen, threshold_li

import binascii
import scipy.misc
import scipy.cluster

NUM_CLUSTERS = 6

class highlightedTextAcquisition():
    '''Set of functions for extracting the Region of Interest from 
    highlighted areas
    '''

    @staticmethod
    def checkClustersColor(codes,dist_thresh=100):
        '''Check the existance of yellow in cluster centers 
        '''
        yellowFlag = False
        for i in range(len(codes)):
            if np.linalg.norm([255,255,0]-codes[i]) < dist_thresh:
                print('Yellow for cluster '+str(i))
                yellowFlag = True

        return yellowFlag

    @staticmethod
    def clusterImage(img,down_size=100,num_clusters=6):
        '''Apply per pixel clustering on a downsampled version of the input image
        based on the corresponding RGB colors and assign to each pixel the RGB
        values of its nearest cluster center
        '''
        ar_img = cv2.resize(img,(down_size,down_size))
        ar = np.asarray(ar_img)
        shape = ar.shape
        ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
        codes, _ = scipy.cluster.vq.kmeans(ar, num_clusters)
        vecs, _ = scipy.cluster.vq.vq(ar, codes)
        c = ar.copy()
        for i, code in enumerate(codes):
            c[scipy.r_[np.where(vecs==i)],:] = code
        yellowFlag = highlightedTextAcquisition.checkClustersColor(codes)

        return c.reshape(*shape).astype(np.uint8), yellowFlag

    @staticmethod
    def extractSalientMap(img,channel):
        '''Find the salient binary map in a downsampled version of the input image
        and draw contours that encompass the regions of interest
        '''
        binary = highlightedTextAcquisition.applyThreshold(img,threshold_li,channel)
        cnts, _ = cv2.findContours(cv2.bitwise_not(binary), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts)>33:
            binary = highlightedTextAcquisition.applyThreshold(img,threshold_yen,channel)
            cnts,_ = cv2.findContours(cv2.bitwise_not(binary),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        binary = binary-254
        binary,_ = highlightedTextAcquisition.convertBinaryMap(img,binary)

        return cnts, binary

    @staticmethod
    def preprocessImage(img,close_size=11):
        '''Convert image to RGB and apply closing
        '''
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pr_img = imgProc.remove_noise(imgProc.closing(rgb,(close_size,close_size)))
        bl = highlightedTextAcquisition.checkBlueChannel(pr_img)
        if bl < 25:
            channel = 1
        else:
            channel = 2
        return pr_img, channel

    @staticmethod
    def convertBinaryMap(img,binary,down_size=100,convertMapColor=True):
        '''Process the Binary Map 
        '''
        binary = np.array(binary)
        if convertMapColor and  np.sum(binary) > binary.shape[0]*binary.shape[1]/2:
                print('Converting...')
                binary = np.where(binary,0.,1.).astype('uint8')
                convertMapColor=True


        return binary, convertMapColor

    @staticmethod
    def cropImage(img, binary, cnts, e=0.01, min_region_percentage=0.001):
        '''Crop original image to patches based on the extracted contours and
        highlighted regions
        '''
        patches,idxs, maps = list(),list(),list()    
        for i,cc in enumerate(cnts):
            approx = cv2.approxPolyDP(cc, 0.04 * 2, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            try:
                new_img = cv2.cvtColor(img[y-int(e*h):y+h+int(e*h),x-2*int(e*w):x+w+2*int(e*w)],cv2.COLOR_RGB2GRAY)
                s =new_img.size/img.size
                if s >= 0.001 and s < 0.5:
                    bins = binary[y-int(e*h):y+h+int(e*h),x-2*int(e*w):x+w+2*int(e*w)]
                    patches.append(new_img)
                    maps.append(bins)
                    idxs.append(i)
            except:
                pass
        
        r_ps, m_ps = list(), list()
        for i,p,m in zip(idxs,patches,maps):
            rect = cv2.minAreaRect(cnts[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x,y = rect[0]
            angle = rect[2]
            r_p = highlightedTextAcquisition.rotateImage(p,int(round(angle)))
            m_p = imgProc.dilate(m,(int(img.shape[1]*0.02),int(img.shape[0]*0.02)))
            m_p = imgProc.dilate(highlightedTextAcquisition.rotateImage(m,int(round(angle))),(int(img.shape[1]*0.02),int(img.shape[0]*0.02)))
            r_ps.append(r_p)
            m_ps.append(m_p)

        return r_ps, m_ps

    @staticmethod
    def applyOCR(r_ps, m_ps, showMaps):
        '''Apply OCR using pytesseract
        '''
        data = []
        for idx in range(len(r_ps)):
            inv_p = highlightedTextAcquisition.invertImage(r_ps[idx])
            inv_p = np.where(m_ps[idx],inv_p,40)
            thresh = threshold_yen(inv_p)
            binary = inv_p > thresh
            binary = cv2.bitwise_not(np.array(inv_p,dtype=np.uint8))
            if showMaps:
                imgProc.imshow([binary])

            # OCR
            d = pytesseract.image_to_string(binary, lang='eng', config='--psm 6')
            data.append(d)
        
        return data

    @staticmethod
    def rotateImage(rgb,angle):
        '''Rotate text image by angle
        '''
        if angle < -45:
            angle = 90+angle
        (h, w) = rgb.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return  rotated

    @staticmethod
    def checkBlueChannel(rgb):
        '''Check the blue channel of an image
        '''
        im = rgb[:,:,2].flatten()
        return np.std(im)

    @staticmethod
    def invertImage(imagem):
        '''Calculate the inverse of an image
        '''
        imagem = np.array(255-imagem,dtype=np.uint8)
        return imagem

    @staticmethod
    def applyThreshold(img,threshold=threshold_li,channel=2):
        '''Binarize image through selected threshold
        '''
        c_2 = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
        im = c_2[:,:,channel]
        thresh = threshold(im)
        binary = im > thresh
        binary = cv2.bitwise_not(np.array(binary,dtype=np.uint8))
        binary = imgProc.closing(binary,(7,7))
        return binary

    @staticmethod
    def run(image, num_clusters=6, min_region_percentage=0.001, close_size=11, padding=0.01, convertMapColor=True, showMaps=False):
        '''Run the whole highlightedTextAcquisition pipeline
        '''
        t0 = time.time()
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pr_img, channel = highlightedTextAcquisition.preprocessImage(image,close_size=close_size)
        cnts, binary = highlightedTextAcquisition.extractSalientMap(pr_img, channel)
        r_ps, m_ps = highlightedTextAcquisition.cropImage(img,binary,cnts,min_region_percentage=min_region_percentage,e=padding)
        data = highlightedTextAcquisition.applyOCR(r_ps, m_ps, showMaps)
        print('Executed time: {:.3} sec'.format(time.time()-t0))

        return data