import numpy as np
import pywt
import cv2
import sys
import math

def w2d(img, mode='haar', level=1):

    #loadImage & copy image
    imArray = cv2.imread(img)
    width, height, dimension= imArray.shape[:3]
    result = np.zeros((height,width,dimension), np.uint8)

    imArray2=waveleteTransform(imArray,width,height)

    imArray3=np.copy(imArray2)
    result2=np.zeros((height/2,width/2,dimension), np.uint8)

    result4=inverseWaveleteTransform(imArray3,height/2,width/2)

    #Datatype conversions
    #convert to grayscale
    # imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    # imArray =  np.float32(imArray)
    # imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    # imArray_H*= 255;
    imArray_H =  np.uint8(imArray_H)
    #Display result
    cv2.imshow('DWT ?',imArray_H)
    cv2.imshow('base Image',imArray)
    cv2.imshow('DWT',imArray2)
    cv2.imshow('imarray3',imArray3)
    cv2.imshow('result4',result4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def waveleteTransform(image,width,height):
    result=np.copy(image)

    #Horizontal processing
    width2 = width/2;
    for i in range(height):

        for j in range(0,width,2):

            j1 = j+1;
            j2 = j/2;

            result[i,j2] = (image[i,j] + image[i,j1])/2;
            result[i,width2+j2] = (image[i,j] - image[i,j1])/2;


    #copy array
    image=np.copy(result)

    # Vertical processing:
    height2 = height/2;
    for i in range(0,height,2):
        for j in range(0,width):

            i1 = i+1;
            i2 = i/2;

            result[i2,j] = (image[i,j] + image[i1,j])/2;
            result[height2+i2,j] = (image[i,j] - image[i1,j])/2;

    return result


def inverseWaveleteTransform(image,nc,nr):
    result=np.zeros((nr,nc,3), np.uint8)
    nr2 = nr/2;

    for i in range(0,nr-1,2):
        for j in range(0,nc):

            i1 = i+1
            i2 = i/2

            result[i,j] += ((image[i2,j]/2) + (image[nr2+i2,j]/2))*2;
            result[i1,j] += ((image[i2,j]/2) - (image[nr2+i2,j]/2))*2;

    # //copy array
    image=np.copy(result)

    # // Horizontal processing:
    nc2 = nc/2;
    for i in range(0,nr) :
        for j in range(0,nc-1,2):

            j1 = j+1;
            j2 = j/2;
            result[i,j] = ((image[i,j2]/2) + (image[i,j2+nc2]/2))*2;
            result[i,j1] =((image[i,j2]/2) - (image[i,j2+nc2]/2))*2;
    #

    return result;



w2d(sys.argv[1],'db1',1)
