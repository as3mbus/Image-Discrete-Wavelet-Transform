import numpy as np
import pywt
import cv2
import sys

def w2d(img, mode='haar', level=1):
    imArray = cv2.imread(img)
    width, height, dimension= imArray.shape[:3]
    result = np.zeros((height,width,dimension), np.uint8)

    result=np.copy(imArray)

    width2 = width/2;
    for i in range(height):

        for j in range(0,width,2):

            j1 = j+1;
            j2 = j/2;

            result[i,j2] = (imArray[i,j] + imArray[i,j1])/2;
            result[i,width2+j2] = (imArray[i,j] - imArray[i,j1])/2;


    #copy array
    imArray=np.copy(result)

    # Vertical processing:
    height2 = height/2;
    for j in range(0,width):
        for i in range(0,height,2):

            i1 = i+1;
            i2 = i/2;

            result[i2,j] = (imArray[i,j] + imArray[i1,j])/2;
            result[height2+i2,j] = (imArray[i,j] - imArray[i1,j])/2;


    imArray2=np.copy(result)
    result2=np.zeros((height,width,dimension), np.uint8)

    # // Vertical processing:

    for i in range(0,height,2) :
        for j in range(0,width) :

            i1 = i+1;
            i2 = i/2;

            result2[i,j] = ((imArray2[i2,j]/2) + (imArray2[height2+i2,j]/2))*2;
            result2[i1,j] = ((imArray2[i2,j]/2) - (imArray2[height2+i2,j]/2))*2;


    # //copy array
    imArray2=np.copy(result2)

    # // Horizontal processing:
    for i in range(0,height):

        for j in range(0,width,2):

            j1 = j+1;
            j2 = j/2;
            result2[i,j] = ((imArray2[i,j2]/2) + (imArray2[i,j2+width2]/2))*2;
            result2[i,j1] =((imArray2[i,j2]/2) - (imArray2[i,j2+width2]/2))*2;



    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
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
    cv2.imshow('image',imArray_H)
    cv2.imshow('result',result)
    cv2.imshow('result2',result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

w2d("index.jpeg",'db1',1)
