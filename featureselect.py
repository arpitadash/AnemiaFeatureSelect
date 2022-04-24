from cv2 import threshold
import numpy as np
import cv2
from itertools import chain
import matplotlib.pyplot as plt
from collections import Counter
import scipy.spatial.distance
import helpfunc as func

#Quick Algorithm
def quickAlgo(img,r_shape1,b_shape1,g_shape1,threshold):
    '''
    This is the quick algorithm that considers the pixel value mean and the high hue ratio to classify images (anemia and non anemia).
    Inputs:
    img: The RGB pixel values for the image
    r_shape1: The reshaped pixel array for the red channel (wid*ht)
    b_shape1: The reshaped pixel array for the blue channel (wid*ht)
    g_shape1: The reshaped pixel array for the green channel (wid*ht)
    Outputs:
    pvm_r1: PVM for red channel
    pvm_b1: PVM for blue channel
    pvm_g1: PVM for green channel 
    '''
    hhr1,img_hh1,hshap1 = func.HHR(img,threshold,0)
    print(hhr1)
    pvm_r1,pvm_b1,pvm_g1 = func.PVM_main(r_shape1,b_shape1,g_shape1)
    print(pvm_r1,pvm_b1,pvm_g1 )
    return hhr1, pvm_r1,pvm_b1,pvm_g1 

def robustAlgo(img):
    img_gry, h_1 = func.entropyCalc(img,printhist=0)
    brightness = func.brightness(img_gry,printbrightness=1)
    
if __name__=="__main__":
    color = ('r', 'g', 'b')
    img = func.loadAndPrintImg('./Anemia/anemia1.PNG',0)
    func.plot_colorHist(color,img,'Anemic',0)
    r_shape1,g_shape1,b_shape1 = func.colorDist(img,'Anemic',0)
    # hhr1,img_hh1,hshap1 = func.HHR(img,110)
    choiceAlgo = int(input('quick algo or robust algo'))
    threshold=110
    if choiceAlgo:
        quickAlgo(img,r_shape1,b_shape1,g_shape1,threshold)
    else
        robustAlgo(img)

