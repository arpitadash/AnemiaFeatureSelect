
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from math import log
import cv2
from itertools import chain
import matplotlib.pyplot as plt
from collections import Counter
import scipy.spatial.distance

def plot_colorHist(color,img,title='',printimg=1):
  """
  Plots the histogram of the color mentioned
  Inputs: 
    color: The color channel (Red/Green/Blue)
    img : The image variable
    title (Optional variables): title of the histogram plot
    printimg: 0 (don't plot histogram) or 1 (plot histogram)
  Output : Null

  """
  for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    if printimg==1:
      plt.plot(histr, color = col)
      plt.xlim([0, 256])
  if printimg==1:    
    plt.title(title)
    plt.show()
  return

def EI(rbright1,gbright1):
  """
  Calculates the erythema index
  Inputs: 
  rbright1: Brightness of the R channel
  gbright1: Brightness of the G channel
  Outputs:
  EI: Erythema index of the image
  """
  return log(rbright1)-log(gbright1)

def plot_pixelDistr(r_shape,color,title=''):
  """
  Plots distribution of the pixels - rgb color image
  Inputs:
    r_shape: 1-D array of values of the colour
    color: Color of interest
    title (optional): title of the plot
    Output: Nothing, print plot
  """
  hist, bin_edges = np.histogram(r_shape)
  n, bins, patches = plt.hist(x=r_shape, bins='auto', color=color,
                              alpha=0.7, rwidth=0.85)
  plt.grid(axis='y', alpha=0.75)
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.title(title+' '+color)
  plt.show()

  return


def show_grayscale(img,printimg=1):
  '''
  Inputs: 
  img: image in array format with RGB channels
  printimg(default 1): Print grayscale image if 1
  Output:
  img_gry: the grayscale image 
  '''
  img_gry = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if printimg==1:
    plt.imshow(img_gry)
    plt.show()
  return img_gry

def convertBGRtoRGB(img1):
  '''
  Inputs:
  img1: Array of the image with BGR channel format
  Outputs:
  img1: Array of the image in RGB channel format
  '''
  # img1[:,:,[0,2]] = img1[:,:,[2,0]]
  return cv2.cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

def entropy(equ_shape,p_x):
  '''
  This function calculates the entropy of the pixels in the grayscale image
  Inputs:
  equ_shape: Equalized histogram of the grayscale image
  p_x: Probability function of the pixel values in the image
  Outputs:
  H_x: The entropy of pixels in the grayscale image
  '''
  H_x = 0
  for i in equ_shape:
    H_x += p_x[i]*np.log(p_x[i])
  H_x = -H_x
  return H_x

def probFunc(equ_shape):
  '''
  The probability of finding a pixel value in the image
  Inputs:
  equ_shape: The equalized histogram of grayscale values
  Outputs:
  p_x: The probability function of the grayscale pixel values
  '''
  unique_x = set(equ_shape)
  # print(unique_x)
  total_pixel = wid*ht
  p_x = {}
  for i in unique_x:
    p_x[i] = len(np.where(equ_shape==i)[0])
  print(sum(p_x.values()))
  return p_x
# print(total_pixel)


def colorDist(img,title='',disp=1):
  """
  Inputs: 
  img: image in array format with RGB channels
  title (optional): Title of the plot
  disp = 0 (no plot) or 1 (show plot)
  Output:
  r_shape:(size : width*height,1); reshaped pixel values for color red (size : width*height,1)
  b_shape:(size : width*height,1); reshaped pixel values for color blue
  g_shape:(size : width*height,1); reshaped pixel values for color green 
  """
  ht = img.shape[0]
  wid = img.shape[1]
  r_shape,b_shape,g_shape = img[:,:,0].reshape(wid*ht,1),img[:,:,2].reshape(wid*ht,1),img[:,:,1].reshape(wid*ht,1),
  colors=['red','blue','green']
  colors_ht = [np.mean(r_shape),np.mean(b_shape),np.mean(g_shape)]
  if disp:
    plt.bar(colors,colors_ht,color=colors,width=0.2)
    plt.grid()
    plt.title(title)
    plt.show()
  return r_shape,g_shape,b_shape

def HHR(img,threshold,printimg=1,binarize=0):
  '''
  Inputs:
  img: image in array format with RGB channels
  threshold: threshold to consider for the high hue ratio. Try 100
  printimg(opt): plot histogram of hue values (default 1); 0 to not plot
  binarize(default=0): if 1, binarize the HHR values (1 if HHR>0 else 0), else return HHR values as such
  Outputs:
  hhr : high hue ratio value
  img_hh: Pixels with the high hue value
  h_shape: HSV pixel values
  '''
  height,width,channel = img.shape
  img_hsv = np.zeros((height,width,3))
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h_shape,s_shape,v_shape = img_hsv[:,:,0].reshape(width*height,1),img_hsv[:,:,1].reshape(width*height,1),img_hsv[:,:,2].reshape(width*height,1),
  colors=['hue','sat','val']
  colors_ht = [np.mean(h_shape),np.mean(s_shape),np.mean(v_shape)]
  if printimg==1:
    plt.imshow(img_hsv)
    plt.title('Image in HSV space')
    plt.show()
    plt.bar(colors,colors_ht,color='blue',width=0.2)
    plt.title('Distribution of HSV values')
    plt.grid()
    plt.show()
  n = [h_shape>=threshold]
  res = np.reshape(n,width*height)
  # n = res
  # print(np.shape(res))
  # print(np.shape(n))
  img_hh = [i for i,res_i in enumerate(res) if res_i==True] #Store pixels with high hue value
  # print(len(img_hh))
  n_pixel_highhue = len(img_hh)
  hhr = n_pixel_highhue/(height*width)
  if binarize:
    if hhr>0:
      hhr = 1
  return hhr,img_hh,h_shape

#Print image
def loadAndPrintImg(imagename,printimg=1):
  '''
  This function loads and prepares the image to an array with RGB channels
  input: i
  imagename: Name of the file to be loaded
  printimg(default 1): print image then 1, else 0
  output: 
  img1: array with pixel values for RGB channels.
  '''
  print(imagename)
  img1 = convertBGRtoRGB(cv2.imread(imagename,1))
  if printimg==1:
    plt.imshow(img1)
    plt.title(imagename)
    plt.show()
  return img1
def PVM(col_shape):
  """
  Inputs: 
  col_shape: [r,g,b]
  Outputs:
  pvm_r: Pixel values mean for each of the R,G,B channels
  """
  # r_pos = np.where((img[:,:,0] > r[0]))
  print('Calculating PVM\n')
  # print(np.shape(col_shape)[0])
  # pvm_r = []
  pvm_r = np.zeros((np.shape(col_shape)[0],1))
  # print(pvm_r.shape)
  for i in range(np.shape(col_shape)[0]):
    # print(i)
    r_shape = col_shape[i]
    # print(r_shape)
    r = np.percentile(r_shape,[40,60])
    # print(r.shape)
    r_pos = np.where((r_shape> r[0]))
    # print(r_pos)
    r_pos_y=r_pos[0]
    # plt.plot(r_shape,color='red')
    # r_pos_reshape=np.reshape(r_pos,(3,wid,ht))
    #Calculating the PVM value
    pvm_r[i] = (1/(r_pos_y[-1]-r_pos_y[0]+1))*np.sum(r_shape[r_pos_y])
  return pvm_r

def PVM_main(rshape1,bshape1,gshape1):
  '''
  Input: Reshaped arrays (wid*ht,1) of the color channels. Usually output from colorDist function
  rshape1: Red channel (wid*ht,1)
  bshape1: Blue channel (wid*ht,1)
  gshape1: Green channel (wid*ht,1)
  Output:
  pvm_r1: pixel value means for each of the color channel
  '''
  pvm_r1 = PVM([rshape1,bshape1,gshape1])
  return pvm_r1

def entropyCalc(img,printhist=1):
  '''
  The function calculates the entropy to detect textures (capillaries that are prominent in the case of pallor) in the image
  First, the image is converted to grayscale. Second, the G component is histogram equalized. Then the entropy is calculated using entropy function.
  Input:
  img: The image array with pixel values for RGB channels
  printhist(default 1): 1 to plot histogram of grayscale values
  Output:
  img_gry: the grayscale image 
  h_1: entropy of the grayscale image.
  '''
  img_gry = show_grayscale(img)
  equ = cv2.equalizeHist(img_gry)
  wid=img.shape[0]
  ht = img.shape[1]
  equ_shape = np.reshape(equ,wid*ht)
  h_1 = entropy(equ_shape,probFunc(equ_shape))
  if printhist==1:
    n, bins, patches = plt.hist(x=np.reshape(equ,wid*ht), bins='auto', color='#0504aa', 
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value of grayscale')
    plt.ylabel('Frequency of value')
    plt.title('Histogram of grayscale image')
  return img_gry,h_1

def brightness(img_gry,printbrightness=1):
  '''
  The brightness of the image which is the average grayscale of the image
  Inputs:
  img_gry: The grayscale image
  printbrightness(defalt 1): To plot thefrequency of the grayscale values of the image 
  Outputs:
  brght: The mean brightness of the image
  '''
  wid = img_gry.shape[0]
  ht = img_gry.shape[1]
  img_gry_reshape = np.reshape(img_gry,wid*ht)
  brght = np.mean(img_gry_reshape)
  c = Counter(img_gry_reshape)
  x = list(c.keys())
  y = list(c.values()) 
  # / np.sum(list(c.values()))
  if printbrightness==1:
    plt.plot(x,y,'bo',x,np.ones(np.shape(y))*brght,'r--')
    plt.xlabel('Brightness Value')
    plt.ylabel('Brightness Frequency')
    plt.show()
  return brght

