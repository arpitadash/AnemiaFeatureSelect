
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
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

def PVM(col_shape):
  """
  Inputs: 
  col_shape: [r,g,b]
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

def show_grayscale(img,printimg=1):
  img_gry = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if printimg==1:
    plt.imshow(img_gry)
    plt.show()
  return img_gry

def convertBGRtoRGB(img1):
  img1[:,:,[0,2]] = img1[:,:,[2,0]]
  return img1

def entropy(equ_shape,p_x):
  H_x = 0
  for i in equ_shape:
    H_x += p_x[i]*np.log(p_x[i])
  H_x = -H_x
  return H_x

def probFunc(equ_shape):
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

def HHR(img,threshold,printimg=1):
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
  # print(hhr)
  return hhr,img_hh,h_shape

#Print image
def loadAndPrintImg(imagename,printimg=1):
  print(imagename)
  img1 = convertBGRtoRGB(cv2.imread(imagename,1))
  if printimg==1:
    plt.imshow(img1)
    plt.title(imagename)
    plt.show()
  return img1

def PVM_main(rshape1,bshape1,gshape1):
  pvm_r1 = PVM([rshape1,bshape1,gshape1])
  # pvm_b1 = PVM(bshape1)
  # pvm_g1 = PVM(gshape1)
  return pvm_r1

def entropyCalc(img,printhist=1):
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

