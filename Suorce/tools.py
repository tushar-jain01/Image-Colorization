# -*- coding: utf-8 -*-
"""
Created on Fri May 28 22:22:31 2021

@author: TUSHAR JAIN
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def loadImagesToArray(dir_path, num_of_img=-1, search_inside =False):
  """
  dir_path : path of directory from which images will be imported
  num_of_imgs (Integer): number of images to be imported from the directory if not 
              given than all images will be imported 
  search_inside (boolean, default : False) : If true all images inside that directory
              along with the images in subdirectory will be added to output array
  """
  images = []
  count = -1
  if search_inside==False:
      for filename in os.listdir(dir_path):
          count+=1
          if(count==num_of_img):
              break
          images.append(img_to_array(load_img(dir_path+os.sep+filename)))
  if search_inside==True:
      for root,dirs,files in os.walk(dir_path):
        for filename in files:
            count+=1
            if(count==num_of_img):
                break
            images.append(img_to_array(load_img(root+os.sep+filename)))
  return np.array(images,dtype=float)/255.0

def DataGenerator():
    DataGen = ImageDataGenerator(        
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
    return DataGen

def RGB2GRAY(img,add_channel_dim=False):
  conv_matrix = np.array([0.212671 ,0.715160,0.072169])
  gray_img = img @ conv_matrix
  if add_channel_dim==True:
    return gray_img.reshape(np.array([*list(gray_img.shape),1]))
  else:
    return gray_img

def RGB2ab(img,use_skimage=True):
  """
  Refrences
  * https://en.wikipedia.org/wiki/Lab_color_space
  * https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py#L990-L1050
  """
  if use_skimage==False:
    def finv(cie):
      cond = cie > 0.008856
      cie[cond] = np.cbrt(cie[cond])
      cie[~cond] = 7.787 * cie[~cond] + 16. / 116.
      return cie     

    conv_matrix =np.array( [[0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]])
    CIE = np.matmul(img,conv_matrix.T)
    CIE[0] = CIE[0]/0.95047
    CIE[2] = CIE[2]/1.08883
    CIE = finv(CIE)
    x, y, z = CIE[..., 0], CIE[..., 1], CIE[..., 2]
    a =  (500*(x-y)+127)/255.0
    b =  (200*(y-z)+127)/255.0
    return np.concatenate([x[..., np.newaxis] for x in [a, b]], axis=-1)
  else:
    Lab = rgb2lab(img)
    a = (Lab[...,1]+127)/255.0
    b = (Lab[...,2]+127)/255.0
    return np.concatenate([x[..., np.newaxis] for x in [a, b]], axis=-1)

def Lab2RGB(gray,ab):
  """
    Parameters
    ----------
    gray : nd array
        lumminnance component of a image.
    ab : TYPE
        a and b componenets of a CIE L*a*b image.

    Returns
    -------
    ndarray with R G B components.
  """
  ab = ab*255.0 -127
  gray = gray*100
  Lab =np.concatenate([x[..., np.newaxis] for x in [gray[...,0], ab[...,0],ab[...,1]]], axis=-1)
  return lab2rgb(Lab)

def compare_results(img_gt,img_in,img_out,save_results=False,save_as=""):
  """
    Parameters
    ----------
    img_gt : ndarray with RGB components 
        Original Required image model is expected to produce this as ouput.
    img_in : grayscaled ndarray
        image used as input to the model.
    img_out : nd array with RGB componets.
        The ouput from the model.
    save_results : boolean, optional
        If True matplotlib.plt will be used to save model. The default is False.
    save_as : String, optional
        Output file name along with path. The default is "".

    Returns
    -------
    None.

  """
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
  ax1.imshow(img_gt)
  ax1.set_title('Ground Truth')
  ax2.imshow(img_in,cmap='gray')
  ax2.set_title('Input')
  ax3.imshow(img_out)
  ax3.set_title('Output')
  axes = [ax1,ax2,ax3]
  for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
  plt.show()
  if save_results==True:
    path = save_as+'.svg'
    fig.savefig(path,dpi=300)

def BatchGenerator(data,imgDataGen,batch_size=64):
  for batch in imgDataGen.flow(data, batch_size=batch_size):
    yield RGB2GRAY(batch,True), RGB2ab(batch)


    
