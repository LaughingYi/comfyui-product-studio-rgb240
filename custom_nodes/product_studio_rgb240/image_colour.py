import cv2
import numpy as np

LUMA=np.array([.2126,.7152,.0722],np.float32)

def foreground_colour(rgb,a):
 radius=max(5,round(max(a.shape)*.003));fg_w=a**8;bg_w=(1-a)**12
 def mean(weight):
  z=cv2.boxFilter(weight,-1,(2*radius+1,2*radius+1),normalize=True)
  value=cv2.boxFilter(rgb*weight[...,None],-1,(2*radius+1,2*radius+1),normalize=True)/np.maximum(z[...,None],1e-5)
  return np.where(z[...,None]>.001,value,rgb)
 F=mean(fg_w);B=mean(bg_w)
 recovered=(rgb-B*(1-a[...,None]))/np.maximum(a[...,None],.08)
 weight=np.clip(a/.20,0,1)[...,None]
 recovered=np.clip(recovered*weight+F*(1-weight),0,1)
 return np.where(((a>.001)&(a<.999))[...,None],recovered,rgb).astype(np.float32)

def tone_gain(rgb,alpha,coefficients,bounds):
 lum=rgb@LUMA;weight=(alpha>.95).astype(np.float32)
 sigma=max(2,rgb.shape[1]*.008)
 smooth=cv2.GaussianBlur(lum*weight,(0,0),sigma)/np.maximum(cv2.GaussianBlur(weight,(0,0),sigma),1e-5)
 gain=np.exp(coefficients[0]+coefficients[1]*(smooth-.45))
 return np.clip(gain,*bounds).astype(np.float32)
