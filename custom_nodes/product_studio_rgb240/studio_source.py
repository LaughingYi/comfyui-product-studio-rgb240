import cv2
import numpy as np
from .source_shadow import source_shadow
from .image_colour import foreground_colour,tone_gain


def compose_source(rgb,alpha,excluded,instances,style,out_size=2048,shadow_strength=1.,light_strength=1.):
 h,w=alpha.shape;ratio=min(1600/max(h,w),1.)
 analysis_size=(max(1,round(w*ratio)),max(1,round(h*ratio)))
 small=cv2.resize(rgb,analysis_size,interpolation=cv2.INTER_AREA)
 small_alpha=cv2.resize(alpha,analysis_size,interpolation=cv2.INTER_AREA)
 small_excluded=cv2.resize(excluded,analysis_size,interpolation=cv2.INTER_AREA)
 measured,reports=source_shadow(small,small_alpha,small_excluded,instances)
 side=int(out_size);scale=min(side/w,side/h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
 interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LANCZOS4
 foreground=foreground_colour(rgb,alpha)
 a=np.clip(cv2.resize(alpha,(nw,nh),interpolation=interpolation),0,1)
 premul=cv2.resize(foreground*alpha[...,None],(nw,nh),interpolation=interpolation)
 colour=premul/np.maximum(a[...,None],1e-5)
 gain=tone_gain(colour,a,style['tone_log_gain_coefficients'],style['tone_gain_bounds'])
 gain=1+(gain-1)*float(light_strength)
 premul=np.clip(colour*gain[...,None],0,1)*a[...,None]
 shadow=np.clip(cv2.resize(measured,(nw,nh),interpolation=cv2.INTER_LINEAR)*float(shadow_strength),0,.65)
 shadow[shadow<.002]=0
 canvas=np.full((side,side,3),240/255,np.float32)
 x,y=(side-nw)//2,(side-nh)//2
 canvas[y:y+nh,x:x+nw]=premul+(240/255)*(1-shadow[...,None])*(1-a[...,None])
 report={'base_rgb':[240,240,240],'global_gradient':False,'shadow_method':'source_photo_luminance_estimation',
  'objects':reports,'shadow_peak':round(float(shadow.max()),4),'geometry':'source_frame','rgb_generated':False,
  'review_needed':not any(r['status']=='measured' and r['peak']>.015 for r in reports)}
 return np.clip(canvas,0,1).astype(np.float32),shadow,report
