import json,time
import cv2
import numpy as np
from PIL import Image


def refine_edges(rgb,coarse,model,processor,excluded=None,band=12):
    hard=(coarse>.5).astype(np.uint8)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band+1,2*band+1))
    lo=cv2.erode(hard,kernel);hi=cv2.dilate(hard,kernel)
    trimap=np.where(lo>0,255,np.where(hi==0,0,128)).astype(np.uint8)
    if excluded is not None:trimap[(excluded>.90)&(coarse<.5)]=0
    unknown=trimap==128;h,w=coarse.shape
    alpha_sum=np.zeros((h,w),np.float32);weight_sum=np.zeros((h,w),np.float32)
    core,stride,pad=512,448,64
    count=0;start=time.monotonic()
    for y in range(0,h,stride):
        for x in range(0,w,stride):
            x1=min(w,x+core);y1=min(h,y+core)
            u=unknown[y:y1,x:x1]
            if not u.any():continue
            rx0=max(0,x-pad);ry0=max(0,y-pad);rx1=min(w,x1+pad);ry1=min(h,y1+pad)
            inputs=processor(images=Image.fromarray(rgb[ry0:ry1,rx0:rx1]),trimaps=Image.fromarray(trimap[ry0:ry1,rx0:rx1]),return_tensors='pt')
            pred=model(**inputs).alphas[0,0].cpu().numpy()[:ry1-ry0,:rx1-rx0]
            local=pred[y-ry0:y1-ry0,x-rx0:x1-rx0]
            wx=np.ones(x1-x,np.float32);wy=np.ones(y1-y,np.float32)
            fade=core-stride
            if x>0:wx[:min(fade,len(wx))]=np.linspace(.01,1,min(fade,len(wx)))
            if x1<w:wx[-min(fade,len(wx)):]=np.linspace(1,.01,min(fade,len(wx)))
            if y>0:wy[:min(fade,len(wy))]=np.linspace(.01,1,min(fade,len(wy)))
            if y1<h:wy[-min(fade,len(wy)):]=np.linspace(1,.01,min(fade,len(wy)))
            weight=wy[:,None]*wx[None,:]*u
            alpha_sum[y:y1,x:x1]+=local*weight;weight_sum[y:y1,x:x1]+=weight;count+=1
    if np.any(unknown & (weight_sum==0)):
        raise RuntimeError('边缘分块存在未覆盖区域。')
    result=np.where(trimap==255,1,np.where(trimap==0,0,alpha_sum/np.maximum(weight_sum,1e-6)))
    report={'tiles':count,'seconds':round(time.monotonic()-start,2),'original_pixel_band':band,
            'tile_overlap_pixels':core-stride,'rgb_redrawn':False}
    return np.clip(result,0,1).astype(np.float32),json.dumps(report,ensure_ascii=False)
