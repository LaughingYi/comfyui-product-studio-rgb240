import cv2
import numpy as np

LUMA=np.array([.2126,.7152,.0722],np.float32)

def material_centres(ab):
 sample=ab[::max(1,len(ab)//4000)]
 centres=[np.median(sample,axis=0)]
 for _ in range(2):
  distance=np.min(np.sum((sample[:,None]-np.array(centres)[None])**2,2),1)
  if np.percentile(distance,95)<49:break
  centres.append(sample[np.argmax(distance)])
 centres=np.array(centres,np.float32)
 for _ in range(8):
  labels=np.argmin(np.sum((sample[:,None]-centres[None])**2,2),1)
  for j in range(len(centres)):
   if (labels==j).any():centres[j]=np.median(sample[labels==j],axis=0)
 return centres

def source_shadow(rgb,alpha,excluded,instances):
 h,w=alpha.shape
 background_weight=((alpha<.01)&(excluded<.01)).astype(np.float32)
 denominator=cv2.GaussianBlur(background_weight,(0,0),1.2)
 smooth=cv2.GaussianBlur(rgb*background_weight[...,None],(0,0),1.2)/np.maximum(denominator[...,None],1e-5)
 lum=smooth@LUMA
 lab=cv2.cvtColor(np.clip(smooth,0,1).astype(np.float32),cv2.COLOR_RGB2LAB)
 valid=(alpha<.01)&(excluded<.01)&(lum>.025)
 prepared=[]
 for item in instances:
  hint=cv2.resize(item.astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST)>.5
  hint=cv2.dilate(hint.astype(np.uint8),np.ones((5,5),np.uint8))>0
  obj=(alpha>.5)&hint;ys,xs=np.where(obj)
  if len(ys):prepared.append((obj,[xs.min(),ys.min(),xs.max()+1,ys.max()+1]))
 upright=[z for z in prepared if (z[1][3]-z[1][1])/(z[1][2]-z[1][0])>.55]
 main=max(upright or prepared,key=lambda z:z[0].sum(),default=None)
 if main is None:return np.zeros_like(alpha),[]
 main_floor=main[1][3];minimum_floor=main_floor-.1*(main[1][3]-main[1][1])
 total=np.zeros((h,w),np.float32);reports=[]
 for obj,box in prepared:
  x0,y0,x1,y1=map(int,box);bw,bh=x1-x0,y1-y0
  if y1<minimum_floor:
   reports.append({'box':list(map(int,box)),'status':'elevated'});continue
  local=obj[y0:y1,x0:x1];cols=np.where(local.any(0))[0]
  bottom=(y1-1)-np.argmax(local[::-1],axis=0)
  curve=np.interp(np.arange(bw),cols,bottom[cols]).astype(np.float32)
  left=max(0,x0-round(.20*bw));right=min(w,x1+round(.20*bw))
  top=max(0,int(np.percentile(curve,10)-.035*bw));end=min(h,y1+round(.20*bw))
  v=valid[top:end,left:right];yy,xx=np.mgrid[top:end,left:right].astype(np.float32)
  base_curve=np.interp(np.arange(left,right),np.arange(x0,x1),curve)
  delta=yy-base_curve[None,:]
  floor_gate=np.exp(-np.maximum(delta,0)/max(4,bw*.09))*np.exp(-np.maximum(-delta,0)/max(2,bw*.018))
  side_distance=np.maximum(np.maximum(x0-xx,xx-x1),0)
  floor_gate*=np.exp(-(side_distance/max(3,bw*.075))**2)
  if v.sum()<200:
   reports.append({'box':list(map(int,box)),'status':'no_floor_samples'});continue
  ab=lab[top:end,left:right,1:3];centres=material_centres(ab[v])
  labels=np.argmin(np.sum((ab[...,None,:]-centres[None,None])**2,3),2)
  normal_x=(xx-(x0+x1)*.5)/bw;normal_y=(yy-y1)/bw
  design=np.stack([np.ones_like(xx),normal_x,normal_y],-1)
  l=lum[top:end,left:right]
  measured=np.zeros_like(l);confidence=np.zeros_like(l)
  for j in range(len(centres)):
   selected=v&(labels==j)
   far=selected&(floor_gate<.45)
   use=far if far.sum()>100 else selected
   if use.sum()<100:continue
   A=design[use][::max(1,int(use.sum())//6000)];b=l[use][::max(1,int(use.sum())//6000)]
   weights=np.ones(len(b),np.float32)
   for _ in range(9):
    coef=np.linalg.lstsq(A*weights[:,None],b*weights,rcond=None)[0]
    residual=b-A@coef
    asym=np.where(residual>0,.80,.20)
    weights=np.sqrt(asym/np.maximum(np.abs(residual),.006))
   baseline=np.clip(design@coef,.05,1.)
   ratio=1-l/np.maximum(baseline,.05)
   reference=ratio[far] if far.sum()>80 else ratio[selected]
   median=float(np.median(reference));noise=float(np.median(np.abs(reference-median)))
   amount=np.clip(ratio-median-max(.008,1.5*noise),0,.65)
   measured[selected]=amount[selected];confidence[selected]=1
  sigma=max(1.2,bw*.004)
  numerator=cv2.GaussianBlur(measured*confidence,(0,0),sigma)
  denominator=cv2.GaussianBlur(confidence,(0,0),sigma)
  amount=numerator/np.maximum(denominator,.025)
  amount[denominator<.025]=0
  amount*=floor_gate
  # Discard tiny residual floor texture and fade into the exact uniform base.
  amount=np.maximum(amount-.004,0)
  field=np.zeros_like(alpha);field[top:end,left:right]=amount
  total=np.maximum(total,field)
  reports.append({'box':list(map(int,box)),'status':'measured','floor_samples':int(v.sum()),'materials':len(centres),'peak':round(float(amount.max()),4)})
 total=cv2.GaussianBlur(total,(0,0),max(2,w*.004))
 total[total<.003]=0
 return np.clip(total,0,.65).astype(np.float32),reports
