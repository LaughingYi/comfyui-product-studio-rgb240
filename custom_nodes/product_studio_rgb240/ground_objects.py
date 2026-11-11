import cv2
import numpy as np

def select_ground_instances(alpha,proposals):
 h,w=alpha.shape;solid=alpha>.5;ys,xs=np.where(solid)
 if not len(ys):return [],[]
 floor_min=ys.min()+.48*(ys.max()-ys.min())
 valid=[]
 for i,m in enumerate(proposals):
  m=m.astype(bool);size=m.sum()
  if size<h*w*.0012:continue
  overlap=solid[m].mean()
  if overlap<.88:continue
  yy,xx=np.where(m)
  if yy.max()<floor_min:continue
  valid.append((size,i,m))
 valid.sort(reverse=True);selected=[];covered=np.zeros((h,w),bool);rows=[]
 for size,i,m in valid:
  new=(m&~covered).sum()/size
  if new<.45:continue
  yy,xx=np.where(m);box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
  selected.append(m);covered|=m;rows.append({'proposal':i,'box':box,'area':int(size),'new_fraction':round(float(new),3)})
 return selected,rows
