"""Repair opaque product masks with local object proposals; never synthesize pixels."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .ground_objects import select_ground_instances


def box_iou(a, b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1])
    x1,y1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0)
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1)


def remove_suspended_lines(alpha):
    hard=(alpha>.35).astype(np.uint8)
    h,w=hard.shape
    k=max(5,round(w*.005))
    thick=cv2.morphologyEx(hard,cv2.MORPH_OPEN,np.ones((1,k),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(thick,8)
    valid=[i for i in range(1,n) if st[i,4]>h*w*.002]
    if not valid:return alpha,0
    top=min(st[i,1] for i in valid)
    upper=(alpha>.02).astype(np.uint8);upper[top:]=0
    n,lab,st,_=cv2.connectedComponentsWithStats(upper,8)
    drop=np.zeros_like(hard)
    for i in range(1,n):
        x,y,bw,bh,area=st[i]
        if bh<max(20,h*.06) or bw>bh*.18:continue
        part=lab[y:y+bh,x:x+bw]==i
        widths=(part & (alpha[y:y+bh,x:x+bw]>.50)).sum(1);widths=widths[widths>0]
        if len(widths) and np.median(widths)<=max(2,w*.002):
            drop[lab==i]=1
    # Remove only the exterior filament; product pixels at the handle stay intact.
    drop=cv2.dilate(drop,np.ones((3,3),np.uint8));drop[top:]=0
    out=alpha.copy();out[drop>0]=0
    return out,int(drop.sum())


def merge_proposals(raw, masks, scenes, remove_lines=True):
    h,w=raw.shape
    masks=np.asarray(masks,dtype=bool)
    drop=np.zeros((h,w),np.uint8)
    paper_drop=np.zeros_like(drop)
    removed=[]
    # Each scene is a detector-confirmed packaging/frame region. Match a complete
    # segmented object, never erase the detector's entire rectangular box.
    for scene in scenes:
        scored=[]
        for i,m in enumerate(masks):
            ys,xs=np.where(m)
            if not len(ys):continue
            box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
            scored.append((box_iou(box,scene['box']),i))
        if not scored:continue
        scored.sort(reverse=True)
        paper=scene['label'] in ['paper card','instruction manual','paper document']
        chosen=[i for score,i in scored if score>.35] if paper else ([scored[0][1]] if scored[0][0]>.50 else [])
        for i in chosen:
            solid=masks[i].astype(np.uint8)
            inverse=np.pad(1-solid,1,constant_values=1)
            cv2.floodFill(inverse,None,(0,0),0)
            surface=np.maximum(solid,inverse[1:-1,1:-1])
            if paper:paper_drop=np.maximum(paper_drop,surface)
            else:drop=np.maximum(drop,surface)
        if chosen:removed.append(scene['label'])
    packaging_drop=drop.copy()
    drop=np.maximum(drop,paper_drop)
    opaque=np.zeros((h,w),np.uint8)
    selected=[]
    candidates=[]
    for i,m in enumerate(masks):
        size=int(m.sum())
        if size<max(16,int(h*w*.00005)):continue
        if float(drop[m].mean())>.50:continue
        v=raw[m]
        weak=float((v>.05).mean());strong=float((v>.80).mean())
        if weak>.35 or strong>.12:
            opaque[m]=1;selected.append(i)
        elif weak>.04 and size<h*w*.04:
            candidates.append(i)
    near=cv2.dilate(opaque,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,round(w*.025))|1,)*2))
    for i in candidates:
        m=masks[i]
        if float(near[m].mean())>.20:
            opaque[m]=1;selected.append(i)
    guard=max(2,round(min(h,w)*.008))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*guard+1,2*guard+1))
    drop=cv2.dilate(packaging_drop,kernel)
    paper_drop=cv2.dilate(paper_drop,kernel)
    # A paper card can lie behind a confidently identified cord. Preserve the
    # connected product surface rather than applying the card's solid silhouette.
    count,labels=cv2.connectedComponents((raw>.80).astype(np.uint8),8)
    support=np.bincount(labels[opaque>0],minlength=count)
    support[0]=0
    trusted=(support[labels]>=32).astype(np.uint8)
    trusted=cv2.dilate(trusted,np.ones((5,5),np.uint8))
    paper_drop[trusted>0]=0
    drop=np.maximum(drop,paper_drop)
    drop[opaque>0]=0
    # Recovered object surfaces are opaque. Genuine holes remain holes because
    # proposals segment the surface rather than filling every enclosed void.
    a=np.maximum(np.clip(raw,0,1),opaque.astype(np.float32))
    a[drop>0]=0
    removed_line_pixels=0
    if remove_lines:a,removed_line_pixels=remove_suspended_lines(a)
    report={'repair_proposals':len(selected),'removed_scene_objects':removed,
            'removed_exterior_line_pixels':removed_line_pixels,
            'review':'自动补全依赖物体候选；完全漏检的配件或手柄上的绑线仍需查看原图复核。'}
    return a.astype(np.float32),drop.astype(np.float32),report


def run_local_repair(rgb, raw, model_dir, remove_boxes=True, remove_lines=True, remove_documents=True):
    # Require the existing local assets rather than downloading during a workflow.
    os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    from ultralytics.models.sam import Predictor
    from ultralytics import YOLO
    from ultralytics.utils import WEIGHTS_DIR
    models=Path(model_dir)
    needed=[models/'mobile_sam.pt']
    if remove_boxes or remove_documents:needed += [models/'yolov8s-worldv2.pt',WEIGHTS_DIR/'clip/ViT-B-32.pt']
    for p in needed:
        if not p.is_file():raise FileNotFoundError(f'本地蒙版修复模型缺失：{p}；未自动下载。')
    class ProductProposals(Predictor):
        def inference(self, im, *args, **kwargs):
            return self.generate(im,points_stride=16,points_batch_size=32,
                                 conf_thres=.75,stability_score_thresh=.85)
    height,width=rgb.shape[:2]
    ratio=min(1600/max(height,width),1.)
    size=(max(1,round(width*ratio)),max(1,round(height*ratio)))
    image=cv2.resize(np.clip(rgb*255,0,255).astype(np.uint8),size)
    bgr=image[...,::-1].copy()
    predictor=ProductProposals(overrides={'model':str(models/'mobile_sam.pt'),'device':'cpu',
        'imgsz':1024,'verbose':False,'save':False})
    result=predictor(bgr)[0]
    if result.masks is None:raise RuntimeError('物体轮廓补全未返回候选蒙版，请检查输入图片。')
    masks=result.masks.data.cpu().numpy().astype(bool)
    scenes=[]
    if remove_boxes or remove_documents:
        detector=YOLO(str(models/'yolov8s-worldv2.pt'))
        classes=(['cardboard box','picture frame','figurine'] if remove_boxes else [])
        if remove_documents:classes+=['paper card','instruction manual','paper document','handbag','fabric dust bag']
        detector.set_classes(classes)
        det=detector.predict(bgr,imgsz=1280,conf=.10,device='cpu',verbose=False)[0]
        for box,cat,score in zip(det.boxes.xyxy.cpu().numpy(),det.boxes.cls.cpu().numpy().astype(int),det.boxes.conf.cpu().numpy()):
            label=det.names[cat]
            if label in ['handbag','fabric dust bag']:continue
            threshold=.20 if label in ['paper card','instruction manual','paper document'] else .35
            if score>=threshold:scenes.append({'label':label,'confidence':float(score),'box':box.tolist()})
    small_raw=cv2.resize(raw,(masks.shape[2],masks.shape[1]),interpolation=cv2.INTER_LINEAR)
    repaired,excluded,report=merge_proposals(small_raw,masks,scenes,remove_lines)
    # Retain the original high-resolution boundary; add only newly recovered areas.
    removed=np.clip(small_raw-repaired,0,1)
    out=np.maximum(raw,cv2.resize(repaired,(width,height),interpolation=cv2.INTER_LINEAR))
    out*=1-cv2.resize((removed>.02).astype(np.float32),(width,height),interpolation=cv2.INTER_LINEAR)
    excluded=cv2.resize(excluded,(width,height),interpolation=cv2.INTER_LINEAR)
    ground_objects,ground_report=select_ground_instances(repaired,masks)
    report['ground_object_proposals']=ground_report
    return np.clip(out,0,1).astype(np.float32),excluded.astype(np.float32),json.dumps(report,ensure_ascii=False),ground_objects
