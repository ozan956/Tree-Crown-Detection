"""Overlay, per test image: GT (grey box) + FINALIZED detections colored by source.

Finalized = the integrated+gate+NMS output (the detections the method actually
returns). Each finalized point is colored by where it came from:
  blue = DL-origin  (point near a reliable WBF/DeepForest box)
  red  = traditional-origin (far-from-box point accepted by the appearance gate)
Grey box = ground truth. Raw pre-gate firings are NOT shown.

Run: .venv/bin/python -m improvement.render_gt_dl_trad
Writes results/overlay_vhrtrees/*.jpg and results/overlay_neon/*.jpg
"""
import os, json, glob, cv2
from evaluation.io_utils import load_gt, load_points, load_dets, dets_by_image
from improvement.precision_gate import (crop_features, box_centers, is_near_box,
                                        merge_close_points, PrecisionGate, NEAR_SCORE)
from improvement.run_gate import _tp_mask

GREY=(170,170,170); BLUE=(220,140,0); RED=(0,40,220); WHITE=(255,255,255)
VHR_IMG="wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/inputs"
GATE_THR=0.5; NMS=26.0


def _box(img,b): cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),GREY,1)
def _dot(img,x,y,color): cv2.circle(img,(int(x),int(y)),4,color,-1); cv2.circle(img,(int(x),int(y)),4,WHITE,1)


def vhrtrees():
    out="results/overlay_vhrtrees"; os.makedirs(out,exist_ok=True)
    ann=json.load(open("data/annotations.coco.json")); names={im["id"]:im["file_name"] for im in ann["images"]}
    gt=load_gt(); pts=load_points(); wbf=dets_by_image(load_dets("data/wbf/best_fuse.json"),NEAR_SCORE)
    # per image: near(DL) points + far(trad) points+feats; train gate on all far
    per={}; Xtr=[]; ytr=[]
    for iid in gt:
        P=pts.get(iid,[]); r={"dl":[], "far_xy":[], "far_X":[]}
        if P:
            img=cv2.imread(os.path.join(VHR_IMG,names[iid])); wc=box_centers(wbf.get(iid,[]))
            for (x,y) in P:
                if is_near_box(x,y,wc): r["dl"].append((x,y)); continue
                f=crop_features(img,x,y,wbf_centers=wc) if img is not None else None
                if f is None: r["dl"].append((x,y)); continue
                r["far_xy"].append((x,y)); r["far_X"].append(f)
                Xtr.append(f); ytr.append(1 if _tp_mask([(x,y)],gt[iid]) else 0)
        per[iid]=r
    gate=PrecisionGate(threshold=GATE_THR).fit(Xtr,ytr)
    n=0
    for iid in gt:
        img=cv2.imread(os.path.join(VHR_IMG,names[iid]))
        if img is None: continue
        for b in gt[iid]: _box(img,b)
        r=per[iid]
        trad=[xy for xy,k in zip(r["far_xy"], gate.keep(r["far_X"]) if r["far_X"] else []) if k]
        # NMS over the union so duplicates dropped, but keep source tag by nearest-origin
        dl=r["dl"]
        dl_k=merge_close_points(dl,NMS)
        trad_k=[p for p in merge_close_points(dl+trad,NMS) if p not in dl_k]  # trad survivors
        for (x,y) in dl_k: _dot(img,x,y,BLUE)
        for (x,y) in trad_k: _dot(img,x,y,RED)
        cv2.putText(img,f"{len(dl_k)} DL  {len(trad_k)} trad",(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,WHITE,2,cv2.LINE_AA)
        cv2.imwrite(os.path.join(out,names[iid]),img); n+=1
    print(f"VHRTrees: {n} images -> {out}/")


def neon():
    out="results/overlay_neon"; os.makedirs(out,exist_ok=True)
    from deepforest import main as dfm
    from improvement.neon.cross_dataset import load_voc_boxes
    from improvement.neon.traditional import traditional_centers
    m=dfm.deepforest(); m.load_model("weecology/deepforest-tree")
    index=json.load(open("data_neon/index.json"))
    n=0
    for it in index:
        p=os.path.join("data_neon/rgb",it["tif"]); img=cv2.imread(p)
        if img is None: continue
        for b in load_voc_boxes(os.path.join("data_neon/ann",it["xml"])): _box(img,b)
        pred=m.predict_image(path=p)
        dl=[((r.xmin+r.xmax)/2,(r.ymin+r.ymax)/2) for r in pred.itertuples()] if pred is not None and len(pred) else []
        wc=dl
        # traditional-origin = trad centers far from any DeepForest box
        trad=[(x,y) for (x,y) in traditional_centers(img.copy()) if not is_near_box(x,y,wc)]
        dl_k=merge_close_points(dl,NMS)
        trad_k=[q for q in merge_close_points(dl+trad,NMS) if q not in dl_k]
        for (x,y) in dl_k: _dot(img,x,y,BLUE)
        for (x,y) in trad_k: _dot(img,x,y,RED)
        cv2.putText(img,f"{len(dl_k)} DL  {len(trad_k)} trad",(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,WHITE,2,cv2.LINE_AA)
        cv2.imwrite(os.path.join(out,it["tif"].replace(".tif",".jpg")),img); n+=1
    print(f"NEON: {n} images -> {out}/")


if __name__=="__main__":
    vhrtrees(); neon()
