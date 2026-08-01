"""Render final integrated+gate+NMS detections on all 222 VHRTrees test images.
One green dot per detected tree, drawn on the source image, for visual inspection.

Gate trained on all far-points then applied to all (this is for eyeballing, not a
held-out metric). Run: .venv/bin/python -m improvement.render_all
Writes results/detections_all/<image>.jpg (222 files).
"""
import os, json, cv2
from evaluation.io_utils import load_gt, load_points, load_dets, dets_by_image
from improvement.precision_gate import (crop_features, box_centers, is_near_box,
                                        merge_close_points, PrecisionGate, NEAR_SCORE)
from improvement.run_gate import _tp_mask

IMG="wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/inputs"
OUT="results/detections_all"; GATE_THR=0.5; NMS=26.0

def main():
    os.makedirs(OUT, exist_ok=True)
    ann=json.load(open("data/annotations.coco.json")); names={im["id"]:im["file_name"] for im in ann["images"]}
    gt=load_gt(); pts=load_points(); wbf=dets_by_image(load_dets("data/wbf/best_fuse.json"),NEAR_SCORE)

    # per-image: near-kept coords + far (coords, feats); train gate on all far
    per={}; Xtr=[]; ytr=[]
    for iid in gt:
        P=pts.get(iid,[]); r={"near":[], "far_xy":[], "far_X":[]}
        if P:
            img=cv2.imread(os.path.join(IMG,names[iid])); wc=box_centers(wbf.get(iid,[]))
            for (x,y) in P:
                if is_near_box(x,y,wc): r["near"].append((x,y)); continue
                f=crop_features(img,x,y,wbf_centers=wc) if img is not None else None
                if f is None: r["near"].append((x,y)); continue
                r["far_xy"].append((x,y)); r["far_X"].append(f)
                Xtr.append(f); ytr.append(1 if _tp_mask([(x,y)],gt[iid]) else 0)
        per[iid]=r
    gate=PrecisionGate(threshold=GATE_THR).fit(Xtr,ytr)

    n=0
    for iid in gt:
        r=per[iid]; kept=list(r["near"])
        if r["far_X"]:
            keep=gate.keep(r["far_X"]); kept+=[xy for xy,k in zip(r["far_xy"],keep) if k]
        kept=merge_close_points(kept,NMS)
        img=cv2.imread(os.path.join(IMG,names[iid]))
        if img is None: continue
        for (x,y) in kept:
            cv2.circle(img,(int(x),int(y)),4,(0,220,0),-1)
            cv2.circle(img,(int(x),int(y)),4,(255,255,255),1)
        cv2.putText(img,f"{len(kept)} trees",(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,0),2,cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT,names[iid]),img); n+=1
    print(f"wrote {n} images to {OUT}/")

if __name__=="__main__":
    main()
