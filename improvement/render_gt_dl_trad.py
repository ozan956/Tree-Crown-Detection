"""Overlay GT (grey box) + DL detections (blue) + traditional detections (red)
on every test image, both datasets. Visual inspection only.

VHRTrees: GT=COCO boxes, DL=WBF boxes (score>=0.8), traditional=tree_centers.json.
NEON:     GT=VOC boxes, DL=DeepForest boxes, traditional=traditional_centers().

Run: .venv/bin/python -m improvement.render_gt_dl_trad
Writes results/overlay_vhrtrees/*.jpg and results/overlay_neon/*.jpg
"""
import os, json, glob, cv2
from evaluation.io_utils import load_gt, load_dets, dets_by_image

GREY=(170,170,170); BLUE=(220,140,0); RED=(0,40,220); WHITE=(255,255,255)
VHR_IMG="wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/inputs"
TREE_CENTERS=glob.glob("wbf_test-*/wbf_test/**/tree_centers.json",recursive=True)[0]


def _box(img,b,color):
    cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),color,1)

def _dot(img,x,y,color,r=4):
    cv2.circle(img,(int(x),int(y)),r,color,-1); cv2.circle(img,(int(x),int(y)),r,WHITE,1)


def vhrtrees():
    out="results/overlay_vhrtrees"; os.makedirs(out,exist_ok=True)
    ann=json.load(open("data/annotations.coco.json"))
    names={im["id"]:im["file_name"] for im in ann["images"]}
    gt=load_gt()
    wbf=dets_by_image(load_dets("data/wbf/best_fuse.json"),0.8)  # DL (blue)
    # traditional centers grouped by image id; (tree_center_x, tree_center_y)
    # lands on crowns (verified visually on a sparse scene).
    tc={}
    for r in json.load(open(TREE_CENTERS)):
        tc.setdefault(r["id"],[]).append((r["tree_center_x"], r["tree_center_y"]))
    n=0
    for iid,boxes in gt.items():
        img=cv2.imread(os.path.join(VHR_IMG,names[iid]))
        if img is None: continue
        for b in boxes: _box(img,b,GREY)                                  # GT grey
        for b,s in wbf.get(iid,[]): _dot(img,(b[0]+b[2])/2,(b[1]+b[3])/2,BLUE)  # DL blue
        for (x,y) in tc.get(iid,[]): _dot(img,x,y,RED,3)                  # traditional red
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
        for b in load_voc_boxes(os.path.join("data_neon/ann",it["xml"])): _box(img,b,GREY)
        pred=m.predict_image(path=p)
        if pred is not None and len(pred):
            for r in pred.itertuples(): _dot(img,(r.xmin+r.xmax)/2,(r.ymin+r.ymax)/2,BLUE)
        for (x,y) in traditional_centers(img.copy()): _dot(img,x,y,RED,3)
        cv2.imwrite(os.path.join(out,it["tif"].replace(".tif",".jpg")),img); n+=1
    print(f"NEON: {n} images -> {out}/")


if __name__=="__main__":
    vhrtrees(); neon()
