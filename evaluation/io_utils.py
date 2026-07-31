import json

def xywh_to_corners(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)

def box_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)

def load_gt(path="data/annotations.coco.json"):
    d = json.load(open(path))
    gt = {}
    for a in d["annotations"]:
        gt.setdefault(a["image_id"], []).append(xywh_to_corners(a["bbox"]))
    return gt

def gt_total(gt):
    return sum(len(v) for v in gt.values())

def load_dets(path):
    return json.load(open(path))

def dets_by_image(dets, score_thr=0.0):
    out = {}
    for d in dets:
        if d["score"] < score_thr:
            continue
        out.setdefault(d["image_id"], []).append((xywh_to_corners(d["bbox"]), d["score"]))
    return out

def load_points(path="data/integrated/points.json"):
    groups = json.load(open(path))
    out = {}
    for g in groups:
        for t in g:
            out.setdefault(t["image_id"], []).append((t["tree_x"], t["tree_y"]))
    return out
