import contextlib, io, json, tempfile, os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def pr_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}

def coco_map(gt_coco_path, dets, cat_id=1):
    if not dets:
        # pycocotools' loadRes indexes anns[0] unconditionally -> IndexError on [].
        # Zero mAP is the correct value for "no predictions" anyway.
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_coco_path)
        # pycocotools needs results as a file or list of dicts with int category_id
        results = [{"image_id": d["image_id"], "category_id": int(d["category_id"]),
                    "bbox": d["bbox"], "score": float(d["score"])} for d in dets]
        tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        try:
            json.dump(results, tf); tf.close()
            coco_dt = coco_gt.loadRes(tf.name)
        finally:
            os.unlink(tf.name)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}
