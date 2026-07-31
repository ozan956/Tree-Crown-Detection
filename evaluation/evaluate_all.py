import os, csv
from evaluation.io_utils import load_gt, load_dets, dets_by_image, load_points
from evaluation.matching import match_one_to_one_iou, match_one_to_one_points
from evaluation.metrics import pr_f1, coco_map
from evaluation.paper_protocol import paper_recall_boxes, paper_recall_points

DET_FILES = {
    "swin": "data/dets/swin.json",
    "cascade_rcnn": "data/dets/cascade_rcnn.json",
    "detr": "data/dets/detr.json",
    "faster_rcnn": "data/dets/faster_rcnn.json",
    "yolo": "data/dets/yolo.json",
    "WBF": "data/wbf/best_fuse.json",
}

def _strict_box(dets_by_img, gt, iou_thr=0.5):
    tp = fp = fn = 0
    keys = set(dets_by_img) | set(gt)
    for iid in keys:
        preds = [b for (b, s) in dets_by_img.get(iid, [])]
        t, f, n = match_one_to_one_iou(preds, gt.get(iid, []), iou_thr)
        tp += t; fp += f; fn += n
    return pr_f1(tp, fp, fn)

def _strict_points(points_by_image, gt, gap=10.0):
    tp = fp = fn = 0
    keys = set(points_by_image) | set(gt)
    for iid in keys:
        t, f, n = match_one_to_one_points(points_by_image.get(iid, []), gt.get(iid, []), gap)
        tp += t; fp += f; fn += n
    return pr_f1(tp, fp, fn)

def build_table(score_thr=0.8, gap=10.0):
    gt = load_gt()
    rows = []
    for method, path in DET_FILES.items():
        dets = load_dets(path)
        dbi = dets_by_image(dets, score_thr)
        paper = paper_recall_boxes(dbi, gt, gap)
        strict = _strict_box(dbi, gt, 0.5)
        cm = coco_map("data/annotations.coco.json", [d for d in dets if d["score"] >= score_thr])
        rows.append({"method": method, "paper_recall": paper["recall"],
                     "strict_recall": strict["recall"], "strict_precision": strict["precision"],
                     "strict_f1": strict["f1"], "AP": cm["AP"], "AP50": cm["AP50"]})
    # integrated (points)
    pts = load_points()
    paper = paper_recall_points(pts, gt, gap)
    strict = _strict_points(pts, gt, gap)
    rows.append({"method": "integrated", "paper_recall": paper["recall"],
                 "strict_recall": strict["recall"], "strict_precision": strict["precision"],
                 "strict_f1": strict["f1"], "AP": None, "AP50": None})
    return rows

def write_table(rows, md_path="results/table.md", csv_path="results/table.csv"):
    os.makedirs("results", exist_ok=True)
    cols = ["method","paper_recall","strict_recall","strict_precision","strict_f1","AP","AP50"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    def fmt(v): return "n/a" if v is None else f"{v:.3f}"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r["method"] if c=="method" else fmt(r[c]) for c in cols) + " |\n")

def main():
    rows = build_table()
    write_table(rows)
    for r in rows:
        print(r)

if __name__ == "__main__":
    main()
