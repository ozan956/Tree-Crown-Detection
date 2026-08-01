# Data snapshot manifest

This directory is a vendored, read-only snapshot of the evaluation inputs used
by the Phase 1 rigorous-evaluation module. All files were copied verbatim
(no transformation) from the two untracked sibling upload folders below on
2026-07-31, and their record counts were verified against the fixed ground
truth for this project. Every downstream evaluation task reads exclusively
from `data/` — never from the `wbf_test-*` source folders.

Source roots (untracked, not committed to this repo):

- `Z1 = wbf_test-20260731T192532Z-1-001/wbf_test`
- `Z2 = wbf_test-20260731T192532Z-1-002/wbf_test`

| File in `data/`                     | Source path (relative to repo root)                                   | Record count |
|--------------------------------------|-------------------------------------------------------------------------|--------------|
| `annotations.coco.json`              | `$Z2/beril-ozan-cem-work/new_idea/_annotations.coco.json`                | 222 images, 13552 annotations |
| `dets/swin.json`                     | `$Z1/swin/bbox_swin.json`                                                | 13387 detections |
| `dets/cascade_rcnn.json`             | `$Z1/cascade-rcnn/bbox_rcnn.json`                                        | 12894 detections |
| `dets/detr.json`                     | `$Z1/detr/bbox_detr.json`                                                | 22200 detections |
| `dets/faster_rcnn.json`              | `$Z1/faster_rcnn/bbox_fasterrcnn.json`                                   | 12610 detections |
| `dets/yolo.json`                     | `$Z2/yolo/bbox_yolo.json`                                                | 14084 detections |
| `wbf/best_fuse.json`                 | `$Z1/beril-ozan-cem-work/new_idea/best_fuse.json`                        | 20846 detections |
| `integrated/points.json`             | `$Z1/beril-ozan-cem-work/new_idea/detected_trees_based_on_shape.json`    | 222 groups, 15811 points total |

## Verification

Counts were verified with the assertion script in Task 1, Step 2
(`.venv/bin/python` against these exact files). All assertions passed;
the script printed `data snapshot OK`.

Ground truth is fixed for this project: 222 images, 13552 boxes.
