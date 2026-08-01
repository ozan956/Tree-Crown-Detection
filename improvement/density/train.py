"""Train the density-map counter on VHRTrees train split (RTX 2070).

Run: .venv/bin/python -m improvement.density.train
Saves the best checkpoint to results/density_model.pt.

Loss: MSE on the density map (standard) plus a small count-consistency term so
the integral tracks the true count. Kept intentionally short (few epochs) — the
goal is a competent, standard baseline, not a tuned SOTA model.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from improvement.density.data import TreeDensityDataset
from improvement.density.model import DensityCSRNetLite

TRAIN_DIR = "wbf_test-20260731T192532Z-1-003/wbf_test/mmdetection/data/tree/train"
TRAIN_JSON = os.path.join(TRAIN_DIR, "_annotations.coco.json")
VAL_DIR = "wbf_test-20260731T192532Z-1-002/wbf_test/mmdetection/data/tree/valid"
VAL_JSON = os.path.join(VAL_DIR, "_annotations.coco.json")
CKPT = "results/density_model.pt"


def count_mae(model, loader, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for img, dm, cnt in loader:
            pred = model(img.to(device))
            pcount = pred.sum(dim=(1, 2, 3)).cpu().numpy()
            errs.extend(np.abs(pcount - cnt.numpy()))
    return float(np.mean(errs)) if errs else float("nan")


def main(epochs=15, batch=4, lr=1e-5, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tr = TreeDensityDataset(TRAIN_JSON, TRAIN_DIR, augment=True)
    va = TreeDensityDataset(VAL_JSON, VAL_DIR, augment=False) if os.path.exists(VAL_JSON) else None
    print(f"train n={len(tr)}" + (f" val n={len(va)}" if va else " (no val split)"))
    tl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(va, batch_size=batch, shuffle=False, num_workers=2) if va else None

    model = DensityCSRNetLite(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss(reduction="sum")

    os.makedirs("results", exist_ok=True)
    best = float("inf")
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for img, dm, cnt in tl:
            img, dm = img.to(device), dm.to(device)
            pred = model(img)
            # density MSE (sum over pixels, mean over batch) + count term
            loss = mse(pred, dm) / img.size(0)
            pcount = pred.sum(dim=(1, 2, 3))
            loss = loss + 0.01 * ((pcount - cnt.to(device)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        msg = f"epoch {ep+1}/{epochs} train_loss={tot/len(tl):.3f}"
        if vl:
            mae = count_mae(model, vl, device)
            msg += f" val_count_MAE={mae:.2f}"
            if mae < best:
                best = mae; torch.save(model.state_dict(), CKPT); msg += " [saved]"
        else:
            torch.save(model.state_dict(), CKPT)
        print(msg, flush=True)
    if not vl:
        print(f"saved final model to {CKPT}")
    else:
        print(f"best val_count_MAE={best:.2f}, checkpoint at {CKPT}")


if __name__ == "__main__":
    main()
