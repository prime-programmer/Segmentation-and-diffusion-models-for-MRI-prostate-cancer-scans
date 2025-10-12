#!/usr/bin/env python3
"""
SimpleITK-based QC overlay generator for nnU-Net outputs.

Creates quick visual checks (PNG overlays) showing MRI (gray),
prediction (filled cyan), and ground truth (magenta outline).

Default paths assume nnU-Net dataset folder structure.
"""

import os, argparse, glob
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------- Helper functions -------------------

def load_img(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return sitk.ReadImage(path)

def sitk_to_numpy(img):
    # Converts (x,y,z) SimpleITK to numpy (z,y,x)
    return sitk.GetArrayFromImage(img)

def normalize_for_display(img_sitk, p_lo=1, p_hi=99):
    """Normalize intensities to [0,1] for display."""
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    lo = np.percentile(arr, p_lo)
    hi = np.percentile(arr, p_hi)
    hi = max(hi, lo + 1e-5)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return arr

def choose_slices(mask_np, k=3):
    """Pick k informative slices based on mask extent; fallback to center."""
    idx = np.where(mask_np > 0)[0]
    if idx.size == 0:
        return [mask_np.shape[0] // 2]
    zmin, zmax = int(idx.min()), int(idx.max())
    zmid = (zmin + zmax) // 2
    return sorted({zmin, zmid, zmax})[:k]

def label_contour(mask_sitk):
    """Return contour voxels from label mask."""
    mask_u8 = sitk.Cast(mask_sitk > 0, sitk.sitkUInt8)
    return sitk.LabelContour(mask_u8)

def colorize_overlay(gray, mask=None, contour=None, title="", out_png=None):
    """Create and save an RGB overlay image."""
    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1)

    if mask is not None:
        m = mask.astype(bool)
        rgb[m, 1] = 1.0
        rgb[m, 2] = 1.0  # cyan fill

    if contour is not None:
        c = contour.astype(bool)
        rgb[c, 0] = 1.0
        rgb[c, 1] = 0.0
        rgb[c, 2] = 1.0  # magenta outline

    plt.figure(figsize=(5,5))
    plt.imshow(rgb, origin="lower")
    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ------------------- Core processing -------------------

def run_case(root, case, out_root, k_slices=3):
    """Generate QC overlays for one case."""
    print(f"Processing {case}...")

    # Locate MRI, prediction, and ground truth paths
    candidates = [
        os.path.join(root, "imagesTs", f"{case}_0000.nii.gz"),
        os.path.join(root, "imagesTr", f"{case}_0000.nii.gz"),
    ]
    img_path = next((p for p in candidates if os.path.exists(p)), None)
    if img_path is None:
        raise FileNotFoundError(f"No MRI found for {case}")

    pred_candidates = [
        os.path.join(root, "predictions_3d_fullres", f"{case}.nii.gz"),
        os.path.join(root, "predictions_3d_fullres", f"{case}_seg.nii.gz"),
    ]
    pred_path = next((p for p in pred_candidates if os.path.exists(p)), None)

    gt_candidates = [
        os.path.join(root, "labelsTs", f"{case}.nii.gz"),
        os.path.join(root, "labelsTs", f"{case}_seg.nii.gz"),
        os.path.join(root, "labelsTr", f"{case}.nii.gz"),
        os.path.join(root, "labelsTr", f"{case}_seg.nii.gz"),
    ]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)

    # Load and normalize MRI
    img_sitk = load_img(img_path)
    mri = normalize_for_display(img_sitk)

    # Prediction and GT
    pred_sitk = load_img(pred_path) if pred_path else None
    gt_sitk = load_img(gt_path) if gt_path else None
    pred = sitk_to_numpy(pred_sitk) if pred_sitk else None
    gt_contour = sitk_to_numpy(label_contour(gt_sitk)) if gt_sitk else None

    # Choose slices and overlay
    ref = pred if pred is not None else mri
    zs = choose_slices(ref, k=k_slices)
    case_dir = os.path.join(out_root, case)
    os.makedirs(case_dir, exist_ok=True)

    for z in zs:
        gray = mri[z]
        pm = pred[z] if pred is not None else None
        gc = gt_contour[z] if gt_contour is not None else None
        out_png = os.path.join(case_dir, f"{case}_z{z}.png")
        colorize_overlay(gray, mask=pm, contour=gc,
                         title=f"{case} z={z} (cyan=pred, magenta=GT)",
                         out_png=out_png)

    print(f"  Saved PNGs → {case_dir}")
    return case_dir


# ------------------- Entry point -------------------

def main():
    ap = argparse.ArgumentParser(description="Generate QC overlays using SimpleITK")
    ap.add_argument("--root", default="/space/local/cug/nnUNet_raw/Dataset001_PROSTATE",
                    help="Root of nnUNet dataset (default: %(default)s)")
    ap.add_argument("--out", default="qc_sitk_pngs", help="Output subfolder name")
    ap.add_argument("--case", help="Specific case ID (e.g. case_000)")
    ap.add_argument("--k", type=int, default=3, help="Slices per case")
    ap.add_argument("--max_cases", type=int, default=16, help="Limit when running all cases")
    args = ap.parse_args()

    out_root = os.path.join(args.root, args.out)
    os.makedirs(out_root, exist_ok=True)

    # Run one case or multiple
    if args.case:
        run_case(args.root, args.case, out_root, k_slices=args.k)
    else:
        cases = sorted(glob.glob(os.path.join(args.root, "imagesTs", "case_*_0000.nii.gz")))
        if not cases:
            cases = sorted(glob.glob(os.path.join(args.root, "imagesTr", "case_*_0000.nii.gz")))
        cases = [os.path.basename(c).replace("_0000.nii.gz", "") for c in cases]
        if args.max_cases and len(cases) > args.max_cases:
            cases = cases[:args.max_cases]
        for c in cases:
            try:
                run_case(args.root, c, out_root, k_slices=args.k)
            except Exception as e:
                print(f"[WARN] {c}: {e}")
        print(f"✅ All QC PNGs saved under: {out_root}")

if __name__ == "__main__":
    main()
