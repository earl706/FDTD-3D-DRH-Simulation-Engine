# Breast tumor segmentation from dataset masks (no neural net).
# Uses ground-truth masks for voxelizations. Supports ISPY1 and BreastDM.
# Single file: --input path/to/image.npy  |  Batch: --data-dir data/ISPY1 (default).
# Saves to results/{DDMMYY-HHMMSS}/: data/ (segmentations + per-case metadata.json), images/ (previews).
# Usage:
#   python breast_tumor_segmentation_model.py -i data/ISPY1/images_std/ispy1_0.npy
#   python breast_tumor_segmentation_model.py --data-dir data/ISPY1 [--max-cases N]
#   python breast_tumor_segmentation_model.py --data-dir data/BreastDM [--max-cases N]
#   python breast_tumor_segmentation_model.py -i data/BreastDM/images/dm_0.npy

from __future__ import annotations

import json
import os
import argparse
from datetime import datetime
import numpy as np

try:
    import scipy.ndimage as ndi

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_breast_volume_for_preview(npy_path):
    """
    Load a preprocessed breast volume from .npy for preview only.
    Returns image_3d (H, W, D) for display.
    """
    vol = np.load(npy_path).astype(np.float32)
    if vol.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {vol.shape}")
    # (D, C, H, W) -> (C, H, W, D)
    vol = np.transpose(vol, (1, 2, 3, 0))
    C = vol.shape[0]
    image_3d = vol[0] if C >= 1 else vol[0:1].squeeze(0)
    return np.asarray(image_3d, dtype=np.float32)


def load_breast_mask(mask_npy_path, target_shape=None):
    """
    Load a binary mask from .npy. ISPY1 saves (D, 1, H, W); also supports 3D (D, H, W) or (H, W, D).
    Returns segmentation (H, W, D) int32 0/1. If target_shape given, resizes with scipy.
    """
    m = np.load(mask_npy_path).astype(np.float32)
    if m.ndim == 4:
        # (D, C, H, W) -> (C, H, W, D) -> take first channel -> (H, W, D)
        m = np.transpose(m, (1, 2, 3, 0))
        m = m[0] if m.shape[0] >= 1 else m.max(axis=0)
    elif m.ndim == 3:
        # 3D: could be (H, W, D) or (D, H, W). Align to (H, W, D) using target_shape if given.
        if target_shape is not None:
            H, W, D = target_shape
            if m.shape == (D, H, W):
                m = np.transpose(m, (1, 2, 0))  # (D, H, W) -> (H, W, D)
            elif m.shape != (H, W, D):
                # try (D, W, H) or other permutation
                if m.shape == (D, W, H):
                    m = np.transpose(m, (2, 1, 0))  # -> (H, W, D)
    else:
        raise ValueError(f"Expected 3D or 4D mask, got shape {m.shape}")
    seg = (m > 0).astype(np.int32)
    if target_shape is not None and seg.shape != target_shape:
        if not HAS_SCIPY:
            raise RuntimeError(
                "Mask shape != image shape and scipy not available for resize."
            )
        zoom = [target_shape[i] / seg.shape[i] for i in range(3)]
        seg = ndi.zoom(seg.astype(np.float32), zoom, order=0)
        seg = (seg > 0.5).astype(np.int32)
    return seg


# Segmentation labels: 0=background, 1=tumor, 2=healthy breast tissue
LABEL_BACKGROUND = 0
LABEL_TUMOR = 1
LABEL_HEALTHY_BREAST = 2


def get_breast_mask_from_volume(image_3d, percentile_low=15):
    """
    Compute a binary breast/tissue mask from the 3D image (H, W, D).
    Uses intensity percentile threshold and hole-filling; keeps largest connected component.
    Returns boolean array (H, W, D).
    """
    if not HAS_SCIPY:
        return np.zeros(image_3d.shape, dtype=bool)
    vol = np.asarray(image_3d, dtype=np.float32)
    valid = vol[np.isfinite(vol) & (vol > 0)]
    if valid.size == 0:
        return np.zeros(vol.shape, dtype=bool)
    thresh = np.percentile(valid, percentile_low)
    mask = (vol > thresh).astype(np.uint8)
    mask = ndi.binary_fill_holes(mask).astype(bool)
    labeled, num_features = ndi.label(mask)
    if num_features > 1:
        sizes = np.bincount(labeled.ravel())[1:]
        largest = np.argmax(sizes) + 1
        mask = labeled == largest
    return mask


def extend_segmentation_with_healthy_breast(seg_tumor_01, image_3d, percentile_low=15):
    """
    Extend binary tumor segmentation (0=bg, 1=tumor) to 3-class: 0=background, 1=tumor, 2=healthy breast.
    Voxels inside the breast mask that are currently 0 become LABEL_HEALTHY_BREAST.
    Returns int32 (H, W, D).
    """
    breast_mask = get_breast_mask_from_volume(image_3d, percentile_low=percentile_low)
    out = np.asarray(seg_tumor_01, dtype=np.int32).copy()
    out[breast_mask & (out == LABEL_BACKGROUND)] = LABEL_HEALTHY_BREAST
    return out


def build_volume_metadata(
    image_3d,
    segmentation_3d,
    case_name,
    image_path=None,
    mask_path=None,
    top_n_slices=10,
):
    """
    Build a JSON-serializable metadata dict for one volume.
    segmentation_3d: (H, W, D) with 0=background, 1=tumor, 2=healthy breast.
    """
    H, W, D = image_3d.shape
    n_voxels = int(H * W * D)
    n_tumor_voxels = int(np.sum(segmentation_3d == LABEL_TUMOR))
    n_healthy_voxels = int(np.sum(segmentation_3d == LABEL_HEALTHY_BREAST))
    slice_sums = np.sum(segmentation_3d == LABEL_TUMOR, axis=(0, 1))
    n_take = min(top_n_slices, D)
    top_slice_indices = np.argsort(slice_sums)[-n_take:][::-1].tolist()

    meta = {
        "case_name": case_name,
        "shape": {"height": int(H), "width": int(W), "depth": int(D)},
        "n_slices": int(D),
        "n_voxels": n_voxels,
        "segmentation_labels": {"0": "background", "1": "tumor", "2": "healthy_breast"},
        "n_tumor_voxels": n_tumor_voxels,
        "n_healthy_breast_voxels": n_healthy_voxels,
        "tumor_volume_fraction": (
            float(n_tumor_voxels / n_voxels) if n_voxels > 0 else 0.0
        ),
        "top_slices_by_tumor_area": top_slice_indices,
        "segmentation_shape": [int(H), int(W), int(D)],
        "segmentation_dtype": str(segmentation_3d.dtype),
    }
    if image_path is not None:
        meta["image_path"] = os.path.abspath(image_path)
    if mask_path is not None:
        meta["mask_path"] = os.path.abspath(mask_path)
    return meta


def save_preview_images(image_3d, segmentation_3d, images_dir, case_name, n_slices=10):
    """
    Save previews for the N slices with largest tumor area. Each slice: 3 panels side by side —
    non-segmented (raw MRI), tumor-highlighted, breast+tumor highlighted.
    segmentation_3d: (H, W, D) with 0=background, 1=tumor, 2=healthy breast.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping preview images.")
        return
    os.makedirs(images_dir, exist_ok=True)
    H, W, D = image_3d.shape
    img = np.asarray(image_3d, dtype=np.float32)
    if img.size > 0 and np.any(np.isfinite(img)):
        lo, hi = np.nanpercentile(img, [1, 99])
        if hi > lo:
            img = (img - lo) / (hi - lo)
        img = np.clip(img, 0, 1)

    # Slices with largest tumor area (descending)
    slice_sums = np.sum(segmentation_3d == LABEL_TUMOR, axis=(0, 1))
    n_take = min(n_slices, D)
    top_indices = np.argsort(slice_sums)[-n_take:][::-1].tolist()

    def draw_tumor_overlay(ax, img_slice, seg_slice):
        ax.imshow(img_slice, cmap="gray", origin="lower")
        tumor_mask = np.ma.masked_where(seg_slice != LABEL_TUMOR, seg_slice)
        ax.imshow(tumor_mask, cmap="Reds", origin="lower", alpha=0.6, vmin=0, vmax=2)

    def draw_breast_and_tumor_overlay(ax, img_slice, seg_slice):
        ax.imshow(img_slice, cmap="gray", origin="lower")
        healthy_mask = np.ma.masked_where(seg_slice != LABEL_HEALTHY_BREAST, seg_slice)
        ax.imshow(
            healthy_mask, cmap="Blues", origin="lower", alpha=0.35, vmin=0, vmax=2
        )
        tumor_mask = np.ma.masked_where(seg_slice != LABEL_TUMOR, seg_slice)
        ax.imshow(tumor_mask, cmap="Reds", origin="lower", alpha=0.6, vmin=0, vmax=2)

    # Individual slice images: 3 panels (non-segmented | tumor-highlighted | breast+tumor)
    for z in top_indices:
        if z >= D:
            continue
        img_slice = img[:, :, z]
        seg_slice = segmentation_3d[:, :, z]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_slice, cmap="gray", origin="lower")
        axes[0].set_title("Non-segmented", fontsize=12)
        axes[0].axis("off")
        draw_tumor_overlay(axes[1], img_slice, seg_slice)
        axes[1].set_title("Tumor highlighted", fontsize=12)
        axes[1].axis("off")
        draw_breast_and_tumor_overlay(axes[2], img_slice, seg_slice)
        axes[2].set_title("Breast + tumor highlighted", fontsize=12)
        axes[2].axis("off")
        plt.suptitle(f"{case_name} — axial slice z={z}", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(
            images_dir, f"{case_name}_tumor_preview_slice_z{z:03d}.png"
        )
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()

    # One combined image: 10 rows x 3 cols (each row = one slice: raw | tumor | breast+tumor)
    nrows = len(top_indices)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i, z in enumerate(top_indices):
        if z >= D:
            continue
        img_slice = img[:, :, z]
        seg_slice = segmentation_3d[:, :, z]
        axes[i, 0].imshow(img_slice, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"z={z} — Non-segmented", fontsize=10)
        axes[i, 0].axis("off")
        draw_tumor_overlay(axes[i, 1], img_slice, seg_slice)
        axes[i, 1].set_title(f"z={z} — Tumor", fontsize=10)
        axes[i, 1].axis("off")
        draw_breast_and_tumor_overlay(axes[i, 2], img_slice, seg_slice)
        axes[i, 2].set_title(f"z={z} — Breast + tumor", fontsize=10)
        axes[i, 2].axis("off")
    plt.suptitle(
        f"{case_name} — 10 slices with largest tumor: non-segmented | tumor | breast+tumor",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(images_dir, f"{case_name}_tumor_preview.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved previews to {images_dir}/")


def _run_one(
    image_path,
    mask_path,
    data_dir_out,
    images_dir_out,
    case_name=None,
):
    """Load image and mask, extend with healthy breast, save 3-class segmentation and previews. Returns segmentation (0=bg, 1=tumor, 2=healthy breast)."""
    if case_name is None:
        case_name = os.path.splitext(os.path.basename(image_path))[0]
    image_3d = load_breast_volume_for_preview(image_path)
    target_shape = image_3d.shape
    seg_tumor = load_breast_mask(mask_path, target_shape=target_shape)
    if seg_tumor.shape != target_shape:
        raise ValueError(
            f"Mask shape {seg_tumor.shape} != image shape {target_shape} for {case_name}. "
            "Check mask/image layout (H,W,D vs D,H,W)."
        )
    n_tumor = int(np.sum(seg_tumor == 1))
    if n_tumor == 0:
        print(f"  Warning: mask has no tumor voxels for {case_name}. Check {mask_path}")
    # 0=background, 1=tumor, 2=healthy breast
    seg = extend_segmentation_with_healthy_breast(seg_tumor, image_3d)
    save_preview_images(image_3d, seg, images_dir_out, case_name)
    seg_path = os.path.join(data_dir_out, f"{case_name}_segmentation.npy")
    np.save(seg_path, seg)
    metadata = build_volume_metadata(
        image_3d, seg, case_name, image_path=image_path, mask_path=mask_path
    )
    meta_path = os.path.join(data_dir_out, f"{case_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return seg


def _infer_masks_dir_from_image_path(image_path):
    """If image is in .../images/ or .../images_std/, return .../masks/ and same basename."""
    abs_path = os.path.abspath(image_path)
    parent = os.path.dirname(abs_path)
    base = os.path.basename(abs_path)
    if os.path.basename(parent) in ("images", "images_std"):
        data_dir = os.path.dirname(parent)
        return os.path.join(data_dir, "masks", base)
    return os.path.join(parent, "masks", base)


def segment_single_file(
    npy_path,
    results_dir=None,
):
    """
    Use ground-truth mask for one MRI file. Mask path is inferred from image path
    (e.g. data/ISPY1/images_std/ispy1_0.npy -> data/ISPY1/masks/ispy1_0.npy).
    Saves to results/{DDMMYY-HHMMSS}/ (or results_dir). Returns (segmentation_3d, results_dir).
    """
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Input file not found: {npy_path}")
    mask_path = _infer_masks_dir_from_image_path(npy_path)
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"Mask not found: {mask_path}. Expect masks next to images (e.g. data/ISPY1/masks/<same_name>.npy)"
        )
    if results_dir is None:
        now = datetime.now()
        timestamp_str = now.strftime("%d%m%y-%H%M%S")
        results_dir = os.path.join("results", timestamp_str)
    data_dir_out = os.path.join(results_dir, "data")
    images_dir_out = os.path.join(results_dir, "images")
    os.makedirs(data_dir_out, exist_ok=True)
    os.makedirs(images_dir_out, exist_ok=True)
    print(f"Output directory: {results_dir}/")
    print(f"  Data (segmentations + metadata): {data_dir_out}/")
    print(f"  Images (previews): {images_dir_out}/")
    print(f"Using dataset masks (no model). Segmenting: {npy_path}")

    seg = _run_one(npy_path, mask_path, data_dir_out, images_dir_out)
    case_name = os.path.splitext(os.path.basename(npy_path))[0]
    print(f"Done. Segmentation and metadata saved to {data_dir_out}/")
    return seg, results_dir


def run_segmentation(
    data_dir="data/ISPY1",
    max_cases=None,
    results_dir=None,
):
    """
    Copy dataset masks as segmentations for all volumes in data_dir.
    Supports data/ISPY1 (uses images_std/ if present, else images/) and data/BreastDM (images/ only).
    Expects masks/ with same filenames. Saves to results/{DDMMYY-HHMMSS}/ (or results_dir if given):
    data/ (segmentations + per-case metadata.json), images/ (previews).
    Returns results_dir (path used for output).
    """
    images_src = os.path.join(data_dir, "images_std")
    if not os.path.isdir(images_src):
        images_src = os.path.join(data_dir, "images")
    if not os.path.isdir(images_src):
        raise FileNotFoundError(
            f"Neither {os.path.join(data_dir, 'images_std')} nor {os.path.join(data_dir, 'images')} found."
        )
    masks_dir = os.path.join(data_dir, "masks")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    image_files = sorted([f for f in os.listdir(images_src) if f.endswith(".npy")])
    if not image_files:
        raise FileNotFoundError(f"No .npy files in {images_src}")

    if max_cases is not None:
        image_files = image_files[:max_cases]

    if results_dir is None:
        now = datetime.now()
        timestamp_str = now.strftime("%d%m%y-%H%M%S")
        results_dir = os.path.join("results", timestamp_str)
    data_dir_out = os.path.join(results_dir, "data")
    images_dir_out = os.path.join(results_dir, "images")
    os.makedirs(data_dir_out, exist_ok=True)
    os.makedirs(images_dir_out, exist_ok=True)
    print(f"Output directory: {results_dir}/")
    print(f"  Data (segmentations + metadata): {data_dir_out}/")
    print(f"  Images (previews): {images_dir_out}/")
    print(f"Using dataset masks (no model). Processing {len(image_files)} case(s).")

    for fname in image_files:
        case_name = os.path.splitext(fname)[0]
        image_path = os.path.join(images_src, fname)
        mask_path = os.path.join(masks_dir, fname)
        if not os.path.isfile(mask_path):
            print(f"  Skip {fname}: mask not found at {mask_path}")
            continue
        _run_one(
            image_path,
            mask_path,
            data_dir_out,
            images_dir_out,
            case_name=case_name,
        )

    print(f"Done. Segmentations and metadata saved to {data_dir_out}/")
    return results_dir


def main():
    parser = argparse.ArgumentParser(
        description="Breast tumor segmentation from dataset masks (no neural net). "
        "Supports data/ISPY1 and data/BreastDM. Single file (--input) or batch (--data-dir). "
        "Saves to results/{DDMMYY-HHMMSS}/."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a single .npy MRI volume. Mask inferred from path (e.g. .../masks/<same_name>.npy).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/ISPY1",
        help="Dataset directory with images/ (and optionally images_std/ for ISPY1) and masks/. "
        "e.g. data/ISPY1 or data/BreastDM. Used when --input is not set.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Max number of cases when using --data-dir (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output directory (single-file or batch). Default: results/DDMMYY-HHMMSS.",
    )
    args = parser.parse_args()

    if args.input is not None:
        segment_single_file(
            npy_path=args.input,
            results_dir=args.output_dir,
        )
    else:
        run_segmentation(
            data_dir=args.data_dir,
            max_cases=args.max_cases,
            results_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
