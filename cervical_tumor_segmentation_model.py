# Cervical tumor segmentation visualization and preview from dataset masks (no neural net).
# Uses ground-truth masks for voxelizations. Supports T2-weighted MRI in NIfTI or .npy.
# Single file: --input path/to/volume.nii --mask path/to/mask.nii  |  Batch: --data-dir <dir>.
# Saves to results/{DDMMYY-HHMMSS}/: data/ (segmentations + per-case metadata.json), images/ (previews).
#
# Dataset: MVT-Net cervical tumor segmentation (T2-weighted MRI, axial/sagittal/coronal views).
# Usage:
#   python cervical_tumor_segmentation_model.py -i volume.nii.gz --mask mask.nii.gz
#   python cervical_tumor_segmentation_model.py --data-dir data [--max-cases N]
#   python cervical_tumor_segmentation_model.py -i volume.npy --mask mask.npy

from __future__ import annotations

import json
import os
import argparse
from datetime import datetime
import numpy as np

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    nib = None
    HAS_NIBABEL = False

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


# Segmentation labels: 0=background, 1=tumor (optional 2=normal cervix/tissue if mask has it)
LABEL_BACKGROUND = 0
LABEL_TUMOR = 1
LABEL_NORMAL_TISSUE = 2


def _is_nifti(path):
    if path is None:
        return False
    p = path.lower()
    return p.endswith(".nii") or p.endswith(".nii.gz")


def load_volume(path):
    """
    Load a 3D MRI volume from NIfTI (.nii, .nii.gz) or NumPy (.npy).
    Returns (H, W, D) float32 array.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Volume not found: {path}")
    if _is_nifti(path):
        if not HAS_NIBABEL:
            raise ImportError(
                "nibabel is required for NIfTI files. pip install nibabel"
            )
        img = nib.load(path)
        vol = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
    else:
        vol = np.load(path).astype(np.float32).squeeze()
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape} from {path}")
    return vol


def load_mask(mask_path, target_shape=None):
    """
    Load a segmentation mask from NIfTI or .npy.
    Returns (H, W, D) int32: 0=background, 1=tumor; values > 1 mapped to LABEL_NORMAL_TISSUE if present.
    If target_shape is given and mask shape differs, resize with nearest-neighbor (requires scipy).
    """
    mask_path = os.path.abspath(mask_path)
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if _is_nifti(mask_path):
        if not HAS_NIBABEL:
            raise ImportError(
                "nibabel is required for NIfTI files. pip install nibabel"
            )
        img = nib.load(mask_path)
        m = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
    else:
        m = np.load(mask_path).astype(np.float32).squeeze()
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {m.shape} from {mask_path}")
    # Binarize or map: 0 -> 0, any positive -> 1 (tumor); optionally keep 2 for normal tissue
    seg = np.round(m).astype(np.int32)
    seg = np.clip(seg, 0, 2)
    # Common convention: 0=bg, 1=tumor; if mask has 0/1 only we're done; if 0/1/2 use as-is
    if np.any(seg > 1):
        pass  # keep 2 as normal tissue
    else:
        seg = (seg > 0).astype(np.int32)  # 0 or 1 only
    if target_shape is not None and seg.shape != target_shape:
        if not HAS_SCIPY:
            raise RuntimeError(
                "Mask shape != volume shape and scipy not available for resize."
            )
        zoom = [target_shape[i] / seg.shape[i] for i in range(3)]
        seg = ndi.zoom(seg.astype(np.float32), zoom, order=0)
        seg = np.round(seg).astype(np.int32)
        seg = np.clip(seg, 0, 2)
    return seg


def build_volume_metadata(
    image_3d,
    segmentation_3d,
    case_name,
    image_path=None,
    mask_path=None,
    top_n_slices=15,
):
    """
    Build a JSON-serializable metadata dict for one volume.
    segmentation_3d: (H, W, D) with 0=background, 1=tumor, optional 2=normal tissue.
    """
    H, W, D = image_3d.shape
    n_voxels = int(H * W * D)
    n_tumor = int(np.sum(segmentation_3d == LABEL_TUMOR))
    n_normal = int(np.sum(segmentation_3d == LABEL_NORMAL_TISSUE))
    slice_sums = np.sum(segmentation_3d == LABEL_TUMOR, axis=(0, 1))
    n_take = min(top_n_slices, D)
    top_slice_indices = np.argsort(slice_sums)[-n_take:][::-1].tolist()
    meta = {
        "case_name": case_name,
        "shape": {"height": int(H), "width": int(W), "depth": int(D)},
        "n_slices": int(D),
        "n_voxels": n_voxels,
        "segmentation_labels": {"0": "background", "1": "tumor", "2": "normal_tissue"},
        "n_tumor_voxels": n_tumor,
        "n_normal_tissue_voxels": n_normal,
        "tumor_volume_fraction": float(n_tumor / n_voxels) if n_voxels > 0 else 0.0,
        "top_slices_by_tumor_area": top_slice_indices,
        "segmentation_shape": [int(H), int(W), int(D)],
        "segmentation_dtype": str(segmentation_3d.dtype),
    }
    if image_path is not None:
        meta["image_path"] = os.path.abspath(image_path)
    if mask_path is not None:
        meta["mask_path"] = os.path.abspath(mask_path)
    return meta


def save_preview_images(
    image_3d,
    segmentation_3d,
    images_dir,
    case_name,
    n_slices=15,
):
    """
    Save previews: (1) three-view (axial, sagittal, coronal) at mid-slice and tumor-hot slice;
    (2) top N axial slices by tumor area with raw + tumor overlay (like breast script).
    segmentation_3d: (H, W, D) with 0=background, 1=tumor, optional 2=normal tissue.
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

    # Slices with largest tumor area (axial = axis 2)
    slice_sums = np.sum(segmentation_3d == LABEL_TUMOR, axis=(0, 1))
    n_take = min(n_slices, D)
    top_indices = np.argsort(slice_sums)[-n_take:][::-1].tolist()
    cx, cy, cz = H // 2, W // 2, D // 2
    tumor_hot_z = int(np.argmax(slice_sums)) if np.any(slice_sums > 0) else cz

    def draw_tumor_overlay(ax, img_slice, seg_slice):
        ax.imshow(img_slice, cmap="gray", origin="lower")
        tumor_mask = np.ma.masked_where(seg_slice != LABEL_TUMOR, seg_slice)
        ax.imshow(tumor_mask, cmap="Reds", origin="lower", alpha=0.6, vmin=0, vmax=2)

    def draw_tumor_and_normal_overlay(ax, img_slice, seg_slice):
        ax.imshow(img_slice, cmap="gray", origin="lower")
        normal_mask = np.ma.masked_where(seg_slice != LABEL_NORMAL_TISSUE, seg_slice)
        ax.imshow(normal_mask, cmap="Blues", origin="lower", alpha=0.35, vmin=0, vmax=2)
        tumor_mask = np.ma.masked_where(seg_slice != LABEL_TUMOR, seg_slice)
        ax.imshow(tumor_mask, cmap="Reds", origin="lower", alpha=0.6, vmin=0, vmax=2)

    # ----- Three-view: axial, sagittal, coronal (mid-slice and tumor-hot) -----
    for view_name, z_slice in [("mid", cz), ("tumor_hot", tumor_hot_z)]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # Axial (x-y at z)
        ax_slice = img[:, :, z_slice]
        seg_slice = segmentation_3d[:, :, z_slice]
        axes[0].imshow(ax_slice, cmap="gray", origin="lower")
        draw_tumor_overlay(axes[0], ax_slice, seg_slice)
        axes[0].set_title("Axial (x-y)", fontsize=12)
        axes[0].axis("off")
        # Sagittal (y-z at x)
        sag_slice = img[cx, :, :]
        sag_seg = segmentation_3d[cx, :, :]
        axes[1].imshow(sag_slice, cmap="gray", origin="lower")
        draw_tumor_overlay(axes[1], sag_slice, sag_seg)
        axes[1].set_title("Sagittal (y-z)", fontsize=12)
        axes[1].axis("off")
        # Coronal (x-z at y)
        cor_slice = img[:, cy, :]
        cor_seg = segmentation_3d[:, cy, :]
        axes[2].imshow(cor_slice, cmap="gray", origin="lower")
        draw_tumor_overlay(axes[2], cor_slice, cor_seg)
        axes[2].set_title("Coronal (x-z)", fontsize=12)
        axes[2].axis("off")
        plt.suptitle(f"{case_name} — three-view ({view_name})", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(images_dir, f"{case_name}_three_view_{view_name}.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    # ----- Per-slice axial: raw | tumor overlay | tumor + normal (if any) -----
    has_normal = np.any(segmentation_3d == LABEL_NORMAL_TISSUE)
    for z in top_indices:
        if z >= D:
            continue
        img_slice = img[:, :, z]
        seg_slice = segmentation_3d[:, :, z]
        n_panels = 3 if has_normal else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
        axes = np.atleast_1d(axes)
        axes[0].imshow(img_slice, cmap="gray", origin="lower")
        axes[0].set_title("T2w MRI", fontsize=12)
        axes[0].axis("off")
        draw_tumor_overlay(axes[1], img_slice, seg_slice)
        axes[1].set_title("Tumor overlay", fontsize=12)
        axes[1].axis("off")
        if has_normal:
            draw_tumor_and_normal_overlay(axes[2], img_slice, seg_slice)
            axes[2].set_title("Tumor + normal tissue", fontsize=12)
            axes[2].axis("off")
        plt.suptitle(f"{case_name} — axial slice z={z}", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(images_dir, f"{case_name}_tumor_preview_slice_z{z:03d}.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    # ----- One combined figure: 15 slices (largest tumor area) in 3x5 grid -----
    n_show = 15
    top_15 = [z for z in top_indices if z < D][:n_show]
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, z in enumerate(top_15):
        row, col = idx // ncols, idx % ncols
        img_slice = img[:, :, z]
        seg_slice = segmentation_3d[:, :, z]
        ax = axes[row, col]
        draw_tumor_overlay(ax, img_slice, seg_slice)
        ax.set_title(f"z={z}", fontsize=10)
        ax.axis("off")
    for idx in range(len(top_15), nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis("off")
    plt.suptitle(
        f"{case_name} — cervical tumor preview (top {len(top_15)} slices by tumor area)",
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
    n_slices=15,
):
    """Load image and mask, save segmentation and previews. Returns segmentation (0=bg, 1=tumor, optional 2=normal)."""
    if case_name is None:
        base = os.path.basename(image_path)
        case_name = base.replace(".nii.gz", "").replace(".nii", "").replace(".npy", "")
    image_3d = load_volume(image_path)
    seg = load_mask(mask_path, target_shape=image_3d.shape)
    if seg.shape != image_3d.shape:
        raise ValueError(
            f"Mask shape {seg.shape} != image shape {image_3d.shape} for {case_name}. "
            "Use target_shape or check orientations."
        )
    n_tumor = int(np.sum(seg == LABEL_TUMOR))
    if n_tumor == 0:
        print(f"  Warning: mask has no tumor voxels for {case_name}. Check {mask_path}")
    save_preview_images(image_3d, seg, images_dir_out, case_name, n_slices=n_slices)
    seg_path = os.path.join(data_dir_out, f"{case_name}_segmentation.npy")
    np.save(seg_path, seg)
    metadata = build_volume_metadata(
        image_3d, seg, case_name, image_path=image_path, mask_path=mask_path
    )
    meta_path = os.path.join(data_dir_out, f"{case_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return seg


def _infer_mask_path(image_path, masks_dir=None):
    """Infer mask path from image path: same basename in masks/ or _mask suffix."""
    base = os.path.basename(image_path)
    name = base.replace(".nii.gz", "").replace(".nii", "").replace(".npy", "")
    candidates = [
        f"{name}_mask.nii.gz",
        f"{name}_mask.nii",
        f"{name}_label.nii.gz",
        f"{name}_label.nii",
        f"{name}.nii.gz",
        f"{name}.nii",
        f"{name}_mask.npy",
        f"{name}.npy",
    ]
    if masks_dir and os.path.isdir(masks_dir):
        for c in candidates:
            candidate = os.path.join(masks_dir, c)
            if os.path.isfile(candidate):
                return candidate
    parent = os.path.dirname(os.path.abspath(image_path))
    for c in [
        f"{name}_mask.nii.gz",
        f"{name}_mask.nii",
        f"{name}_label.nii.gz",
        f"{name}_label.nii",
        f"{name}_mask.npy",
    ]:
        candidate = os.path.join(parent, c)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(parent, "masks", base)


def segment_single_file(
    image_path,
    mask_path=None,
    results_dir=None,
):
    """
    Visualize and save segmentation for one MRI volume. Mask path inferred if not given
    (e.g. same dir with _mask suffix, or .../masks/<same_name>).
    Saves to results/{DDMMYY-HHMMSS}/ (or results_dir). Returns (segmentation_3d, results_dir).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input file not found: {image_path}")
    if mask_path is None:
        parent = os.path.dirname(os.path.abspath(image_path))
        masks_dir = os.path.join(parent, "masks")
        mask_path = _infer_mask_path(image_path, masks_dir)
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"Mask not found: {mask_path}. "
            "Place masks in .../masks/ with same name or pass --mask explicitly."
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
    print(f"Using dataset masks (no model). Image: {image_path}  Mask: {mask_path}")

    seg = _run_one(image_path, mask_path, data_dir_out, images_dir_out, n_slices=15)
    print(f"Done. Segmentation and metadata saved to {data_dir_out}/")
    return seg, results_dir


def _discover_per_case_pairs(data_dir):
    """
    Discover (image_path, mask_path, case_name) when data_dir contains one subdir per case,
    each with image.nii.gz / image.nii / image.npy and label.nii.gz / label.nii / label.npy.
    Returns list of tuples or empty list if layout not found.
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        return []
    image_names = ["image.nii.gz", "image.nii", "image.npy"]
    label_names = ["label.nii.gz", "label.nii", "label.npy"]
    pairs = []
    for name in sorted(os.listdir(data_dir)):
        subdir = os.path.join(data_dir, name)
        if not os.path.isdir(subdir):
            continue
        image_path = None
        for iname in image_names:
            p = os.path.join(subdir, iname)
            if os.path.isfile(p):
                image_path = p
                break
        if image_path is None:
            continue
        mask_path = None
        for lname in label_names:
            p = os.path.join(subdir, lname)
            if os.path.isfile(p):
                mask_path = p
                break
        if mask_path is None:
            continue
        pairs.append((image_path, mask_path, name))
    return pairs


def run_segmentation(
    data_dir="data",
    max_cases=None,
    results_dir=None,
    n_slices=15,
):
    """
    Process all volumes in data_dir. Supports two layouts:
    (1) data_dir/images/ and data_dir/masks/ (or labels/) with matching filenames.
    (2) data_dir/<id>/image.nii.gz and data_dir/<id>/label.nii.gz (one subdir per case).
    Saves to results/{DDMMYY-HHMMSS}/ (or results_dir): data/, images/.
    Returns results_dir.
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Try per-case subdir layout first (e.g. cervical_dataset/50/image.nii.gz, .../label.nii.gz)
    per_case = _discover_per_case_pairs(data_dir)
    if per_case:
        if max_cases is not None:
            per_case = per_case[:max_cases]
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
        print(
            f"Using per-case layout (image.nii.gz + label.nii.gz per subdir). Processing {len(per_case)} case(s)."
        )
        for image_path, mask_path, case_name in per_case:
            _run_one(
                image_path,
                mask_path,
                data_dir_out,
                images_dir_out,
                case_name=case_name,
                n_slices=n_slices,
            )
        print(f"Done. Segmentations and metadata saved to {data_dir_out}/")
        return results_dir

    # Flat layout: images/ + masks/ (or image/ + labels/)
    images_src = os.path.join(data_dir, "images")
    if not os.path.isdir(images_src):
        images_src = os.path.join(data_dir, "image")
    if not os.path.isdir(images_src):
        raise FileNotFoundError(
            f"Images directory not found under {data_dir}. "
            "Expect 'images/' or 'image/', or per-case subdirs with image.nii.gz and label.nii.gz."
        )
    masks_dir = os.path.join(data_dir, "masks")
    if not os.path.isdir(masks_dir):
        masks_dir = os.path.join(data_dir, "labels")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(
            f"Masks directory not found: {masks_dir} (or 'labels/')."
        )

    exts = (".nii.gz", ".nii", ".npy")
    image_files = []
    for f in sorted(os.listdir(images_src)):
        if any(f.endswith(e) for e in exts):
            image_files.append(f)
    if not image_files:
        raise FileNotFoundError(
            f"No volume files (.nii, .nii.gz, .npy) in {images_src}"
        )

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
        base = fname.replace(".nii.gz", "").replace(".nii", "").replace(".npy", "")
        image_path = os.path.join(images_src, fname)
        mask_path = None
        for mname in [
            f"{base}_mask.nii.gz",
            f"{base}_mask.nii",
            f"{base}.nii.gz",
            f"{base}.nii",
            f"{base}_mask.npy",
            f"{base}.npy",
        ]:
            candidate = os.path.join(masks_dir, mname)
            if os.path.isfile(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            print(f"  Skip {fname}: no matching mask in {masks_dir}")
            continue
        _run_one(
            image_path,
            mask_path,
            data_dir_out,
            images_dir_out,
            case_name=base,
            n_slices=n_slices,
        )

    print(f"Done. Segmentations and metadata saved to {data_dir_out}/")
    return results_dir


def main():
    parser = argparse.ArgumentParser(
        description="Cervical tumor segmentation visualization from dataset masks (no neural net). "
        "T2-weighted MRI; supports NIfTI and .npy. Single file (--input, --mask) or batch (--data-dir). "
        "Saves to results/{DDMMYY-HHMMSS}/."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a single MRI volume (.nii, .nii.gz, or .npy). Mask inferred from path if --mask not set.",
    )
    parser.add_argument(
        "--mask",
        "-m",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to segmentation mask (same format as volume). Required if not inferrable from --input.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset directory with images/ and masks/ (or labels/). Used when --input is not set.",
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
        help="Output directory. Default: results/DDMMYY-HHMMSS.",
    )
    parser.add_argument(
        "--n-slices",
        type=int,
        default=15,
        help="Number of top tumor slices to include in preview (default: 15).",
    )
    args = parser.parse_args()

    if args.input is not None:
        segment_single_file(
            image_path=args.input,
            mask_path=args.mask,
            results_dir=args.output_dir,
        )
    elif args.data_dir is not None:
        run_segmentation(
            data_dir=args.data_dir,
            max_cases=args.max_cases,
            results_dir=args.output_dir,
            n_slices=args.n_slices,
        )
    else:
        parser.error(
            "Provide either --input <volume> (and optionally --mask) or --data-dir <dir>."
        )


if __name__ == "__main__":
    main()
