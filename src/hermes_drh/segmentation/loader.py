"""
Segmentation loading for FDTD brain simulation.

Load BraTS-style segmentation from NIfTI file or from 4 modalities (FLAIR, T1, T1CE, T2)
via U-Net. Used by the main engine and by benchmark mode.

Pipeline contract (anatomy-agnostic): the pipeline expects a loader that returns
  (labels_3d, output_base, t_end_segmentation_or_None)
where labels_3d is int32 3D array of tissue labels, output_base is a string for
output filenames, and the third value is optional timing (or None) for segmentation.
Brain uses load_segmentation_and_output_base; breast/cervix use anatomy-specific loaders.
"""

import os
import time

import numpy as np
from scipy import ndimage

try:
    import nibabel as nib
except ImportError:
    nib = None


def _find_modalities_in_dir(dir_path):
    """
    Find FLAIR, T1, T1CE, T2 NIfTI files in a directory.
    Supports: exact names (flair.nii, t1.nii, t1ce.nii, t2.nii) or
    BraTS-style (*_flair.nii, *_t1.nii, *_t1ce.nii, *_t2.nii).
    Returns (flair_path, t1_path, t1ce_path, t2_path).
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Modalities directory not found: {dir_path}")
    names = {}
    for f in os.listdir(dir_path):
        if not f.endswith(".nii") and not f.endswith(".nii.gz"):
            continue
        fp = os.path.join(dir_path, f)
        if not os.path.isfile(fp):
            continue
        low = f.lower()
        if low == "flair.nii" or low.endswith("_flair.nii"):
            names["flair"] = fp
        elif low == "t1ce.nii" or low.endswith("_t1ce.nii"):
            names["t1ce"] = fp
        elif (low == "t1.nii" or low.endswith("_t1.nii")) and "t1ce" not in low:
            names["t1"] = fp
        elif low == "t2.nii" or low.endswith("_t2.nii"):
            names["t2"] = fp
    for key in ("flair", "t1", "t1ce", "t2"):
        if key not in names:
            raise FileNotFoundError(
                f"Missing modality in {dir_path}: no file matching {key} "
                "(e.g. {key}.nii or *_t1ce.nii)"
            )
    return names["flair"], names["t1"], names["t1ce"], names["t2"]


def load_segmentation_for_benchmark(args):
    """
    Load segmentation (from --seg, --modalities, or --modalities-dir) for
    full-pipeline benchmark. Returns labels_3d (int32, 3D).
    """
    use_modalities = (args.modalities is not None) or (args.modalities_dir is not None)
    if use_modalities:
        if args.modalities_dir is not None:
            flair_path, t1_path, t1ce_path, t2_path = _find_modalities_in_dir(
                args.modalities_dir
            )
        else:
            flair_path, t1_path, t1ce_path, t2_path = tuple(args.modalities)
        from hermes_drh.segmentation.brain import run_segmentation_from_modalities

        labels_3d = run_segmentation_from_modalities(
            flair_path,
            t1_path,
            t1ce_path,
            t2_path,
            args.checkpoint,
            extend_with_normal_brain=not args.no_normal_brain,
        )
    else:
        seg_path = args.seg or os.environ.get(
            "BRAIN_SEGMENTATION_NII", "brain_segmentation.nii"
        )
        if not os.path.isfile(seg_path):
            raise FileNotFoundError(
                f"Segmentation file not found: {seg_path}. "
                "Use --seg path.nii, --modalities F T1 T1CE T2, or --modalities-dir DIR"
            )
        if nib is None:
            raise ImportError("nibabel is required to load segmentation NIfTI")
        img = nib.load(seg_path)
        labels_3d = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
        if labels_3d.ndim != 3:
            raise ValueError(f"Expected 3D segmentation, got shape {labels_3d.shape}")
        labels_3d = np.round(labels_3d).astype(np.int32)
        labels_3d = np.clip(labels_3d, 0, 4)
    return labels_3d


def load_segmentation_and_output_base(args, use_modalities):
    """
    Load segmentation and determine output_base.
    If use_modalities: resolve FLAIR/T1/T1CE/T2 paths, run U-Net segmentation,
    set output_base from modality/dir name. Otherwise: load from args.seg,
    set output_base from seg path.
    Returns (labels_3d, output_base, t_end_segmentation).
    """
    if nib is None and not use_modalities:
        raise ImportError("nibabel is required to load segmentation NIfTI")
    t_end_segmentation = None
    if use_modalities:
        if args.modalities_dir is not None:
            flair_path, t1_path, t1ce_path, t2_path = _find_modalities_in_dir(
                args.modalities_dir
            )
        else:
            flair_path, t1_path, t1ce_path, t2_path = tuple(args.modalities)
        from hermes_drh.segmentation.brain import run_segmentation_from_modalities

        labels_3d = run_segmentation_from_modalities(
            flair_path,
            t1_path,
            t1ce_path,
            t2_path,
            args.checkpoint,
            extend_with_normal_brain=not args.no_normal_brain,
        )
        if args.modalities_dir is not None:
            output_base = os.path.basename(os.path.normpath(args.modalities_dir))
        else:
            output_base = os.path.splitext(os.path.basename(flair_path))[0]
            if output_base.endswith(".nii"):
                output_base = os.path.splitext(output_base)[0]
        t_end_segmentation = time.perf_counter()
        return (labels_3d, output_base, t_end_segmentation)
    seg_path = args.seg or os.environ.get(
        "BRAIN_SEGMENTATION_NII", "brain_segmentation.nii"
    )
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(
            f"Segmentation file not found: {seg_path}. "
            "Use --seg path.nii, --modalities F T1 T1CE T2, or --modalities-dir DIR"
        )
    output_base = os.path.splitext(os.path.basename(seg_path))[0]
    if output_base.endswith(".nii"):
        output_base = os.path.splitext(output_base)[0]
    img = nib.load(seg_path)
    labels_3d = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
    if labels_3d.ndim != 3:
        raise ValueError(f"Expected 3D segmentation, got shape {labels_3d.shape}")
    labels_3d = np.round(labels_3d).astype(np.int32)
    labels_3d = np.clip(labels_3d, 0, 4)
    t_end_segmentation = time.perf_counter()
    return (labels_3d, output_base, t_end_segmentation)


def load_breast_segmentation_for_pipeline(args):
    """
    Load breast segmentation for the shared pipeline (contract: labels_3d, output_base, None).
    Uses args.seg (path to .npy) and args.max_dim for downsampling.
    Labels: 0=background, 1=tumor, 2=healthy breast.
    """
    seg_path = getattr(args, "seg", None) or os.environ.get(
        "BREAST_SEGMENTATION_NPY", "breast_segmentation.npy"
    )
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Breast segmentation not found: {seg_path}")
    labels_3d = np.load(seg_path).astype(np.int32)
    if labels_3d.ndim != 3:
        raise ValueError(f"Expected 3D segmentation, got shape {labels_3d.shape}")
    max_dim = getattr(args, "max_dim", 60)
    nx, ny, nz = labels_3d.shape
    if max(nx, ny, nz) > max_dim:
        scale = min(max_dim / nx, max_dim / ny, max_dim / nz, 1.0)
        labels_3d = ndimage.zoom(
            labels_3d.astype(np.float32),
            (scale, scale, scale),
            order=0,
            mode="nearest",
        )
        labels_3d = np.round(labels_3d).astype(np.int32)
        labels_3d = np.clip(labels_3d, 0, 2)
    output_base = f"breast_fdtd_{int(time.time())}"
    return (labels_3d, output_base, None)


def load_cervix_segmentation_for_pipeline(args):
    """
    Load cervix segmentation for the shared pipeline (contract: labels_3d, output_base, None).
    Uses args.seg (path to .npy or NIfTI) and args.max_dim for downsampling.
    Labels: 0=background, 1=tumor, 2=healthy cervix.
    """
    seg_path = getattr(args, "seg", None) or os.environ.get(
        "CERVIX_SEGMENTATION_PATH", "cervix_segmentation.npy"
    )
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Cervix segmentation not found: {seg_path}")
    if seg_path.endswith(".npy"):
        labels_3d = np.load(seg_path).astype(np.int32)
    else:
        if nib is None:
            raise ImportError("nibabel is required to load cervix NIfTI segmentation")
        img = nib.load(seg_path)
        labels_3d = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
        labels_3d = np.round(labels_3d).astype(np.int32)
        labels_3d = np.clip(labels_3d, 0, 2)
    if labels_3d.ndim != 3:
        raise ValueError(f"Expected 3D segmentation, got shape {labels_3d.shape}")
    max_dim = getattr(args, "max_dim", 60)
    nx, ny, nz = labels_3d.shape
    if max(nx, ny, nz) > max_dim:
        scale = min(max_dim / nx, max_dim / ny, max_dim / nz, 1.0)
        labels_3d = ndimage.zoom(
            labels_3d.astype(np.float32),
            (scale, scale, scale),
            order=0,
            mode="nearest",
        )
        labels_3d = np.round(labels_3d).astype(np.int32)
        labels_3d = np.clip(labels_3d, 0, 2)
    output_base = f"cervix_fdtd_{int(time.time())}"
    return (labels_3d, output_base, None)
