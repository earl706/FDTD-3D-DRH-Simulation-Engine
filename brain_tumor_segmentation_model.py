# Brain tumor segmentation logic derived from "Brain Tumor Segmentation with 3D U-Net"
# (BrainTumorSegmentation-3DUNet-StreamlitApp), used under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
#
# Headless inference (no Streamlit). Loads 4 BraTS modalities and returns BraTS-style
# segmentation labels 0=background, 1=necrotic, 2=edema, 3=enhancing.

from __future__ import annotations

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy import ndimage

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Model architecture (from BrainTumorSegmentation-3DUNet-StreamlitApp)
# ---------------------------------------------------------------------------


class Conv3D_Block(nn.Module):
    def __init__(
        self,
        inp_feat,
        out_feat,
        kernel=3,
        stride=1,
        padding=1,
        residual=None,
        dropout_rate=0.2,
    ):
        super(Conv3D_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                inp_feat,
                out_feat,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_feat,
                out_feat,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )
        self.residual = residual
        if self.residual == "conv" and inp_feat != out_feat:
            self.residual_upsampler = nn.Conv3d(
                inp_feat, out_feat, kernel_size=1, bias=False
            )
        else:
            self.residual_upsampler = None

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.residual == "conv":
            if self.residual_upsampler is not None:
                res = self.residual_upsampler(res)
            return out + res
        return out


class Deconv3D_Block(nn.Module):
    def __init__(self, inp_feat, out_feat, kernel=2, stride=2, padding=0):
        super(Deconv3D_Block, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(
                inp_feat,
                out_feat,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.deconv(x)


class UNet3D_BraTS(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        feat_channels=(16, 32, 64, 128, 256),
        residual="conv",
        dropout_rate=0.2,
    ):
        super(UNet3D_BraTS, self).__init__()
        self.num_classes = num_classes
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.pool4 = nn.MaxPool3d(2)
        self.conv_blk1 = Conv3D_Block(
            in_channels, feat_channels[0], residual=residual, dropout_rate=dropout_rate
        )
        self.conv_blk2 = Conv3D_Block(
            feat_channels[0],
            feat_channels[1],
            residual=residual,
            dropout_rate=dropout_rate,
        )
        self.conv_blk3 = Conv3D_Block(
            feat_channels[1],
            feat_channels[2],
            residual=residual,
            dropout_rate=dropout_rate,
        )
        self.conv_blk4 = Conv3D_Block(
            feat_channels[2],
            feat_channels[3],
            residual=residual,
            dropout_rate=dropout_rate,
        )
        self.conv_blk5 = Conv3D_Block(
            feat_channels[3],
            feat_channels[4],
            residual=residual,
            dropout_rate=dropout_rate,
        )
        self.dec_conv_blk4 = Conv3D_Block(
            feat_channels[3] + feat_channels[3],
            feat_channels[3],
            residual=residual,
            dropout_rate=dropout_rate,
        )
        self.dec_conv_blk3 = Conv3D_Block(
            feat_channels[2] + feat_channels[2],
            feat_channels[2],
            residual=residual,
            dropout_rate=dropout_rate,
        )
        self.dec_conv_blk2 = Conv3D_Block(
            feat_channels[1] + feat_channels[1],
            feat_channels[1],
            residual=residual,
            dropout_rate=dropout_rate,
        )
        self.dec_conv_blk1 = Conv3D_Block(
            feat_channels[0] + feat_channels[0],
            feat_channels[0],
            residual=residual,
            dropout_rate=dropout_rate,
        )
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])
        self.final_conv = nn.Conv3d(
            feat_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv_blk1(x)
        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)
        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)
        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)
        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        logits = self.final_conv(d_high1)
        if self.training:
            return logits
        return self.softmax(logits)


# ---------------------------------------------------------------------------
# Loading and inference (headless)
# ---------------------------------------------------------------------------


def load_model(checkpoint_path):
    """Load trained 3D U-Net from checkpoint. Returns (model, device)."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D_BraTS(
        in_channels=4,
        num_classes=4,
        feat_channels=[16, 32, 64, 128, 256],
        residual="conv",
        dropout_rate=0.2,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def load_patient_volume_from_paths(flair_path, t1_path, t1ce_path, t2_path):
    """Load 4 BraTS modalities from NIfTI paths. Returns (4, H, W, D) float32, order FLAIR,T1,T1CE,T2."""
    order = [
        ("flair", flair_path),
        ("t1", t1_path),
        ("t1ce", t1ce_path),
        ("t2", t2_path),
    ]
    volumes = []
    for name, path in order:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Modality {name} not found: {path}")
        vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        volumes.append(vol)
    return np.stack(volumes, axis=0).astype(np.float32)


def extract_patches(volume, patch_size, stride):
    """Extract patches from volume [C,H,W,D] with sliding window."""
    C, H, W, D = volume.shape
    ph, pw, pd = patch_size
    sh, sw, sd = stride
    patches = []
    coords = []
    for h in range(0, H - ph + 1, sh):
        for w in range(0, W - pw + 1, sw):
            for d in range(0, D - pd + 1, sd):
                patch = volume[:, h : h + ph, w : w + pw, d : d + pd]
                patches.append(patch)
                coords.append((h, w, d))
    return np.array(patches), coords


def reconstruct_volume(
    patch_preds, coords, volume_shape, patch_size, stride, num_classes
):
    """Reconstruct full volume prediction by averaging overlapping patch outputs."""
    _, H, W, D = volume_shape
    ph, pw, pd = patch_size
    sh, sw, sd = stride
    output_probs = np.zeros((num_classes, H, W, D), dtype=np.float32)
    count_map = np.zeros((H, W, D), dtype=np.float32)
    for pred, (h, w, d) in zip(patch_preds, coords):
        output_probs[:, h : h + ph, w : w + pw, d : d + pd] += pred
        count_map[h : h + ph, w : w + pw, d : d + pd] += 1.0
    count_map[count_map == 0] = 1.0
    output_probs /= count_map
    seg = np.argmax(output_probs, axis=0)
    return seg, output_probs


def predict_segmentation(
    model,
    device,
    volume,
    patch_size=(128, 128, 64),
    stride=(64, 64, 32),
    batch_size=2,
):
    """Run inference on (4,H,W,D) volume. Returns (segmentation, output_probs)."""
    patches, coords = extract_patches(volume, patch_size, stride)
    patches_tensor = torch.from_numpy(patches).to(device)
    all_preds = []
    batch_starts = list(range(0, len(patches_tensor), batch_size))
    it = batch_starts
    if tqdm is not None:
        it = tqdm(it, desc="Segmentation patches", unit="batch")
    with torch.no_grad():
        for batch_start in it:
            batch = patches_tensor[batch_start : batch_start + batch_size]
            preds = model(batch)
            preds = preds.cpu().numpy()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)
    segmentation, output_probs = reconstruct_volume(
        all_preds, coords, volume.shape, patch_size, stride, 4
    )
    return segmentation, output_probs


def select_slices_biggest_tumor(segmentation_3d, n_slices=10, axis=2):
    """
    Return the indices of the n_slices axial slices with the largest tumor area
    (tumor = labels 1, 2, or 3). axis=2 for axial (xy plane at fixed z).
    """
    # Tumor voxels per slice along axis (axial: sum over axes 0,1)
    sum_axes = tuple(i for i in range(3) if i != axis)
    tumor_per_slice = np.sum(
        (segmentation_3d >= 1) & (segmentation_3d <= 3), axis=sum_axes
    )
    n = min(n_slices, tumor_per_slice.shape[0])
    # Indices of top n slices by tumor count (descending)
    top_indices = np.argsort(tumor_per_slice)[::-1][:n]
    return top_indices.tolist()


def _seg_to_rgb_preview(lab):
    """Segmentation to RGB for preview: orange=necrotic, green=edema, red=enhancing, black=bg."""
    lab = np.clip(lab, 0, 3).astype(np.int32)
    rgb = np.zeros((*lab.shape, 3))
    rgb[lab == 1] = [1, 0.5, 0]  # necrotic: orange
    rgb[lab == 2] = [0, 0.8, 0]  # edema: green
    rgb[lab == 3] = [1, 0, 0]  # enhancing: red
    rgb[lab == 0] = [0, 0, 0]  # background: black
    return rgb


# Class names and colors matching BrainTumorSegmentation-3DUNet-StreamlitApp (create_matplotlib_plot)
# When extend_segmentation_with_normal_brain is used, label 4 = Normal brain
CLASS_NAMES = ["Background", "Necrotic/Core", "Edema", "Enhancing", "Normal brain"]
SEG_COLORS_STREAMLIT = [
    "black",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#a09080",
]  # black, orange, green, red, tan for normal brain


def create_slice_preview_figure(volume_4d, segmentation_3d, slice_idx, slice_axis=2):
    """
    Create one 2x3 figure for a single slice: FLAIR, T1, T1CE, T2, Predicted Segmentation,
    FLAIR + Segmentation Overlay. volume_4d shape (4, H, W, D), segmentation_3d (H, W, D).
    slice_axis=2 -> axial slice at slice_idx.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return None
    if slice_axis != 2:
        raise NotImplementedError("Only slice_axis=2 (axial) is implemented")
    # Axial: take slice at D index slice_idx
    flair = volume_4d[0, :, :, slice_idx]
    t1 = volume_4d[1, :, :, slice_idx]
    t1ce = volume_4d[2, :, :, slice_idx]
    t2 = volume_4d[3, :, :, slice_idx]
    seg_slice = segmentation_3d[:, :, slice_idx]
    seg_rgb = _seg_to_rgb_preview(seg_slice)
    overlay = 0.5 * (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    overlay = np.stack([overlay] * 3, axis=-1)
    overlay = np.clip(overlay + 0.5 * seg_rgb, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(flair, cmap="gray", origin="lower")
    axes[0, 0].set_title(f"FLAIR (Slice {slice_idx})")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(t1, cmap="gray", origin="lower")
    axes[0, 1].set_title(f"T1 (Slice {slice_idx})")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(t1ce, cmap="gray", origin="lower")
    axes[0, 2].set_title(f"T1CE (Slice {slice_idx})")
    axes[0, 2].axis("off")
    axes[1, 0].imshow(t2, cmap="gray", origin="lower")
    axes[1, 0].set_title(f"T2 (Slice {slice_idx})")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(seg_rgb, origin="lower")
    axes[1, 1].set_title(f"Predicted Segmentation (Slice {slice_idx})")
    axes[1, 1].axis("off")
    legend_elements = [
        Patch(facecolor="black", label="Background"),
        Patch(facecolor="orange", label="Necrotic/Core"),
        Patch(facecolor="green", label="Edema"),
        Patch(facecolor="red", label="Enhancing"),
    ]
    axes[1, 1].legend(handles=legend_elements, loc="upper right", fontsize=8)
    axes[1, 2].imshow(overlay, origin="lower")
    axes[1, 2].set_title(f"FLAIR + Segmentation Overlay (Slice {slice_idx})")
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def create_slice_preview_figure_streamlit_style(
    volume_4d, segmentation_3d, slice_idx, slice_axis=2
):
    """
    Same layout and technique as BrainTumorSegmentation-3DUNet-StreamlitApp create_matplotlib_plot:
    2x3 grid (FLAIR, T1, T1CE | T2, Predicted Segmentation, FLAIR+Overlay), ListedColormap for seg,
    overlay = FLAIR (gray alpha=0.7) + masked segmentation (alpha=0.8). volume_4d (4,H,W,D).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError:
        return None
    if slice_axis != 2:
        raise NotImplementedError("Only slice_axis=2 (axial) is implemented")
    seg_cmap = ListedColormap(SEG_COLORS_STREAMLIT)
    modality_names = ["FLAIR", "T1", "T1CE", "T2"]
    # Positions: (0,0) FLAIR, (0,1) T1, (0,2) T1CE, (1,0) T2, (1,1) Seg, (1,2) Overlay
    positions = [(0, 0), (0, 1), (0, 2), (1, 0)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, (row, col) in enumerate(positions):
        img_slice = volume_4d[i, :, :, slice_idx]
        im = axes[row, col].imshow(img_slice, cmap="gray", origin="lower")
        axes[row, col].set_title(
            f"{modality_names[i]} (Slice {slice_idx})", fontsize=14, fontweight="bold"
        )
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
    seg_slice = segmentation_3d[:, :, slice_idx]
    seg_vmax = 4 if np.max(segmentation_3d) >= 4 else 3
    seg_cmap_local = ListedColormap(SEG_COLORS_STREAMLIT[: seg_vmax + 1])
    im_seg = axes[1, 1].imshow(
        seg_slice,
        cmap=seg_cmap_local,
        origin="lower",
        vmin=0,
        vmax=seg_vmax,
        interpolation="nearest",
    )
    axes[1, 1].set_title(
        f"Predicted Segmentation (Slice {slice_idx})", fontsize=14, fontweight="bold"
    )
    axes[1, 1].axis("off")
    legend_patches = [
        Patch(color=SEG_COLORS_STREAMLIT[i], label=CLASS_NAMES[i])
        for i in range(seg_vmax + 1)
    ]
    axes[1, 1].legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    base_img = volume_4d[0, :, :, slice_idx]
    axes[1, 2].imshow(base_img, cmap="gray", origin="lower", alpha=0.7)
    mask_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)
    axes[1, 2].imshow(
        mask_overlay,
        cmap=seg_cmap_local,
        origin="lower",
        vmin=0,
        vmax=seg_vmax,
        alpha=0.8,
        interpolation="nearest",
    )
    axes[1, 2].set_title(
        f"FLAIR + Segmentation Overlay (Slice {slice_idx})",
        fontsize=14,
        fontweight="bold",
    )
    axes[1, 2].axis("off")
    axes[1, 2].legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.tight_layout()
    return fig


def create_ten_slice_preview(volume_4d, segmentation_3d, slice_indices, save_path=None):
    """
    Create a combined figure with 10 rows (one per slice), each row: FLAIR, T1, T1CE, T2,
    Predicted Segmentation, FLAIR+Overlay. If save_path is set, save and close; else return fig.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return None
    n_rows = min(10, len(slice_indices))
    fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    modality_names = ["FLAIR", "T1", "T1CE", "T2"]
    legend_elements = [
        Patch(facecolor="black", label="Background"),
        Patch(facecolor="orange", label="Necrotic/Core"),
        Patch(facecolor="green", label="Edema"),
        Patch(facecolor="red", label="Enhancing"),
    ]
    for row, slice_idx in enumerate(slice_indices[:n_rows]):
        flair = volume_4d[0, :, :, slice_idx]
        t1 = volume_4d[1, :, :, slice_idx]
        t1ce = volume_4d[2, :, :, slice_idx]
        t2 = volume_4d[3, :, :, slice_idx]
        seg_slice = segmentation_3d[:, :, slice_idx]
        seg_rgb = _seg_to_rgb_preview(seg_slice)
        overlay = 0.5 * (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
        overlay = np.stack([overlay] * 3, axis=-1)
        overlay = np.clip(overlay + 0.5 * seg_rgb, 0, 1)
        for col, (data, title) in enumerate(
            [
                (flair, "FLAIR"),
                (t1, "T1"),
                (t1ce, "T1CE"),
                (t2, "T2"),
                (seg_rgb, "Segmentation"),
                (overlay, "FLAIR+Overlay"),
            ]
        ):
            if col < 4:
                axes[row, col].imshow(data, cmap="gray", origin="lower")
            else:
                axes[row, col].imshow(data, origin="lower")
            axes[row, col].set_title(f"{title} (Slice {slice_idx})", fontsize=9)
            axes[row, col].axis("off")
        axes[row, 4].legend(handles=legend_elements, loc="upper right", fontsize=6)
        axes[row, 5].legend(handles=legend_elements, loc="upper right", fontsize=6)
    fig.suptitle(
        "10 slices with biggest tumor area – modalities and segmentation", fontsize=12
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def create_3x15_tumor_previews(
    volume_4d,
    segmentation_3d,
    sar_3d,
    temperature_3d,
    output_dir,
    case_name="case",
    n_slices=15,
):
    """
    Create three 3x5 grid preview images for axial slices with largest tumor area:
    (1) FLAIR + segmentation overlay, (2) SAR, (3) Temperature.
    Saves one PNG per modality to output_dir.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Ensure spatial shapes are compatible
    H, W, D = segmentation_3d.shape
    if volume_4d.shape[1:] != (H, W, D):
        raise ValueError(
            f"volume_4d spatial shape {volume_4d.shape[1:]} "
            f"does not match segmentation_3d shape {segmentation_3d.shape}"
        )
    if sar_3d.shape != (H, W, D) or temperature_3d.shape != (H, W, D):
        raise ValueError(
            "sar_3d and temperature_3d must have shape (H, W, D) "
            f"matching segmentation_3d; got {sar_3d.shape}, {temperature_3d.shape}"
        )

    # Select axial slices (axis=2) with largest tumor area
    top_indices = select_slices_biggest_tumor(
        segmentation_3d, n_slices=n_slices, axis=2
    )
    if not top_indices:
        return

    top_indices = [z for z in top_indices if 0 <= z < D][:n_slices]
    if not top_indices:
        return

    n_show = len(top_indices)
    nrows, ncols = 3, 5

    def _make_flair_seg_grid():
        flair_vol = volume_4d[0]
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        for idx, z in enumerate(top_indices[: nrows * ncols]):
            row, col = idx // ncols, idx % ncols
            flair = flair_vol[:, :, z]
            seg_slice = segmentation_3d[:, :, z]
            seg_rgb = _seg_to_rgb_preview(seg_slice)
            base = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
            base = np.clip(base, 0, 1)
            base_rgb = np.stack([base] * 3, axis=-1)
            overlay = np.clip(0.5 * base_rgb + 0.5 * seg_rgb, 0, 1)
            axes[row, col].imshow(overlay, origin="lower")
            axes[row, col].set_title(f"z={z}", fontsize=10)
            axes[row, col].axis("off")
        for idx in range(n_show, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].axis("off")
        fig.suptitle(
            f"{case_name} — FLAIR + segmentation (top {n_show} slices by tumor area)",
            fontsize=12,
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir, f"{case_name}_tumor_preview_flair_segmentation.png"
            ),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _make_scalar_grid(
        volume_3d, title_prefix, filename_suffix, cmap, vmin_phys, vmax_phys, cbar_label
    ):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        im = None
        for idx, z in enumerate(top_indices[: nrows * ncols]):
            row, col = idx // ncols, idx % ncols
            img_slice = volume_3d[:, :, z]
            # Per-slice contrast: robustly normalize this slice to [0, 1]
            if np.any(np.isfinite(img_slice)):
                lo, hi = np.nanpercentile(img_slice, [1, 99])
                if hi > lo:
                    img_slice = (img_slice - lo) / (hi - lo)
            img_slice = np.clip(img_slice, 0.0, 1.0)
            im = axes[row, col].imshow(
                img_slice, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0
            )
            axes[row, col].set_title(f"z={z}", fontsize=10)
            axes[row, col].axis("off")
        for idx in range(n_show, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].axis("off")
        fig.suptitle(
            f"{case_name} — {title_prefix} (top {n_show} slices by tumor area)",
            fontsize=12,
        )
        # Layout subplots, leaving room on the right for a shared colorbar
        fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])

        if im is not None:
            # Colorbar in real physical units, even though images are slice-normalized.
            # We map normalized [0,1] tick positions back to [vmin_phys, vmax_phys].
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cax)
            n_ticks = 5
            tick_positions = np.linspace(0.0, 1.0, n_ticks)
            tick_values = np.linspace(vmin_phys, vmax_phys, n_ticks)
            cbar.set_ticks(tick_positions)
            # Show full numeric precision (up to 5 decimal places)
            cbar.set_ticklabels([f"{v:.5f}" for v in tick_values])
            cbar.set_label(cbar_label)

        fig.savefig(
            os.path.join(output_dir, f"{case_name}_distribution_{filename_suffix}.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)

    _make_flair_seg_grid()

    # Match FDTD per-slice panels: use the same global magnitude ranges
    # SAR: vmin=0, vmax=max(SAR) (see fdtd_brain_simulation_engine.py:4512-4515)
    sar_max = float(np.max(sar_3d)) if np.max(sar_3d) > 0 else 1.0
    _make_scalar_grid(
        sar_3d,
        "SAR",
        "sar",
        cmap="inferno",
        vmin_phys=0.0,
        vmax_phys=sar_max,
        cbar_label="SAR (W/kg)",
    )

    # Temperature: vmin=min(T), vmax=max(T) (see fdtd_brain_simulation_engine.py:4530-4540)
    T_min = float(np.min(temperature_3d))
    T_max = float(np.max(temperature_3d))
    if T_max <= T_min:
        T_max = T_min + 0.1
    _make_scalar_grid(
        temperature_3d,
        "Temperature",
        "temperature",
        cmap="coolwarm",
        vmin_phys=T_min,
        vmax_phys=T_max,
        cbar_label="T (°C)",
    )


# Label for non-tumor brain tissue (used when extend_with_normal_brain=True)
LABEL_NORMAL_BRAIN = 4


def get_brain_mask_from_volume(volume_4d, modality_index=0, percentile_low=15):
    """
    Compute a binary brain mask from a 4D volume using one modality (default FLAIR).
    Uses intensity percentile threshold and hole-filling to get a contiguous brain region.
    Returns boolean 3D array of shape (H, W, D).
    """
    vol = volume_4d[modality_index]
    # Use percentile of non-zero voxels to avoid skew from large background
    valid = vol[vol > 0]
    if valid.size == 0:
        return np.zeros(vol.shape, dtype=bool)
    thresh = np.percentile(valid, percentile_low)
    mask = (vol > thresh).astype(np.uint8)
    # Fill holes so interior of brain is connected
    mask = ndimage.binary_fill_holes(mask).astype(bool)
    # Optional: remove small disconnected components (keep largest)
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = np.bincount(labeled.ravel())[1:]
        largest = np.argmax(sizes) + 1
        mask = labeled == largest
    return mask


def extend_segmentation_with_normal_brain(labels_3d, volume_4d, modality_index=0):
    """
    Extend BraTS segmentation (0=bg, 1=necrotic, 2=edema, 3=enhancing) by labeling
    non-tumor brain voxels as normal brain (LABEL_NORMAL_BRAIN).
    Voxels inside the brain mask that are currently 0 (background) become LABEL_NORMAL_BRAIN.
    Returns a new int32 array; shape must match volume_4d spatial dimensions (H, W, D).
    """
    assert (
        labels_3d.shape == volume_4d.shape[1:]
    ), "labels_3d shape must match volume spatial shape"
    brain_mask = get_brain_mask_from_volume(volume_4d, modality_index=modality_index)
    out = np.asarray(labels_3d, dtype=np.int32).copy()
    # Inside brain mask, any voxel that was background (0) becomes normal brain (4)
    out[brain_mask & (out == 0)] = LABEL_NORMAL_BRAIN
    return out


def run_segmentation_from_modalities(
    flair_path,
    t1_path,
    t1ce_path,
    t2_path,
    checkpoint_path,
    extend_with_normal_brain=True,
):
    """
    Load 4 NIfTIs, run 3D U-Net, return segmentation as int array.
    Labels: 0=background (air), 1=necrotic, 2=edema, 3=enhancing,
    and if extend_with_normal_brain=True: 4=normal brain (non-tumor brain tissue).
    """
    model, device = load_model(checkpoint_path)
    volume = load_patient_volume_from_paths(flair_path, t1_path, t1ce_path, t2_path)
    segmentation, _ = predict_segmentation(model, device, volume)
    segmentation = segmentation.astype(np.int32)
    if extend_with_normal_brain:
        segmentation = extend_segmentation_with_normal_brain(
            segmentation, volume, modality_index=0
        )
    return segmentation
