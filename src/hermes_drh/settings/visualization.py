"""
Per-anatomy visualization config: label colors (RGB) and figure titles.

Used by data_analysis_validation save/plot helpers when viz_config is provided.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VisualizationConfig:
    """Label→RGB and display strings for one anatomy."""

    label_colors: dict  # int -> (r, g, b) in [0,1]
    geometry_slice_title: str
    tumor_preview_suptitle: str
    tumor_3d_title: str
    slice_panel_suffix: Optional[str] = None  # e.g. "top 10 by tumor area" or None for generic


# Brain (BraTS): 0=bg, 1=necrotic, 2=edema, 3=enhancing, 4=normal brain
BRAIN_LABEL_COLORS = {
    0: (0.15, 0.15, 0.15),
    1: (1.0, 0.2, 0.2),
    2: (0.2, 0.8, 0.2),
    3: (0.2, 0.2, 1.0),
    4: (0.6, 0.55, 0.5),
}
BRAIN_VISUALIZATION_CONFIG = VisualizationConfig(
    label_colors=BRAIN_LABEL_COLORS,
    geometry_slice_title="FDTD geometry (mid-Z): dark=air, R/G/B=tumor, tan=normal brain",
    tumor_preview_suptitle="Brain segmentation – tumor visualization (before FDTD)",
    tumor_3d_title="3D tumor (red=necrotic, green=edema, blue=enhancing)",
    slice_panel_suffix="top 10 by tumor area",
)

# Breast: 0=air, 1=tumor, 2=healthy breast
BREAST_LABEL_COLORS = {
    0: (0.15, 0.15, 0.15),
    1: (1.0, 0.2, 0.2),
    2: (1.0, 0.75, 0.8),
}
BREAST_VISUALIZATION_CONFIG = VisualizationConfig(
    label_colors=BREAST_LABEL_COLORS,
    geometry_slice_title="FDTD geometry (mid-Z): dark=air, red=tumor, light=healthy breast",
    tumor_preview_suptitle="Breast segmentation – tumor / healthy breast (before FDTD)",
    tumor_3d_title="3D tumor (red) / healthy breast",
    slice_panel_suffix=None,
)

# Cervix: 0=air, 1=tumor, 2=healthy cervix
CERVIX_LABEL_COLORS = {
    0: (0.15, 0.15, 0.15),
    1: (1.0, 0.2, 0.2),
    2: (0.9, 0.7, 0.75),
}
CERVIX_VISUALIZATION_CONFIG = VisualizationConfig(
    label_colors=CERVIX_LABEL_COLORS,
    geometry_slice_title="FDTD geometry (mid-Z): dark=air, red=tumor, light=healthy cervix",
    tumor_preview_suptitle="Cervix segmentation – tumor / healthy cervix (before FDTD)",
    tumor_3d_title="3D tumor (red) / healthy cervix",
    slice_panel_suffix=None,
)


def get_visualization_config(anatomy: str) -> VisualizationConfig:
    """Return VisualizationConfig for the given anatomy."""
    a = anatomy.lower().strip()
    if a == "brain":
        return BRAIN_VISUALIZATION_CONFIG
    if a == "breast":
        return BREAST_VISUALIZATION_CONFIG
    if a == "cervix":
        return CERVIX_VISUALIZATION_CONFIG
    raise ValueError(f"Unknown anatomy: {anatomy!r}. Use 'brain', 'breast', or 'cervix'.")
