# hermes-drh

**HERMES** (Hyperthermia Electromagnetic and Robust Modeling Evaluation Suite) is a Python framework for patient-specific deep regional hyperthermia (DRH) research: MRI-derived voxel models, 3D FDTD electromagnetic simulation, SAR and simplified thermal post-processing, optional four-quadrant antenna optimization, and visualization.

## Install

From the `CODE/` directory (development):

```bash
pip install -e ".[dev]"
```

End users (PyPI, when published):

```bash
pip install hermes-drh
pip install "hermes-drh[segmentation]"   # BraTS 3D U-Net path (PyTorch + MONAI)
pip install "hermes-drh[all]"            # segmentation + Streamlit dashboard
```

Requires **Python 3.11–3.13**. For MP4 animations, install **ffmpeg** on your system.

## Quickstart

```bash
# From existing segmentation NIfTI
hermes-simulate --seg path/to/labels.nii.gz

# BraTS-style modalities folder
hermes-simulate --modalities-dir path/to/001

# YAML defaults (bundled example config)
hermes-simulate --config "$(python -c "from importlib.resources import files; print(files('hermes_drh').joinpath('configs/simulation_example.yaml'))")" --modalities-dir path/to/001

# Build MP4s from a completed streamed run
hermes-build-animations results/your_run_dir --skip-3d
```

Set `HERMES_CHECKPOINT=/path/to/best_model.pth` when using `--modalities` / `--modalities-dir` without passing `--checkpoint`.

## Dashboard

```bash
pip install "hermes-drh[dashboard]"
hermes-dashboard
```

## Thesis paper bundle (not in this package)

LaTeX figure generation and `run_paper_bundle.py` live in the thesis repository only; they are **not** installed from PyPI.

## License

MIT — see [LICENSE](LICENSE).
