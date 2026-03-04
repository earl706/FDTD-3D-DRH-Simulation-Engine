# FDTD Brain Simulation Pipeline — Unified Documentation

This document describes, in simple words, the three main programs that work together: the **FDTD brain simulation engine**, the **brain tumor segmentation model**, and the **animation builder**. It explains what each does, how they connect, and how to run them.

---

## 1. What the pipeline does (in plain words)

The pipeline does three things:

1. **Finds where the tumor is** in the brain using MRI scans (FLAIR, T1, T1ce, T2). It uses a neural network (3D U-Net) to label each voxel as background, necrotic core, edema, enhancing tumor, or (optionally) normal brain.

2. **Runs a physics simulation** (FDTD = Finite-Difference Time-Domain) that models how electromagnetic waves from an antenna heat the head. The simulation uses the segmentation to assign different tissue properties (conductivity, permittivity, density) to tumor vs healthy tissue. You can optionally **optimize the antenna** (phases and amplitudes of four sources) so that more heating goes to the tumor and less to healthy tissue.

3. **Produces results**: 3D maps of electric field (E), SAR (Specific Absorption Rate), and temperature over time, plus still images and **animations** (videos) of those quantities. When the simulation is run in “streaming” mode (to save memory), animations are built by a **separate script** that reads the saved frames from disk.

So in short: **MRI → segmentation → FDTD simulation → E, SAR, temperature → images and videos.**

---

## 2. How the three programs fit together

```
  You provide:
  - Either: 4 BraTS NIfTI files (FLAIR, T1, T1ce, T2) in a folder, or
  - Or:      A single segmentation NIfTI file (already labeled)

                    │
                    ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  fdtd_brain_simulation_engine.py                                 │
  │  - If you gave 4 modalities: calls brain_tumor_segmentation_     │
  │    model.py to get segmentation (3D U-Net).                      │
  │  - If you gave segmentation: loads it directly.                  │
  │  - Builds 3D grid, assigns tissue properties from labels.        │
  │  - Optionally runs antenna optimization (phase/amplitude sweep).  │
  │  - Runs FDTD: E-field → SAR → temperature over time.             │
  │  - Saves: data (NIfTI, NPZ frames, JSON), images, and            │
  │    (if not streaming) in-memory animations.                       │
  │  - If --stream-frames: writes E/SAR/T frames to disk in chunks   │
  │    and can spawn build_animations_from_streamed_frames.py.       │
  └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
  results/DDMMYY-HHMMSS/
  ├── data/          (segmentation, metadata, E_frames, SAR_frames, Temperature_frames, etc.)
  ├── images/        (PNG slices, geometry, comparison plots)
  └── animations/    (MP4 videos — or empty if streamed and not yet built)

                    │
                    ▼  (when you used --stream-frames)
  ┌─────────────────────────────────────────────────────────────────┐
  │  build_animations_from_streamed_frames.py                         │
  │  - Reads E_frames, SAR_frames, Temperature_frames from data/     │
  │    (in chunks, so it doesn’t need to load everything into RAM).   │
  │  - Builds 2D combined video (E, SAR, Temp on one slice) and     │
  │    optional 3D isometric videos (E, SAR, Temp).                   │
  │  - Writes MP4s into results/.../animations/ (or a path you set). │
  └─────────────────────────────────────────────────────────────────┘
```

- **brain_tumor_segmentation_model.py** is **not** run by you directly for this pipeline. The **engine** imports it and calls `run_segmentation_from_modalities(...)` when you pass `--modalities` or `--modalities-dir`. So you only run the engine (and optionally the animation script); the segmentation module is used inside the engine.

---

## 3. fdtd_brain_simulation_engine.py

### 3.1 What it is

The main program. It:

- Loads or produces a **brain segmentation** (labels: 0=background, 1=necrotic, 2=edema, 3=enhancing; optionally 4=normal brain).
- Builds a **3D simulation grid** and assigns **tissue properties** (permittivity, conductivity, density) from the segmentation.
- Can run **antenna optimization**: it tries many combinations of phases and amplitudes for four quadrant sources to maximize “tumor SAR / healthy SAR” (and optionally penalize hot spots in healthy tissue).
- Runs the **FDTD** simulation: updates electric and magnetic fields step by step in time, then computes **SAR** and **temperature** from the E-field.
- Saves all results under **results/DDMMYY-HHMMSS/** (data, images, and sometimes animations).

It is **CPU-based** (NumPy + Numba). No GPU is used for FDTD.

### 3.2 Inputs

- **Option A — From MRI (four modalities):**  
  A folder with four NIfTI files (FLAIR, T1, T1ce, T2), e.g. `dataset/validation_data/001`. The engine will run the 3D U-Net (from `brain_tumor_segmentation_model.py`) to get the segmentation.

- **Option B — From segmentation only:**  
  A single NIfTI file with integer labels (0–3 or 0–4). No MRI needed; the engine loads this and skips segmentation.

### 3.3 Outputs (where they go)

Everything is written under **results/DDMMYY-HHMMSS/**:

- **data/** — Segmentation NIfTI, metadata JSON, performance JSON, optional optimization/trace/freq-sweep/geom-sweep JSONs, SAR/temperature NIfTI, and (when streaming or when saving frames) **E_frames/**, **SAR_frames/**, **Temperature_frames/** as `.npz` part files.
- **images/** — PNGs: tumor preview slices, FDTD geometry slice, antenna comparison, optimization trace, etc.
- **animations/** — MP4 videos. If you **did not** use `--stream-frames`, the engine builds these itself (in memory). If you **did** use `--stream-frames`, the engine does not build animations; you run `build_animations_from_streamed_frames.py` to create them from the saved frames.

### 3.4 Main steps inside the engine (in order)

1. **Get segmentation** — From file or by running the 3D U-Net on the four modalities.
2. **Downsample (optional)** — If the grid would be larger than `--max-dim`, the segmentation is downsampled so the longest side equals `--max-dim`.
3. **Set up grid and tissue** — Build 3D arrays for permittivity, conductivity, density (and thermal properties) from the labels.
4. **Antenna optimization (optional)** — If `--optimize-antenna` is set: frequency sweep (optional), geometry sweep (source position/offset and z-planes), then phase/amplitude optimization with multi-start. The best antenna settings are then used for the main run.
5. **Main FDTD run** — Time-step the Maxwell equations, compute E and H; at each step (or at intervals) compute instantaneous SAR and, after the run, temperature. Either keep frames in memory or **stream** them to disk (E_frames, SAR_frames; Temperature_frames are computed from SAR after the run).
6. **Save** — Write metadata, segmentation, SAR/temperature NIfTI, frame part files (if streamed or saved), and build in-memory animations only when not streaming. If streaming, it can automatically start `build_animations_from_streamed_frames.py` for you.

### 3.5 Important command-line options (plain words)

- **Input**
  - `--modalities-dir DIR` — Use the four BraTS NIfTI files in this folder; run segmentation first.
  - `--seg FILE` or positional `FILE` — Use an existing segmentation NIfTI; skip segmentation.
  - `--checkpoint PATH` — Path to the 3D U-Net weights (e.g. `best_model.pth`). Used only when running from modalities.
  - `--no-normal-brain` — Do not add label 4 (normal brain); keep only 0–3.

- **Grid size**
  - `--max-dim N` — Downsample so no grid dimension exceeds N (e.g. 120 or 192). Smaller N = less memory and faster, but coarser.

- **Antenna optimization**
  - `--optimize-antenna` — Turn on optimization (phase, amplitude, optional geometry/frequency).
  - `--f0 FREQ` — Carrier frequency in Hz (e.g. 100e6).
  - `--opt-phase-steps N` — Number of phase values per quadrant in the coarse sweep (e.g. 24 or 48).
  - `--opt-amp-steps N` — Number of amplitude values in the coarse sweep.
  - `--opt-amp-min`, `--opt-amp-max` — Amplitude range.
  - `--opt-multi-start N` — Number of random starting phases to try (e.g. 4 or 8).
  - `--opt-refine-iters N` — Refinement steps after the coarse sweep.
  - `--opt-geom-offsets 8 10 12` — Source “ring” offsets (cell indices from the boundary). Use at least 8 so sources stay outside the PML; 2 3 4 put sources inside the absorbing layer and are not recommended.
  - `--opt-geom-zplanes Z1 Z2 ...` — Z-indices for the source plane sweep. Omit to use a single plane (middle). If you set them, use values that bracket the tumor (e.g. around the printed “Tumor centroid z-index”).
  - `--opt-penalty-weight W` — How much to penalize hot spots in healthy tissue (0 = only maximize tumor/healthy SAR ratio; e.g. 0.1 = also reduce healthy-tissue hotspots).
  - `--opt-source-scale S` — Scale factor for the optimized source in the final run (SAR scales with S²).

- **Streaming (saves RAM)**
  - `--stream-frames` — Write E and SAR frames to disk in chunks instead of keeping them in memory. Use this for large grids or limited RAM. Animations are then built by `build_animations_from_streamed_frames.py`.
  - `--stream-frame-interval N` — Save a frame every N time steps (default 1).

- **Other**
  - `--pulse-amplitude A` — For non-optimized runs, amplitude of the Gaussian pulse (SAR scales with A²).

### 3.6 Example commands

**From a folder of four modalities, with antenna optimization and streaming (good for 16 GB RAM):**

```bash
python fdtd_brain_simulation_engine.py --modalities-dir dataset/validation_data/001 --optimize-antenna \
  --f0 100e6 \
  --opt-phase-steps 24 \
  --opt-amp-steps 8 \
  --opt-amp-min 0.2 --opt-amp-max 100 \
  --opt-refine-iters 3 \
  --opt-multi-start 4 \
  --opt-penalty-weight 0 \
  --opt-geom-offsets 8 10 12 \
  --opt-source-scale 5000 \
  --max-dim 192 --stream-frames
```

**From an existing segmentation file, no optimization:**

```bash
python fdtd_brain_simulation_engine.py --seg brain_segmentation.nii --max-dim 120
```

---

## 4. brain_tumor_segmentation_model.py

### 4.1 What it is

A **headless** (no GUI) module that runs **brain tumor segmentation** from four BraTS MRI modalities. It is used **by the FDTD engine** when you pass `--modalities` or `--modalities-dir`. You do not normally run this script by itself; the engine imports it and calls `run_segmentation_from_modalities(...)`.

### 4.2 What it does (in plain words)

- Loads four NIfTI files: FLAIR, T1, T1ce, T2.
- Normalizes each volume (zero mean, unit variance).
- Runs a **3D U-Net** (BraTS-style, 4 input channels, 4 classes) in patches over the volume, then stitches the predictions back together (with averaging in overlapping regions).
- Outputs integer labels: **0** = background, **1** = necrotic core, **2** = edema, **3** = enhancing tumor. Optionally it can add **4** = normal brain (non-tumor brain tissue) by using a simple brain mask and labeling remaining background inside the mask as 4.

### 4.3 Model and code origin

The architecture (e.g. `UNet3D_BraTS`, `Conv3D_Block`, `Deconv3D_Block`) is derived from the project **“Brain Tumor Segmentation with 3D U-Net” (BrainTumorSegmentation-3DUNet-StreamlitApp)**, used under the Apache License 2.0. This file provides a **headless** version: load checkpoint, run inference on a 4D volume, return the segmentation array.

### 4.4 Key functions (for reference)

- **`load_model(checkpoint_path)`** — Loads the 3D U-Net from a `.pth` file; returns model and device (CPU or CUDA).
- **`load_patient_volume_from_paths(flair_path, t1_path, t1ce_path, t2_path)`** — Loads and normalizes the four NIfTIs; returns a 4D array (4, H, W, D).
- **`predict_segmentation(model, device, volume, patch_size, stride, batch_size)`** — Runs the U-Net in patches; returns segmentation and class probabilities.
- **`extend_segmentation_with_normal_brain(labels_3d, volume_4d, modality_index=0)`** — Sets label 4 for non-tumor brain voxels inside a brain mask.
- **`run_segmentation_from_modalities(flair_path, t1_path, t1ce_path, t2_path, checkpoint_path, extend_with_normal_brain=True)`** — Full pipeline: load → predict → optionally extend with normal brain; returns the final integer segmentation. This is what the FDTD engine calls.

Other helpers (e.g. `select_slices_biggest_tumor`, `create_slice_preview_figure`, `create_ten_slice_preview`) are used by the engine for preview images and slice selection.

### 4.5 Dependencies

- PyTorch, NumPy, NiBabel, SciPy. Optional: tqdm for progress bars.

---

## 5. build_animations_from_streamed_frames.py

### 5.1 What it is

A **standalone script** that builds **MP4 animations** from the **saved** E-field, SAR, and temperature frames. You need it when the FDTD engine was run with **`--stream-frames`**, because in that case the engine does not keep all frames in memory and does not build the videos itself. This script reads the frame **part files** from disk in chunks, so it can handle long runs without running out of RAM.

### 5.2 When to use it

- **Use it** when you ran the engine with `--stream-frames`. The engine may also spawn this script automatically at the end of a streamed run (if the script is found next to the engine).
- **You don’t need it** when you did **not** use `--stream-frames`: the engine builds and saves animations itself (in memory) and writes them to `results/.../animations/`.

### 5.3 Inputs

- **results_dir** (positional or via `--data-dir`) — The results folder from the simulation (e.g. `results/170226-142342`). The script expects **data/** inside it (or you point to that data dir explicitly).
- Inside **data/** it looks for:
  - **metadata** — `{output_base}_metadata.json` (or you pass `--output-base`).
  - **E_frames/** — `{output_base}_E_frames_part0.npz`, `part1.npz`, ...
  - **SAR_frames/** — `{output_base}_SAR_frames_part0.npz`, ...
  - **Temperature_frames/** — `{output_base}_Temperature_frames_part0.npz`, ... (optional; if missing, only E and SAR are animated.)
  - **Segmentation** — `{output_base}_segmentation.nii.gz` (for tumor contour overlay).

The script uses the same **chunk size** as in the metadata (`E_frames_chunk_size`) so it knows how many frames are in each part file.

### 5.4 What it produces

- **2D combined animation** — One video with 2 or 3 panels: E-field (magnitude), SAR, and (if available) temperature, all on **one slice**: the slice through the **tumor centroid (z)**. Tumor contour is overlaid. Saved as e.g. `{output_base}_efield_sar_temp_2d.mp4` or `{output_base}_sar_2d.mp4` if no temperature.
- **3D isometric animations** (unless you pass `--skip-3d`):
  - E-field 3D: `{output_base}_efield_3d.mp4`
  - SAR 3D: `{output_base}_sar_3d.mp4`
  - Temperature 3D (if Temperature_frames exist): `{output_base}_temperature_3d.mp4`

All are written to an **animations** directory: by default `results_dir/animations/`, or you can pass a second positional argument to set another path.

### 5.5 Command-line options (plain words)

- **results_dir** / **animations_dir** — First (and optional second) positional: results folder and optional output folder for animations.
- `--data-dir PATH` — Override the data directory (default is `results_dir/data`).
- `--output-base NAME` — Override the base name used to find metadata and frame files.
- `--subsample N` — Use every Nth frame (e.g. 10 for faster, shorter videos).
- `--skip-3d` — Only build the 2D combined animation; skip the three 3D MP4s.
- `--fps N` — Frames per second for the output videos (default 60).

### 5.6 Example

```bash
python build_animations_from_streamed_frames.py results/170226-142342
python build_animations_from_streamed_frames.py results/170226-142342 --subsample 10 --skip-3d
```

---

## 6. Results folder layout (summary)

After a full run (with or without streaming), a typical layout is:

```
results/DDMMYY-HHMMSS/
├── data/
│   ├── {base}_metadata.json          # Grid, timing, options, antenna (if optimized)
│   ├── {base}_segmentation.nii.gz    # Labels 0–4
│   ├── {base}_performance.json
│   ├── E_frames/                     # When frames saved/streamed
│   │   └── {base}_E_frames_part0.npz, part1.npz, ...
│   ├── SAR_frames/
│   │   └── {base}_SAR_frames_part0.npz, ...
│   ├── Temperature_frames/
│   │   └── {base}_Temperature_frames_part0.npz, ...
│   ├── {base}_sar.nii.gz, {base}_temperature.nii.gz
│   └── (optional) antenna/trace/freq-sweep/geom-sweep JSONs and NPYs
├── images/
│   └── {base}_*.png                  # Previews, geometry, comparisons
└── animations/
    └── {base}_*.mp4                  # After in-memory build or after build_animations_from_streamed_frames.py
```

`{base}` is the output base name (e.g. from the modalities folder name or the segmentation filename).

---

## 7. Quick reference: common workflows

| Goal | What to run |
|------|-----------------------------|
| Full run from MRI folder, optimize antenna, low memory | `fdtd_brain_simulation_engine.py --modalities-dir dataset/validation_data/001 --optimize-antenna ... --max-dim 192 --stream-frames` |
| Build videos after a streamed run | `build_animations_from_streamed_frames.py results/DDMMYY-HHMMSS` |
| Run from existing segmentation only | `fdtd_brain_simulation_engine.py --seg brain_seg.nii --max-dim 120` |
| Shorter/faster optimization (fewer steps) | Use smaller `--opt-phase-steps`, `--opt-amp-steps`, `--opt-multi-start`, `--opt-refine-iters` and omit or reduce `--opt-geom-zplanes` / `--opt-geom-offsets` |

---

## 8. Glossary (plain words)

- **FDTD** — A method that updates electric and magnetic fields in small time steps on a 3D grid to simulate how waves travel and heat tissue.
- **SAR (Specific Absorption Rate)** — Rate of energy absorbed per unit mass (W/kg). Used to quantify heating from RF.
- **PML** — Perfectly Matched Layer; a boundary region that absorbs waves so they don’t reflect back into the domain.
- **BraTS** — Brain Tumor Segmentation challenge; standard format: 4 MRI modalities and labels 0–3 (and often 4 for normal brain).
- **NIfTI** — Common file format for 3D/4D medical images (e.g. `.nii`, `.nii.gz`).
- **Streaming (--stream-frames)** — Writing E and SAR (and later temperature) frames to disk in chunks during the run instead of keeping them all in RAM, so you can run larger grids or longer runs on limited memory.

---

*This documentation covers `fdtd_brain_simulation_engine.py`, `brain_tumor_segmentation_model.py`, and `build_animations_from_streamed_frames.py` as of the current codebase. For exact defaults and extra options, run each script with `--help`.*
