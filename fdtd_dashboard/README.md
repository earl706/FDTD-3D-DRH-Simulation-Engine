# FDTD Simulation Results Dashboard

Streamlit dashboard for viewing results from `fdtd_brain_simulation_engine.py`.

## Setup

From the `fdtd_dashboard` directory:

```bash
pip install -r requirements.txt
```

Or from the thesis root:

```bash
pip install -r fdtd_dashboard/requirements.txt
```

## Run

From the thesis root:

```bash
streamlit run fdtd_dashboard/streamlit_app.py
```

Or from inside `fdtd_dashboard`:

```bash
streamlit run streamlit_app.py
```

Then open the URL shown (e.g. http://localhost:8501).

## Features

- **Run selection**: Pick a run from `results/{timestamp}/data/` (metadata and performance JSON).
- **Compare all**: Load all runs and show scalability (wall time and memory vs voxels).
- **Overview**: KPIs (total time, time/step, voxels, peak memory) and grid info.
- **Performance**: Time breakdown (FDTD, SAR, thermal) as pie and bar charts.
- **Region stats**: SAR and temperature min/max/mean for tumor vs non-tumor.
- **Antenna**: Optimized frequency, amplitudes, and phases when antenna optimization was used.
- **Scalability**: Scatter plots of wall time and peak memory vs voxels across runs.
- **Data preview**: Optional NIfTI slice view for SAR and temperature (if files exist).

## Data layout

The app expects:

- `results/{timestamp}/data/{base}_metadata.json`
- `results/{timestamp}/data/{base}_performance.json`
- Optional: `{base}_SAR.nii.gz`, `{base}_temperature.nii.gz`

These are produced by `fdtd_brain_simulation_engine.py` in the parent directory.
