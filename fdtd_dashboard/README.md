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

## Run simulation on your machine (no server compute)

When the app is deployed (e.g. Streamlit Cloud), you can run the FDTD simulation on your own PC so the server does not run the engine:

1. In the **Run simulation** tab, choose **"My machine (local runner)"**.
2. Configure inputs and parameters as usual, then click **Run simulation**.
3. Download the **run package (ZIP)** and the **Run page (HTML)**.
4. On your PC, from the repo root: `python fdtd_dashboard/local_runner.py`
5. Extract the ZIP and open the downloaded HTML in your browser.
6. Enter the path to the extracted folder and click **Run simulation on my machine**.

The browser sends the run config to the local runner (localhost:8765), which starts `fdtd_brain_simulation_engine.py` on your machine. Results appear under `results/` locally.

## Data layout

The app expects:

- `results/{timestamp}/data/{base}_metadata.json`
- `results/{timestamp}/data/{base}_performance.json`
- Optional: `{base}_SAR.nii.gz`, `{base}_temperature.nii.gz`

These are produced by `fdtd_brain_simulation_engine.py` in the parent directory.
