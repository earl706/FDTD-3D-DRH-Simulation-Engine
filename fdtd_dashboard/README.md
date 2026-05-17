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

## Tabs (main UI)

| Tab | Purpose |
|-----|---------|
| **Simulation** | Upload MRI / set CLI-style parameters / run the engine locally or download a package for the local runner. |
| **Overview** | KPIs (total time, time per FDTD step, voxels, peak memory, animation time), grid metadata, and pipeline phase charts (pie/bar). |
| **Region** | Region statistics (SAR and temperature in tumor vs non-tumor), tissue properties from metadata, and objective \(J\) when available. |
| **Antenna** | Four-quadrant APA metadata: optimized or fixed amplitudes/phases and carrier frequency when applicable. |
| **Scalability** | **Multiple runs:** scatter plots of wall time and peak memory vs voxels (use sidebar **Compare all runs**). **Single run:** same-style performance summary (KPIs + phase breakdown) until more runs exist. |
| **Slice** | Single-run slice viewer, time series, and per-frame SAR/temperature (disabled when comparing all runs). |
| **Images/Animations** | PNGs under `images/` and MP4s under `animations/` for the loaded run(s). |

## Sidebar

- **Run selection:** choose one run from `results/{timestamp}/data/`, or enable **Compare all runs (scalability)** to load every run.
- **Data preview:** optional NIfTI slice preview when a single run is selected.

## Run simulation on your machine (no server compute)

When the app is deployed (e.g. Streamlit Cloud), you can run the FDTD simulation on your own PC so the server does not run the engine:

1. In the **Simulation** tab, choose **"My machine (local runner)"**.
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
- Optional: `{base}_SAR.nii.gz`, `{base}_temperature.nii.gz`, `{base}_segmentation.nii.gz`, frames under `data/SAR_frames/`, etc.

These are produced by `fdtd_brain_simulation_engine.py` in the parent directory.
