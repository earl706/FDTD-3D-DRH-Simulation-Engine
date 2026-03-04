"""
FDTD Simulation Results Dashboard (Streamlit).

Renders metadata.json, performance.json, and optional NIfTI visualizations
from results produced by fdtd_brain_simulation_engine.py.

Usage:
  streamlit run streamlit_app.py

  From thesis root:
  streamlit run fdtd_dashboard/streamlit_app.py
"""

import json
import os
import subprocess
import sys
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Paths and discovery
# -----------------------------------------------------------------------------


def get_results_root():
    """Results directory: same repo root as this script, then 'results'."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return repo_root / "results"


def get_repo_root():
    """Thesis repo root (parent of fdtd_dashboard)."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def get_uploads_dir():
    """Directory for uploaded MRI data (saved to disk for engine)."""
    return get_results_root() / "uploads"


def discover_runs(results_root):
    """List run folders (timestamp dirs) that contain at least one metadata.json."""
    if not results_root or not results_root.is_dir():
        return []
    runs = []
    for path in sorted(results_root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        data_dir = path / "data"
        if not data_dir.is_dir():
            continue
        for f in data_dir.glob("*_metadata.json"):
            runs.append(
                {
                    "run_id": path.name,
                    "data_dir": data_dir,
                    "metadata_file": f,
                    "output_base": f.stem.replace("_metadata", ""),
                }
            )
            break  # one metadata per run folder for now
    return runs


def load_metadata(metadata_path):
    """Load and return metadata JSON."""
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_performance(performance_path):
    """Load and return performance JSON."""
    with open(performance_path, "r") as f:
        return json.load(f)


def load_run_data(run):
    """Load metadata and performance for a run. performance is also in metadata."""
    meta = load_metadata(run["metadata_file"])
    perf_path = run["data_dir"] / f"{run['output_base']}_performance.json"
    perf = (
        load_performance(perf_path)
        if perf_path.exists()
        else meta.get("performance", {})
    )
    return {
        "metadata": meta,
        "performance": perf,
        "run_id": run["run_id"],
        "output_base": run["output_base"],
        "data_dir": run["data_dir"],
    }


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------


def render_kpis(perf, run_id):
    total_s = perf.get("total_simulation_time_s") or perf.get("total_wall_time_s") or 0
    phases = perf.get("phases_s") or {}
    anim_s = phases.get("animations")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total simulation time (s)", f"{total_s:.2f}")
    with col2:
        st.metric("Time/step (ms)", f"{perf.get('time_per_step_ms') or 0:.2f}")
    with col3:
        nv = perf.get("number_of_voxels") or 0
        st.metric("Voxels", f"{nv:,}")
    with col4:
        mem = perf.get("peak_memory_MB")
        st.metric("Peak memory (MB)", f"{mem:.1f}" if mem is not None else "—")
    with col5:
        st.metric("Animation time (s)", f"{anim_s:.2f}" if anim_s is not None else "—")


def render_time_breakdown(perf):
    phases = perf.get("phases_s")
    if phases:
        # Phases: segmentation, setup, [antenna_optimization], fdtd_simulation, sar_computation, thermal_solver, saving_data, animations
        rows = [
            (name.replace("_", " ").title(), val)
            for name, val in phases.items()
            if val is not None
        ]
        if not rows:
            st.warning("No phase data in performance.")
            return
        df = pd.DataFrame(rows, columns=["Phase", "Time (s)"])
        total = df["Time (s)"].sum()
        if total <= 0:
            st.warning("No time breakdown data.")
            return
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=df["Phase"],
                        values=df["Time (s)"],
                        hole=0.4,
                        textinfo="label+percent",
                    )
                ]
            )
            fig_pie.update_layout(title="Pipeline phases", height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.bar(df, x="Phase", y="Time (s)", color="Time (s)")
            fig_bar.update_layout(xaxis_tickangle=-30, height=320, margin=dict(b=120))
            st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        total_sim = perf.get("total_simulation_time_s")
        if total_sim is not None:
            st.caption(
                f"Total simulation time (segmentation → animations): {total_sim:.2f} s"
            )
        return
    # Fallback: legacy format (FDTD, SAR, Thermal only)
    time_fdtd = perf.get("time_fdtd_s") or 0
    time_sar = perf.get("time_sar_s") or 0
    time_thermal = perf.get("time_thermal_s") or 0
    total = time_fdtd + time_sar + time_thermal
    if total <= 0:
        st.warning("No time breakdown data.")
        return
    df = pd.DataFrame(
        {
            "Stage": ["FDTD", "SAR", "Thermal"],
            "Time (s)": [time_fdtd, time_sar, time_thermal],
        }
    )
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=df["Stage"],
                    values=df["Time (s)"],
                    hole=0.4,
                    textinfo="label+percent",
                )
            ]
        )
        fig_pie.update_layout(title="Time breakdown", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar = px.bar(df, x="Stage", y="Time (s)", color="Time (s)")
        fig_bar.update_layout(height=300)
        st.plotly_chart(fig_bar, use_container_width=True)


def _region_stats_rows(rs, key_prefix):
    """Flatten region_stats entries whose key starts with key_prefix into Metric/Value rows."""
    rows = []
    for key, val in rs.items():
        if not key.startswith(key_prefix):
            continue
        if isinstance(val, dict):
            for k, v in val.items():
                rows.append({"Metric": f"{key} ({k})", "Value": v})
        else:
            rows.append({"Metric": key, "Value": val})
    return rows


def render_region_stats(meta):
    rs = meta.get("region_stats") or {}
    if not rs:
        st.info("No region stats in this run.")
        return

    # SAR section
    sar_rows = _region_stats_rows(rs, "sar_")
    if sar_rows:
        st.subheader("SAR (Specific Absorption Rate)")
        df_sar = pd.DataFrame(sar_rows)
        fig_sar = px.bar(df_sar, x="Metric", y="Value", title="SAR by region")
        fig_sar.update_layout(xaxis_tickangle=-45, height=360)
        st.plotly_chart(fig_sar, use_container_width=True)
        st.dataframe(df_sar, use_container_width=True, hide_index=True)

    # Temperature section
    temp_rows = _region_stats_rows(rs, "temperature_")
    if temp_rows:
        st.subheader("Temperature")
        df_temp = pd.DataFrame(temp_rows)
        fig_temp = px.bar(df_temp, x="Metric", y="Value", title="Temperature by region")
        fig_temp.update_layout(xaxis_tickangle=-45, height=360)
        st.plotly_chart(fig_temp, use_container_width=True)
        st.dataframe(df_temp, use_container_width=True, hide_index=True)

    if not sar_rows and not temp_rows:
        st.caption("No SAR or temperature region stats in this run.")


def render_tissue_properties(meta):
    """Render tissue/dielectric and thermal table from metadata."""
    rows = meta.get("tissue_properties")
    if not rows:
        st.info("No tissue properties in metadata (older run).")
        return
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "label": "Label",
            "name": "Tissue",
            "eps_r": "εr",
            "sigma_S_per_m": "σ (S/m)",
            "rho_kg_per_m3": "ρ (kg/m³)",
            "k_W_per_mK": "k (W/(m·K))",
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_antenna_params(meta):
    if not meta.get("antenna_optimized"):
        st.info("Antenna was not optimized for this run.")
        return
    st.subheader("Antenna optimization")
    st.write(f"**Optimized frequency:** {meta.get('optimized_f0_Hz', 0) / 1e6:.1f} MHz")
    alphas = meta.get("optimized_alphas") or []
    thetas = meta.get("optimized_thetas_rad") or []
    if alphas:
        fig_a = go.Figure(
            data=[go.Bar(x=[f"Q{i+1}" for i in range(len(alphas))], y=alphas)]
        )
        fig_a.update_layout(title="Amplitude (α) per quadrant", height=280)
        st.plotly_chart(fig_a, use_container_width=True)
    if thetas:
        fig_t = go.Figure(
            data=[go.Bar(x=[f"Q{i+1}" for i in range(len(thetas))], y=thetas)]
        )
        fig_t.update_layout(title="Phase θ (rad) per quadrant", height=280)
        st.plotly_chart(fig_t, use_container_width=True)


def render_scalability(runs_data):
    """Compare multiple runs: time and memory vs voxels."""
    if len(runs_data) < 2:
        st.info("Select multiple runs or 'Compare all' to see scalability.")
        return
    rows = []
    for d in runs_data:
        p = d["performance"]
        nv = p.get("number_of_voxels") or 0
        rows.append(
            {
                "run_id": d["run_id"],
                "voxels": nv,
                "total_wall_time_s": p.get("total_wall_time_s") or 0,
                "peak_memory_MB": p.get("peak_memory_MB") or 0,
                "time_per_step_ms": p.get("time_per_step_ms") or 0,
            }
        )
    df = pd.DataFrame(rows)
    col1, col2 = st.columns(2)
    with col1:
        fig_time = px.scatter(
            df, x="voxels", y="total_wall_time_s", hover_data=["run_id"]
        )
        fig_time.update_layout(
            title="Wall time vs voxels", xaxis_title="Voxels", yaxis_title="Time (s)"
        )
        st.plotly_chart(fig_time, use_container_width=True)
    with col2:
        fig_mem = px.scatter(df, x="voxels", y="peak_memory_MB", hover_data=["run_id"])
        fig_mem.update_layout(
            title="Peak memory vs voxels",
            xaxis_title="Voxels",
            yaxis_title="Memory (MB)",
        )
        st.plotly_chart(fig_mem, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _load_nifti(path):
    """Load NIfTI and return 3D array and shape; (None, None) if missing or error."""
    try:
        import nibabel as nib
    except ImportError:
        return None, None
    if not path.exists():
        return None, None
    try:
        nii = nib.load(str(path))
        vol = nii.get_fdata()
        return vol, vol.shape
    except Exception:
        return None, None


# BraTS-style label colors for segmentation (0=bg, 1=necrotic, 2=edema, 3=enhancing, 4=normal)
_SEGMENTATION_COLORSCALE = [
    [0.0, "rgb(38,38,38)"],  # background
    [0.25, "rgb(255,51,51)"],  # necrotic
    [0.5, "rgb(51,204,51)"],  # edema
    [0.75, "rgb(51,51,255)"],  # enhancing
    [1.0, "rgb(179,179,140)"],  # normal brain
]


def render_slice_viewer(data_dir, output_base, run_id=None):
    """
    Slider-controlled slice viewer for SAR, Temperature, and Segmentation NIfTIs.
    Uses axial slices (index k in third dimension). Only shown when a single run is selected.
    """
    try:
        import nibabel as nib
    except ImportError:
        st.warning("Install `nibabel` to use the slice viewer.")
        return

    sar_path = data_dir / f"{output_base}_SAR.nii.gz"
    temp_path = data_dir / f"{output_base}_temperature.nii.gz"
    seg_path = data_dir / f"{output_base}_segmentation.nii.gz"

    volumes = {}
    shape = None
    for name, path, key in [
        ("SAR", sar_path, "sar"),
        ("Temperature", temp_path, "temp"),
        ("Segmentation", seg_path, "seg"),
    ]:
        vol, sh = _load_nifti(path)
        if vol is not None:
            volumes[key] = (name, vol)
            if shape is None:
                shape = sh

    if not volumes:
        st.caption("No NIfTI data found (SAR, temperature, or segmentation).")
        return

    nx, ny, nz = shape
    k_default = nz // 2
    slice_key = f"slice_z_{run_id or output_base}"
    k = st.slider(
        "Axial slice (z)",
        0,
        nz - 1,
        k_default,
        key=slice_key,
        help="Select slice index along the z (axial) axis.",
    )

    cols = st.columns(len(volumes))
    for idx, (key, (name, vol)) in enumerate(volumes.items()):
        with cols[idx]:
            slice_2d = vol[:, :, k]
            if key == "seg":
                # Discrete labels 0..4: use normalized values for colorscale
                z_plot = np.clip(slice_2d, 0, 4).astype(np.float64) / 4.0
                fig = go.Figure(
                    data=go.Heatmap(
                        z=z_plot.T,
                        colorscale=_SEGMENTATION_COLORSCALE,
                        showscale=True,
                    )
                )
            else:
                fig = go.Figure(data=go.Heatmap(z=slice_2d.T, colorscale="Viridis"))
            fig.update_layout(
                title=f"{name} (z={k})",
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(scaleanchor="y", constrain="domain"),
                yaxis=dict(constrain="domain"),
            )
            st.plotly_chart(fig, use_container_width=True)


def load_time_series(data_dir, output_base):
    """Load time_series.json if present. Returns dict with time_step, max/mean SAR and temperature, or None."""
    path = Path(data_dir) / f"{output_base}_time_series.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_frames_chunk(_data_dir_str, _output_base, kind, part):
    """
    Load one NPZ chunk of SAR_frames or Temperature_frames.
    kind is 'SAR_frames' or 'Temperature_frames'. Returns array (chunk_len, nx, ny, nz).
    """
    data_dir = Path(_data_dir_str)
    if kind == "SAR_frames":
        path = data_dir / "SAR_frames" / f"{_output_base}_SAR_frames_part{part}.npz"
        key = "SAR_frames"
    else:
        path = (
            data_dir
            / "Temperature_frames"
            / f"{_output_base}_Temperature_frames_part{part}.npz"
        )
        key = "Temperature_frames"
    if not path.exists():
        return None
    try:
        data = np.load(path)
        return data[key]
    except Exception:
        return None


def render_time_series_plots(data_dir, output_base):
    """Line plots: time step vs max/mean SAR and time step vs max/mean temperature."""
    ts = load_time_series(data_dir, output_base)
    if not ts or "time_step" not in ts:
        st.caption("No time series data (run may predate time_series.json).")
        return
    steps = ts["time_step"]
    if not steps:
        return
    df = pd.DataFrame(
        {
            "Time step": steps,
            "Max SAR (W/kg)": ts.get("max_sar_W_per_kg", []),
            "Mean SAR (W/kg)": ts.get("mean_sar_W_per_kg", []),
            "Max temperature (°C)": ts.get("max_temperature_C", []),
            "Mean temperature (°C)": ts.get("mean_temperature_C", []),
        }
    )
    col1, col2 = st.columns(2)
    with col1:
        fig_sar = go.Figure()
        fig_sar.add_trace(
            go.Scatter(
                x=df["Time step"], y=df["Max SAR (W/kg)"], name="Max SAR", mode="lines"
            )
        )
        fig_sar.add_trace(
            go.Scatter(
                x=df["Time step"],
                y=df["Mean SAR (W/kg)"],
                name="Mean SAR",
                mode="lines",
            )
        )
        fig_sar.update_layout(
            title="Time step vs SAR",
            xaxis_title="Time step",
            yaxis_title="SAR (W/kg)",
            height=360,
        )
        st.plotly_chart(fig_sar, use_container_width=True)
    with col2:
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(
                x=df["Time step"],
                y=df["Max temperature (°C)"],
                name="Max temperature",
                mode="lines",
            )
        )
        fig_temp.add_trace(
            go.Scatter(
                x=df["Time step"],
                y=df["Mean temperature (°C)"],
                name="Mean temperature",
                mode="lines",
            )
        )
        fig_temp.update_layout(
            title="Time step vs temperature",
            xaxis_title="Time step",
            yaxis_title="Temperature (°C)",
            height=360,
        )
        st.plotly_chart(fig_temp, use_container_width=True)


def render_timestep_viewer(data_dir, output_base, meta, run_id=None):
    """
    Slider-controlled viewer for per-frame SAR and Temperature volumes (from NPZ chunks).
    Shows axial mid-slice for the selected frame index.
    """
    data_dir = Path(data_dir)
    n_frames = meta.get("n_frames") or 0
    chunk_size = meta.get("E_frames_chunk_size", 20)
    sar_parts = meta.get("SAR_frames_n_parts") or 0
    temp_parts = meta.get("Temperature_frames_n_parts") or 0
    grid = meta.get("grid_shape") or []
    if len(grid) != 3:
        grid = [0, 0, 0]
    nx, ny, nz = grid
    if n_frames <= 0 or (sar_parts <= 0 and temp_parts <= 0):
        st.caption(
            "No per-frame SAR/Temperature data (run may predate SAR_frames/Temperature_frames)."
        )
        return
    frame_key = f"timestep_frame_{run_id or output_base}"
    frame_idx = st.slider(
        "Frame index",
        0,
        n_frames - 1,
        min(n_frames // 2, n_frames - 1) if n_frames else 0,
        key=frame_key,
        help="Select which saved frame to display (each frame corresponds to a time step).",
    )
    part = frame_idx // chunk_size
    local_idx = frame_idx % chunk_size
    data_dir_str = str(data_dir)
    cols = st.columns(2)
    if sar_parts > 0 and part < sar_parts:
        chunk_sar = load_frames_chunk(data_dir_str, output_base, "SAR_frames", part)
        if chunk_sar is not None and local_idx < len(chunk_sar):
            vol = chunk_sar[local_idx]
            k = vol.shape[2] // 2
            slice_2d = vol[:, :, k]
            with cols[0]:
                fig = go.Figure(data=go.Heatmap(z=slice_2d.T, colorscale="Viridis"))
                fig.update_layout(
                    title=f"SAR frame {frame_idx} (z={k})",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(scaleanchor="y", constrain="domain"),
                    yaxis=dict(constrain="domain"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            with cols[0]:
                st.caption("SAR frame not available.")
    else:
        with cols[0]:
            st.caption("No SAR frames.")
    if temp_parts > 0 and part < temp_parts:
        chunk_temp = load_frames_chunk(
            data_dir_str, output_base, "Temperature_frames", part
        )
        if chunk_temp is not None and local_idx < len(chunk_temp):
            vol = chunk_temp[local_idx]
            k = vol.shape[2] // 2
            slice_2d = vol[:, :, k]
            with cols[1]:
                fig = go.Figure(data=go.Heatmap(z=slice_2d.T, colorscale="Viridis"))
                fig.update_layout(
                    title=f"Temperature frame {frame_idx} (z={k})",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(scaleanchor="y", constrain="domain"),
                    yaxis=dict(constrain="domain"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            with cols[1]:
                st.caption("Temperature frame not available.")
    else:
        with cols[1]:
            st.caption("No temperature frames.")


def render_nifti_slice(run, output_base, data_dir):
    """Sidebar: quick link to Slice viewer tab when NIfTI data exists."""
    sar_path = data_dir / f"{output_base}_SAR.nii.gz"
    temp_path = data_dir / f"{output_base}_temperature.nii.gz"
    seg_path = data_dir / f"{output_base}_segmentation.nii.gz"
    if any(p.exists() for p in (sar_path, temp_path, seg_path)):
        st.caption(
            "Use the **Slice viewer** tab to browse SAR, Temperature, and Segmentation slices."
        )


# -----------------------------------------------------------------------------
# Run simulation (upload MRI + CLI params + subprocess)
# -----------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = "best_model.pth"


def _assign_modality_from_filename(name):
    """Return 'flair', 't1', 't1ce', or 't2' if name (lowercase) matches, else None. Prefer t1ce before t1."""
    low = (name or "").lower()
    if "flair" in low:
        return "flair"
    if "t1ce" in low:
        return "t1ce"
    if "t2" in low:
        return "t2"
    if "t1" in low:
        return "t1"
    return None


def _modality_files_complete(uploaded_files):
    """Check that we can assign exactly flair, t1, t1ce, t2 from the uploaded filenames."""
    assigned = set()
    for u in uploaded_files:
        key = _assign_modality_from_filename(u.name)
        if key and key not in assigned:
            assigned.add(key)
    return assigned == {"flair", "t1", "t1ce", "t2"}


def _read_progress_json():
    """Load progress JSON from results/uploads/last_run_progress.json. Return dict or None."""
    path = get_results_root() / "uploads" / "last_run_progress.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.fragment(run_every=timedelta(seconds=3))
def _render_live_progress():
    """Live progress UI: progress bar, phases, message, log tail. Reruns every 3s when on Run simulation tab."""
    progress = _read_progress_json()
    log_path = get_results_root() / "uploads" / "last_run.log"
    run_pid = st.session_state.get("run_sim_pid")
    if run_pid is not None:
        try:
            os.kill(run_pid, 0)
            st.info("Simulation **running**. Progress updates automatically below.")
        except (OSError, AttributeError):
            st.session_state["run_sim_pid"] = None

    if progress:
        percent = progress.get("percent", 0)
        phase = progress.get("phase", "")
        message = progress.get("message", "")
        phases_done = progress.get("phases_done") or []
        extra = {
            k: v
            for k, v in progress.items()
            if k not in ("phase", "message", "percent", "phases_done", "updated_at")
        }

        st.progress(percent / 100.0)
        st.markdown(f"**{phase.replace('_', ' ').title()}** — {message}")
        if extra.get("grid_shape"):
            st.caption(
                f"Grid: {extra['grid_shape'][0]}×{extra['grid_shape'][1]}×{extra['grid_shape'][2]}"
            )
        if extra.get("time_step") is not None and extra.get("time_steps"):
            st.caption(f"FDTD step {extra['time_step']} / {extra['time_steps']}")

        # Phase checklist
        all_phases = [
            "setup",
            "segmentation",
            "antenna_optimization",
            "fdtd_simulation",
            "sar_computation",
            "thermal_solver",
            "saving_and_animations",
            "complete",
        ]
        done_set = set(phases_done)
        cols = st.columns(min(len(all_phases), 4))
        for i, p in enumerate(all_phases):
            with cols[i % len(cols)]:
                label = p.replace("_", " ").title()
                if p in done_set:
                    st.success(f"✓ {label}")
                elif p == phase:
                    st.spinner(text=f"{label}", show_time=False, width="content")
                else:
                    st.spinner(label)
        if progress.get("phase") == "complete":
            st.success("Simulation finished. Refresh the run list to see the new run.")
    else:
        st.caption("No progress file yet. Start a simulation to see live progress.")

    if log_path.exists():
        with st.expander("View log (last 500 lines)", expanded=False):
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()
                tail = 500
                if len(lines) > tail:
                    lines = lines[-tail:]
                    st.caption(f"Last {tail} lines.")
                st.text_area(
                    "Log",
                    value="".join(lines),
                    height=300,
                    key="run_log_fragment",
                    disabled=True,
                )
            except Exception as e:
                st.caption(f"Could not read log: {e}")
    else:
        st.caption("No log yet. Start a simulation to see output here.")


def _build_engine_argv(
    repo_root,
    seg_path=None,
    modalities_dir=None,
    checkpoint=None,
    no_normal_brain=False,
    optimize_antenna=False,
    f0_hz=100e6,
    opt_time_steps=700,
    opt_phase_steps=24,
    opt_amp_steps=9,
    opt_amp_min=0.2,
    opt_amp_max=2.5,
    opt_refine_iters=8,
    opt_multi_start=3,
    opt_penalty_weight=0.0,
    opt_freq_sweep=None,
    opt_geom_offsets=None,
    opt_geom_zplanes=None,
    pulse_amplitude=100.0,
    time_steps=500,
    max_dim=120,
):
    """Build argv list for fdtd_brain_simulation_engine.py."""
    engine_path = repo_root / "fdtd_brain_simulation_engine.py"
    if not engine_path.exists():
        return None
    # -u: unbuffered stdout so log file updates in real time
    argv = [sys.executable, "-u", str(engine_path)]
    if modalities_dir is not None:
        argv.append("--modalities-dir")
        argv.append(str(modalities_dir))
        if checkpoint:
            argv.extend(["--checkpoint", str(checkpoint)])
        if no_normal_brain:
            argv.append("--no-normal-brain")
    elif seg_path is not None:
        argv.append(str(seg_path))
    else:
        return None
    argv.extend(["--time-steps", str(time_steps)])
    argv.extend(["--max-dim", str(max_dim)])
    if optimize_antenna:
        argv.append("--optimize-antenna")
        argv.extend(["--f0", str(f0_hz)])
        argv.extend(["--opt-time-steps", str(opt_time_steps)])
        argv.extend(["--opt-phase-steps", str(opt_phase_steps)])
        argv.extend(["--opt-amp-steps", str(opt_amp_steps)])
        argv.extend(["--opt-amp-min", str(opt_amp_min)])
        argv.extend(["--opt-amp-max", str(opt_amp_max)])
        argv.extend(["--opt-refine-iters", str(opt_refine_iters)])
        argv.extend(["--opt-multi-start", str(opt_multi_start)])
        argv.extend(["--opt-penalty-weight", str(opt_penalty_weight)])
        if opt_freq_sweep:
            argv.append("--opt-freq-sweep")
            argv.extend([str(f) for f in opt_freq_sweep])
        if opt_geom_offsets is not None and len(opt_geom_offsets) > 0:
            argv.append("--opt-geom-offsets")
            argv.extend([str(o) for o in opt_geom_offsets])
        if opt_geom_zplanes is not None and len(opt_geom_zplanes) > 0:
            argv.append("--opt-geom-zplanes")
            argv.extend([str(z) for z in opt_geom_zplanes])
    else:
        argv.extend(["--pulse-amplitude", str(pulse_amplitude)])
    return argv


def render_run_simulation():
    """Run simulation tab: MRI upload, CLI parameter form, and Run button."""
    repo_root = get_repo_root()
    uploads_dir = get_uploads_dir()
    uploads_dir.mkdir(parents=True, exist_ok=True)
    engine_path = repo_root / "fdtd_brain_simulation_engine.py"
    if not engine_path.exists():
        st.error(
            f"Engine not found at `{engine_path}`. Run the dashboard from the thesis repo root."
        )
        return

    st.markdown("Upload MRI data, set parameters, then click **Run simulation**.")

    input_mode = st.radio(
        "Input",
        [
            "Pre-segmented NIfTI (1 file)",
            "BraTS modalities (4 files: FLAIR, T1, T1CE, T2)",
            "Modalities directory (ZIP or multiple NIfTI files)",
        ],
        key="run_sim_input_mode",
    )
    use_modalities = input_mode.startswith("BraTS") or input_mode.startswith(
        "Modalities directory"
    )
    modalities_ready = False
    seg_ready = False
    modalities_dir_ready = (
        False  # True when ZIP or multi-file for --modalities-dir is ready
    )
    seg_file = None
    flair = t1 = t1ce = t2 = None

    if input_mode == "Pre-segmented NIfTI (1 file)":
        st.caption("Upload one BraTS-style segmentation NIfTI (labels 0,1,2,3 or 0–4).")
        seg_file = st.file_uploader(
            "Segmentation NIfTI", type=["nii", "nii.gz"], key="seg"
        )
        seg_ready = seg_file is not None
    elif input_mode == "BraTS modalities (4 files: FLAIR, T1, T1CE, T2)":
        st.caption(
            "Upload four NIfTI files. They will be saved to disk when you click Run."
        )
        flair = st.file_uploader("FLAIR", type=["nii", "nii.gz"], key="flair")
        t1 = st.file_uploader("T1", type=["nii", "nii.gz"], key="t1")
        t1ce = st.file_uploader("T1ce", type=["nii", "nii.gz"], key="t1ce")
        t2 = st.file_uploader("T2", type=["nii", "nii.gz"], key="t2")
        modalities_ready = flair and t1 and t1ce and t2
    else:
        # Modalities directory: ZIP or multiple files → folder for --modalities-dir
        st.caption(
            "Upload a **ZIP** containing modality NIfTIs, or **multiple NIfTI files** "
            "(FLAIR, T1, T1ce, T2). They will be written to a folder and passed as `--modalities-dir`."
        )
        mod_dir_upload_type = st.radio(
            "Upload as",
            ["ZIP file (folder of NIfTIs)", "Multiple NIfTI files"],
            horizontal=True,
            key="mod_dir_upload_type",
        )
        if mod_dir_upload_type.startswith("ZIP"):
            zip_upload = st.file_uploader(
                "ZIP file",
                type=["zip"],
                key="mod_dir_zip",
            )
            modalities_dir_ready = zip_upload is not None
        else:
            multi_files = st.file_uploader(
                "Select 4 NIfTI files (FLAIR, T1, T1ce, T2)",
                type=["nii", "nii.gz"],
                accept_multiple_files=True,
                key="mod_dir_multi",
            )
            # Engine expects flair.nii, t1.nii, t1ce.nii, t2.nii (or *_flair.nii etc.)
            modalities_dir_ready = (
                multi_files is not None
                and len(multi_files) >= 4
                and _modality_files_complete(multi_files)
            )
            if multi_files and len(multi_files) < 4:
                st.warning("Upload at least 4 NIfTI files (FLAIR, T1, T1ce, T2).")
            elif (
                multi_files
                and len(multi_files) >= 4
                and not _modality_files_complete(multi_files)
            ):
                st.warning(
                    "Could not assign 4 modalities from filenames. "
                    "Include 'flair', 't1', 't1ce', 't2' in the file names (case-insensitive)."
                )

    st.divider()
    st.subheader("Parameters (CLI equivalents)")

    with st.expander("Basic", expanded=True):
        checkpoint_path = st.text_input(
            "Checkpoint path (for BraTS segmentation)",
            value=str(repo_root / _DEFAULT_CHECKPOINT),
            help="Path to 3D U-Net .pth. Used only when using BraTS modalities.",
            key="checkpoint",
        )
        no_normal_brain = st.checkbox(
            "No normal brain (BraTS only: keep only tumor 1–3 and background 0)",
            value=False,
            key="no_normal_brain",
        )
        time_steps = st.number_input(
            "Time steps (standard run)",
            min_value=1,
            value=500,
            step=100,
            help="FDTD time steps for non-optimized run. Ignored when Optimize antenna is on.",
            key="time_steps",
        )
        max_dim = st.number_input(
            "Max grid dimension",
            min_value=10,
            value=120,
            step=10,
            help="Segmentation is downsampled if any dimension exceeds this.",
            key="max_dim",
        )
        optimize_antenna = st.checkbox(
            "Optimize antenna (4-quadrant APA optimization)",
            value=False,
            key="optimize_antenna",
        )
        if not optimize_antenna:
            pulse_amplitude = st.number_input(
                "Pulse amplitude (standard run)",
                min_value=0.1,
                value=100.0,
                step=10.0,
                key="pulse_amp",
            )
        else:
            pulse_amplitude = 100.0

    if optimize_antenna:
        with st.expander("Antenna optimization", expanded=True):
            f0_mhz = st.number_input(
                "Frequency (MHz)", min_value=10.0, value=100.0, step=10.0, key="f0_mhz"
            )
            f0_hz = f0_mhz * 1e6
            opt_time_steps = st.number_input(
                "Opt time steps", min_value=1, value=700, step=100, key="opt_ts"
            )
            opt_phase_steps = st.number_input(
                "Opt phase steps", min_value=1, value=24, key="opt_ps"
            )
            opt_amp_steps = st.number_input(
                "Opt amp steps", min_value=1, value=9, key="opt_as"
            )
            opt_amp_min = st.number_input(
                "Opt amp min", value=0.2, step=0.1, key="opt_amin"
            )
            opt_amp_max = st.number_input(
                "Opt amp max", value=2.5, step=0.1, key="opt_amax"
            )
            opt_refine_iters = st.number_input(
                "Opt refine iters", min_value=0, value=8, key="opt_ri"
            )
            opt_multi_start = st.number_input(
                "Opt multi-start", min_value=1, value=3, key="opt_ms"
            )
            opt_penalty_weight = st.number_input(
                "Opt penalty weight", value=0.0, step=0.1, key="opt_pw"
            )
            opt_freq_sweep_str = st.text_input(
                "Opt freq sweep (Hz, space-separated)",
                value="",
                placeholder="e.g. 70e6 100e6 130e6 170e6 200e6",
                help="Frequencies to sweep; best f0 auto-selected. Leave empty to use Frequency above.",
                key="opt_freq_sweep",
            )
            opt_geom_offsets_str = st.text_input(
                "Opt geom offsets (cells from PML)",
                value="",
                placeholder="e.g. 10 15 20 25",
                key="opt_geom_offsets",
            )
            opt_geom_zplanes_str = st.text_input(
                "Opt geom z-planes (indices)",
                value="",
                placeholder="e.g. 30 41 50",
                key="opt_geom_zplanes",
            )
    else:
        f0_hz = 100e6
        opt_time_steps = 700
        opt_phase_steps = 24
        opt_amp_steps = 9
        opt_amp_min = 0.2
        opt_amp_max = 2.5
        opt_refine_iters = 8
        opt_multi_start = 3
        opt_penalty_weight = 0.0
        opt_freq_sweep_str = ""
        opt_geom_offsets_str = ""
        opt_geom_zplanes_str = ""

    # Parse optional list params (only used when optimize_antenna)
    def _parse_float_list(s):
        if not s or not s.strip():
            return None
        out = []
        for x in s.strip().split():
            try:
                out.append(float(x))
            except ValueError:
                pass
        return out if out else None

    def _parse_int_list(s):
        if not s or not s.strip():
            return None
        out = []
        for x in s.strip().split():
            try:
                out.append(int(x))
            except ValueError:
                pass
        return out if out else None

    opt_freq_sweep = _parse_float_list(opt_freq_sweep_str) if optimize_antenna else None
    opt_geom_offsets = (
        _parse_int_list(opt_geom_offsets_str) if optimize_antenna else None
    )
    opt_geom_zplanes = (
        _parse_int_list(opt_geom_zplanes_str) if optimize_antenna else None
    )

    run_clicked = st.button("Run simulation", type="primary", key="run_sim_btn")
    if run_clicked:
        can_run = (
            (input_mode == "Pre-segmented NIfTI (1 file)" and seg_ready)
            or (
                input_mode == "BraTS modalities (4 files: FLAIR, T1, T1CE, T2)"
                and modalities_ready
            )
            or (
                input_mode == "Modalities directory (ZIP or multiple NIfTI files)"
                and modalities_dir_ready
            )
        )
        if not can_run:
            if input_mode == "Pre-segmented NIfTI (1 file)":
                st.error("Upload a segmentation NIfTI file.")
            elif input_mode == "BraTS modalities (4 files: FLAIR, T1, T1CE, T2)":
                st.error("Upload all four BraTS modality files.")
            else:
                st.error(
                    "Upload a ZIP file or at least 4 NIfTI files with FLAIR/T1/T1ce/T2 in their names."
                )
        else:
            import time as _time

            run_id = _time.strftime("%Y%m%d_%H%M%S", _time.gmtime())
            run_upload_dir = uploads_dir / run_id
            run_upload_dir.mkdir(parents=True, exist_ok=True)
            modalities_dir_saved = None
            seg_path_saved = None

            if input_mode == "Pre-segmented NIfTI (1 file)":
                ext = ".nii.gz" if (seg_file.name or "").endswith(".gz") else ".nii"
                seg_path_saved = run_upload_dir / f"seg{ext}"
                seg_path_saved.write_bytes(seg_file.getvalue())
            elif input_mode == "BraTS modalities (4 files: FLAIR, T1, T1CE, T2)":
                for name, upl in [
                    ("flair", flair),
                    ("t1", t1),
                    ("t1ce", t1ce),
                    ("t2", t2),
                ]:
                    (run_upload_dir / f"{name}.nii").write_bytes(upl.getvalue())
                modalities_dir_saved = run_upload_dir
            else:
                # Modalities directory: ZIP or multiple files
                if mod_dir_upload_type.startswith("ZIP"):
                    with zipfile.ZipFile(zip_upload, "r") as zf:
                        zf.extractall(run_upload_dir)
                    modalities_dir_saved = run_upload_dir
                else:
                    # Assign each file to flair/t1/t1ce/t2 and save as flair.nii, t1.nii, t1ce.nii, t2.nii
                    for u in multi_files:
                        key = _assign_modality_from_filename(u.name)
                        if key:
                            (run_upload_dir / f"{key}.nii").write_bytes(u.getvalue())
                    modalities_dir_saved = run_upload_dir
            argv = _build_engine_argv(
                repo_root,
                seg_path=seg_path_saved,
                modalities_dir=(
                    str(modalities_dir_saved) if modalities_dir_saved else None
                ),
                checkpoint=checkpoint_path or None,
                no_normal_brain=no_normal_brain,
                optimize_antenna=optimize_antenna,
                f0_hz=f0_hz,
                opt_time_steps=opt_time_steps,
                opt_phase_steps=opt_phase_steps,
                opt_amp_steps=opt_amp_steps,
                opt_amp_min=opt_amp_min,
                opt_amp_max=opt_amp_max,
                opt_refine_iters=opt_refine_iters,
                opt_multi_start=opt_multi_start,
                opt_penalty_weight=opt_penalty_weight,
                opt_freq_sweep=opt_freq_sweep,
                opt_geom_offsets=opt_geom_offsets,
                opt_geom_zplanes=opt_geom_zplanes,
                pulse_amplitude=pulse_amplitude,
                time_steps=time_steps,
                max_dim=max_dim,
            )
            if argv is None:
                st.error(
                    "Invalid input: provide either segmentation path or modalities directory."
                )
            else:
                log_path = get_results_root() / "uploads" / "last_run.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w") as logf:
                    proc = subprocess.Popen(
                        argv,
                        cwd=str(repo_root),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                    )
                if "run_sim_pid" not in st.session_state:
                    st.session_state["run_sim_pid"] = None
                st.session_state["run_sim_pid"] = proc.pid
                st.session_state["run_sim_log_path"] = str(log_path)
                st.success(
                    f"Simulation started (PID {proc.pid}). Output will appear under `results/`. "
                    "Refresh the page or run list to see the new run when it finishes."
                )
                st.caption(f"Log: `{log_path}`")

    # Live progress (fragment reruns every 3s so progress updates without manual refresh)
    st.divider()
    st.subheader("Simulation progress")
    _render_live_progress()


def get_media_dirs(data_dir):
    """Return (images_dir, animations_dir) from run data dir (results/{run_id}/data -> images + animations)."""
    parent = data_dir.parent if hasattr(data_dir, "parent") else Path(data_dir).parent
    return parent / "images", parent / "animations"


def render_images_and_animations(runs_data):
    """Render PNGs from images/ and MP4s from animations/ for selected run(s)."""
    N_IMAGE_COLS = 3
    N_VIDEO_COLS = 2

    for d in runs_data:
        run_id = d["run_id"]
        data_dir = d["data_dir"]
        output_base = d["output_base"]
        images_dir, animations_dir = get_media_dirs(data_dir)

        if len(runs_data) > 1:
            st.subheader(f"Run: {run_id}")

        # Images: 3-column grid with labels
        st.markdown("#### Images")
        if not images_dir.exists():
            st.caption(f"No images folder at `{images_dir}`.")
        else:
            pngs = sorted(images_dir.glob("*.png"))
            if not pngs:
                st.caption("No PNG files in images folder.")
            else:
                matching = [p for p in pngs if output_base in p.name]
                to_show = matching if matching else pngs
                for i in range(0, len(to_show), N_IMAGE_COLS):
                    row = to_show[i : i + N_IMAGE_COLS]
                    cols = st.columns(N_IMAGE_COLS)
                    for j, path in enumerate(row):
                        with cols[j]:
                            st.caption(path.name)
                            try:
                                st.image(str(path), use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not load: {e}")

        # Animations: 2-column grid with labels
        st.markdown("#### Animations")
        if not animations_dir.exists():
            st.caption(f"No animations folder at `{animations_dir}`.")
        else:
            mp4s = sorted(animations_dir.glob("*.mp4"))
            if not mp4s:
                st.caption("No MP4 files in animations folder.")
            else:
                matching = [p for p in mp4s if output_base in p.name]
                to_show = matching if matching else mp4s
                for i in range(0, len(to_show), N_VIDEO_COLS):
                    row = to_show[i : i + N_VIDEO_COLS]
                    cols = st.columns(N_VIDEO_COLS)
                    for j, path in enumerate(row):
                        with cols[j]:
                            st.caption(path.name)
                            try:
                                st.video(str(path))
                            except Exception as e:
                                st.error(f"Could not load: {e}")

        if len(runs_data) > 1:
            st.divider()


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------

st.set_page_config(page_title="FDTD Results", page_icon="⚛️", layout="wide")

st.title("FDTD Simulation Results Dashboard")
st.caption(
    "Renders metadata.json, performance.json, and optional NIfTI from fdtd_brain_simulation_engine.py"
)

results_root = get_results_root()
runs = discover_runs(results_root)

if not runs:
    st.sidebar.warning("No runs yet. Use the **Run simulation** tab to create one.")

# Sidebar: run selection
st.sidebar.header("Run selection")
run_labels = []
run_by_label = {}
for r in runs:
    label = (
        f"{r['run_id']} ({r['output_base']})"
        if len([x for x in runs if x["run_id"] == r["run_id"]]) > 1
        else r["run_id"]
    )
    run_labels.append(label)
    run_by_label[label] = r

runs_data = []
if runs:
    compare_all = st.sidebar.checkbox("Compare all runs (scalability)", value=False)
    if compare_all:
        selected_run_ids = list({r["run_id"] for r in runs})
    else:
        selected = st.sidebar.selectbox(
            "Select run", run_labels, format_func=lambda x: x
        )
        selected_run_ids = [run_by_label[selected]["run_id"]]
    if compare_all:
        for run in runs:
            try:
                runs_data.append(load_run_data(run))
            except Exception as e:
                st.sidebar.error(f"Failed to load {run['run_id']}: {e}")
    else:
        run = run_by_label[selected]
        try:
            runs_data.append(load_run_data(run))
        except Exception as e:
            st.sidebar.error(f"Failed to load {run['run_id']}: {e}")
else:
    compare_all = False
    selected_run_ids = []

# Tabs
(
    tab_run_simulation,
    tab_overview,
    tab_performance,
    tab_region,
    tab_tissue,
    tab_antenna,
    tab_scalability,
    tab_slice_viewer,
    tab_timestep_viewer,
    tab_media,
) = st.tabs(
    [
        "Run simulation",
        "Overview",
        "Performance",
        "Region stats (SAR/T)",
        "Tissue properties",
        "Antenna",
        "Scalability",
        "Slice viewer",
        "Timestep viewer",
        "Images & Animations",
    ]
)

with tab_run_simulation:
    st.header("Run simulation")
    render_run_simulation()

with tab_overview:
    st.header("Overview")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    for i, d in enumerate(runs_data):
        if len(runs_data) > 1:
            st.subheader(d["run_id"])
        render_kpis(d["performance"], d["run_id"])
        meta = d["metadata"]
        gs = meta.get("grid_shape") or []
        st.write(
            f"**Grid:** {gs[0]}×{gs[1]}×{gs[2]} | **Voxel size:** {meta.get('voxel_size_m')} m | **Time steps:** {meta.get('time_steps')}"
        )

with tab_performance:
    st.header("Performance")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    for i, d in enumerate(runs_data):
        if len(runs_data) > 1:
            st.subheader(d["run_id"])
        render_time_breakdown(d["performance"])

with tab_region:
    st.header("Region stats (SAR & temperature)")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    for i, d in enumerate(runs_data):
        if len(runs_data) > 1:
            st.subheader(d["run_id"])
        render_region_stats(d["metadata"])

with tab_tissue:
    st.header("Tissue & dielectric properties")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    for i, d in enumerate(runs_data):
        if len(runs_data) > 1:
            st.subheader(d["run_id"])
        render_tissue_properties(d["metadata"])

with tab_antenna:
    st.header("Antenna parameters")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    for i, d in enumerate(runs_data):
        if len(runs_data) > 1:
            st.subheader(d["run_id"])
        render_antenna_params(d["metadata"])

with tab_scalability:
    st.header("Scalability (multi-run)")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    else:
        render_scalability(runs_data)

with tab_slice_viewer:
    st.header("Slice viewer (SAR, Temperature, Segmentation)")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    elif compare_all or len(runs_data) != 1:
        st.caption("Select a single run in the sidebar to browse NIfTI slices.")
    else:
        d = runs_data[0]
        render_slice_viewer(d["data_dir"], d["output_base"], d["run_id"])

with tab_timestep_viewer:
    st.header("Timestep viewer (time series & per-frame volumes)")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    elif compare_all or len(runs_data) != 1:
        st.caption(
            "Select a single run in the sidebar to view time series and per-frame SAR/Temperature."
        )
    else:
        d = runs_data[0]
        meta = d["metadata"]
        data_dir = d["data_dir"]
        output_base = d["output_base"]
        run_id = d["run_id"]
        st.subheader("Time step vs SAR / temperature")
        render_time_series_plots(data_dir, output_base)
        st.subheader("Per-frame SAR & temperature (axial slice)")
        render_timestep_viewer(data_dir, output_base, meta, run_id)

with tab_media:
    st.header("Images & Animations")
    if not runs_data:
        st.info("No runs yet. Use the **Run simulation** tab to create one.")
    else:
        render_images_and_animations(runs_data)

# Optional NIfTI preview for first selected run
if runs_data and not compare_all:
    d = runs_data[0]
    st.sidebar.header("Data preview")
    render_nifti_slice(d, d["output_base"], d["data_dir"])
