#!/usr/bin/env python3
"""
Build PAPER/generated/paper_results_section.tex from paper_bundle_manifest.json.

Run by run_paper_bundle.py after simulations. Requires \\graphicspath{{figures/results_three_runs/}}
in the main LaTeX preamble (see official_paper.tex).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def _num_latex(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"\\num{{{x:.4e}}}"


def _num_fixed(x: Optional[float], places: int = 2) -> str:
    """Fixed-point \\num{...} for wall time, temperature, etc."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"\\num{{{x:.{places}f}}}"


def _peak_temperature_C(meta: Dict[str, Any]) -> Optional[float]:
    rs = meta.get("region_stats") or {}
    vals: List[float] = []
    for key in ("temperature_tumor_C", "temperature_non_tumor_tissue_C"):
        t = rs.get(key)
        if isinstance(t, dict) and t.get("max") is not None:
            vals.append(float(t["max"]))
    return max(vals) if vals else None


def _wall_time_s_meta(meta: Dict[str, Any]) -> Optional[float]:
    """
    End-to-end pipeline wall time in seconds for tables.

    Prefer ``performance.total_simulation_time_s`` (segmentation through save,
    figures, and animations). Fall back to legacy ``total_wall_time_s``, which
    was recorded before the save/export phase on older runs.
    """
    perf = meta.get("performance") or {}
    v = perf.get("total_simulation_time_s")
    if v is None:
        v = perf.get("total_wall_time_s")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _format_duration_hms(seconds: Optional[float]) -> str:
    """Wall-clock duration as H:MM:SS for LaTeX tables (no siunitx dependency)."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "---"
    s = max(0, int(round(float(seconds))))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def _short_proj_from_filename(fn: str) -> str:
    if "axial_maxz" in fn:
        return "axial"
    if "sagittal_maxx" in fn:
        return "sagittal"
    if "coronal_maxy" in fn:
        return "coronal"
    return "view"


def _grid_shape_tex(shape: List[int]) -> str:
    return f"{shape[0]} $\\times$ {shape[1]} $\\times$ {shape[2]}"


def _excitation_row(meta: Dict[str, Any]) -> str:
    if meta.get("antenna_optimized"):
        f0 = meta.get("optimized_f0_Hz", meta.get("pulse_freq_Hz", 100e6))
        mhz = f0 / 1e6
        return f"Optimized 4-quadrant APA (\\SI{{{mhz:.0f}}}{{\\mega\\hertz}})"
    pt = meta.get("pulse_type", "")
    if pt == "gaussian":
        d = meta.get("prop_direction", "+y")
        return f"Volume Gaussian ({d})"
    if pt == "cw":
        cp = meta.get("cw_periods", "")
        cp_s = f", {cp} periods" if cp else ""
        return f"CW sinusoidal\\allowbreak{cp_s}"
    return _tex_escape(str(pt))


def _fit_power_law(
    x: List[float], y: List[float]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    import numpy as np

    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mask = (xa > 0) & (ya > 0)
    if np.sum(mask) < 2:
        return None, None, None
    logx = np.log(xa[mask])
    logy = np.log(ya[mask])
    b, loga = np.polyfit(logx, logy, 1)
    a = float(np.exp(loga))
    pred = loga + b * logx
    ss_res = float(np.sum((logy - pred) ** 2))
    ss_tot = float(np.sum((logy - np.mean(logy)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return a, float(b), float(r2)


def generate_section(manifest_path: Path, paper_dir: Path) -> Path:
    man = _load_json(manifest_path)
    modalities = man["modalities_dir"]
    base = man.get("output_base")
    runs: Dict[str, Any] = man["runs"]
    brats: List[Dict[str, Any]] = list(man.get("brats_cw_validation") or [])
    run_figure_sets: List[Dict[str, Any]] = list(man.get("run_figure_sets") or [])

    def meta_for(key: str) -> Optional[Dict[str, Any]]:
        run_cfg = runs.get(key) or {}
        rdir_s = run_cfg.get("results_dir")
        ob = run_cfg.get("output_base") or base
        if not rdir_s or not ob:
            return None
        rdir = Path(rdir_s)
        data = rdir / "data"
        mp = data / f"{ob}_metadata.json"
        if not mp.is_file():
            return None
        return _load_json(mp)

    def meta_brats(entry: Dict[str, Any]) -> Dict[str, Any]:
        rdir = Path(entry["results_dir"])
        ob = entry["output_base"]
        mp = rdir / "data" / f"{ob}_metadata.json"
        if not mp.is_file():
            raise FileNotFoundError(f"Missing metadata: {mp}")
        return _load_json(mp)

    m_g = meta_for("gaussian")
    m_c = meta_for("cw")
    m_o = meta_for("optimize")
    has_main_runs = all(m is not None for m in (m_g, m_c, m_o))
    if has_main_runs and not run_figure_sets:
        run_figure_sets = [
            {
                "run_key": "gaussian",
                "run_label": "Gaussian",
                "sar": "sar_gaussian.png",
                "temperature": "temp_gaussian.png",
            },
            {
                "run_key": "cw",
                "run_label": "CW",
                "sar": "sar_cw.png",
                "temperature": "temp_cw.png",
            },
            {
                "run_key": "optimize",
                "run_label": "Optimized APA",
                "sar": "sar_optimized.png",
                "temperature": "temp_optimized.png",
            },
        ]
    has_benchmark = bool((runs.get("benchmark") or {}).get("results_dir"))

    lines: List[str] = []
    lines.append("% -*- tex -*- ")
    lines.append("% Auto-generated results section — do not edit by hand.")
    lines.append(r"\section{Results and Discussion}")
    lines.append("")

    # --- 4.1 Overview ---
    lines.append(r"\subsection{Overview of Experiments}")
    overview_tail = (
        "A separate scalability benchmark with the same anatomy resampled to $N^3$ grids is summarized in "
        "Sec.~\\ref{subsec:scalability}."
    )
    if brats:
        brats_case_ids = ", ".join(
            _tex_escape(str(e.get("case_id", "")))
            for e in brats
            if e.get("case_id") is not None
        )
        brats_case_ids = brats_case_ids or _tex_escape(
            ", ".join(str(i + 1) for i in range(len(brats)))
        )
        overview_tail += (
            " Additional CW-only simulations on other BraTS validation folders are reported in "
            f"Sec.~\\ref{{subsec:brats_cw_validation}} (cases {brats_case_ids})."
        )
    if has_main_runs:
        lines.append(
            "This chapter reports three electromagnetic simulation configurations on the same patient-specific "
            "brain voxel model derived from multimodal MRI (BraTS-style case format). "
            "Runs are: (i) Gaussian volume excitation, (ii) continuous-wave (CW) narrowband excitation, and "
            "(iii) four-quadrant annular phased-array optimization followed by a final FDTD solve using the optimized "
            "sources. All figures and tables in this section are derived from the finalized batch outputs included in this thesis. "
            + overview_tail
        )
    else:
        lines.append(
            "This chapter was generated without the Gaussian/CW/optimized main-run bundle for the primary case "
            "(for example, when using --skip-main-runs). The section below therefore reports only the artifacts "
            "available from the executed bundle configuration."
        )
    lines.append("")

    # --- 4.2 Geometry + config table ---
    def row(label: str, m: Dict[str, Any]) -> str:
        gs = m["grid_shape"]
        dx_mm = float(m["voxel_size_m"]) * 1000.0
        dt = float(m["time_step_s"])
        nt = int(m["time_steps"])
        f0 = m.get("optimized_f0_Hz") or m.get("pulse_freq_Hz", 100e6)
        f0_mhz = f0 / 1e6
        cf = float(m["courant_factor"])
        return (
            f"{label} & {_grid_shape_tex(gs)} & \\num{{{dx_mm:.2f}}} & \\num{{{dt:.3e}}} & {nt} & "
            f"\\num{{{f0_mhz:.0f}}} & \\num{{{cf:.2f}}} \\\\"
        )

    if has_main_runs:
        lines.append(r"\subsection{Simulation Configuration and Input Geometry}")
        lines.append(
            "Table~\\ref{tab:results_config} lists grid spacing, time step, step count, carrier frequency, "
            "and Courant factor. PML thickness follows the dynamic rule in Chapter~3. Tissue labels follow BraTS "
            "conventions (background, necrotic core, edema, enhancing tumor, normal brain)."
        )
        lines.append("")
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.85\linewidth]{geometry_same_case.png}")
        lines.append(
            r"\caption{Representative mid-$z$ slice of voxel labels (tumor vs.\ healthy tissue) for the FDTD domain.}"
        )
        lines.append(r"\label{fig:results_geometry}")
        lines.append(r"\end{figure}")
        lines.append("")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Simulation parameters for the three reported electromagnetic runs (shared anatomy).}"
        )
        lines.append(r"\label{tab:results_config}")
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\hline")
        lines.append(
            r"Run & $N_x\times N_y\times N_z$ & $\Delta x$ (mm) & $\Delta t$ (s) & $N_{\mathrm{steps}}$ & $f_0$ (MHz) & $c_{\mathrm{CFL}}$ \\"
        )
        lines.append(r"\hline")
        lines.append(row("Gaussian", m_g or {}))
        lines.append(row("CW", m_c or {}))
        lines.append(row("Optimized", m_o or {}))
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- 4.3 SAR + temperature maps (systematic per-run); E-field timelines in Sec.~multiview ---
    if has_main_runs:
        lines.append(r"\subsection{Electromagnetic Heating Maps}")
        lines.append(
            "Each run reports SAR and steady-state temperature from the simplified bioheat model "
            "(same anatomy and colormap conventions). "
            "E-field evolution is summarized separately via 15-timestep max-projection timelines "
            "(Sec.~\\ref{subsec:multiview_projection})."
        )
        ordered_sets: List[Dict[str, Any]] = []
        preferred_order = ["gaussian", "cw", "optimize"]
        for key in preferred_order:
            for item in run_figure_sets:
                if str(item.get("run_key")) == key:
                    ordered_sets.append(item)
                    break
        if not ordered_sets:
            ordered_sets = run_figure_sets
        for item in ordered_sets:
            run_key = str(item.get("run_key", "run"))
            run_label = _tex_escape(str(item.get("run_label", run_key)))
            sar_png = str(item.get("sar", ""))
            temp_png = str(item.get("temperature", ""))
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=0.85\linewidth]{{{sar_png}}}")
            lines.append(rf"\caption{{{run_label}: SAR distribution (W/kg).}}")
            lines.append(rf"\label{{fig:maps_sar_{run_key}}}")
            lines.append(r"\end{figure}")
            lines.append("")
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=0.85\linewidth]{{{temp_png}}}")
            lines.append(
                rf"\caption{{{run_label}: temperature ($^\circ$C) from the thermal solve.}}"
            )
            lines.append(rf"\label{{fig:maps_temp_{run_key}}}")
            lines.append(r"\end{figure}")
            lines.append("")

    # --- 4.4 SAR ---

    def sar_stats_rows(label: str, m: Dict[str, Any]) -> str:
        rs = m.get("region_stats") or {}
        t = rs.get("sar_tumor_W_per_kg") or {}
        h = rs.get("sar_non_tumor_tissue_W_per_kg") or {}
        return (
            f"{label} & {_num_latex(t.get('min'))} & {_num_latex(t.get('mean'))} & {_num_latex(t.get('max'))} & "
            f"{_num_latex(h.get('min'))} & {_num_latex(h.get('mean'))} & {_num_latex(h.get('max'))} \\\\"
        )

    if has_main_runs:
        lines.append(r"\subsection{SAR Distributions}")
        lines.append(
            "SAR maps for each configuration are shown in Figs.~\\ref{fig:maps_sar_gaussian}--\\ref{fig:maps_sar_optimize}. "
            "Table~\\ref{tab:results_sar_stats} summarizes min/mean/max SAR in tumor and healthy-tissue masks."
        )
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
        lines.append(
            r"\caption{Region statistics for SAR (W/kg) in tumor vs.\ healthy tissue.}"
        )
        lines.append(r"\label{tab:results_sar_stats}")
        lines.append(r"\resizebox{\linewidth}{!}{%")
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\hline")
        lines.append(
            r"Run & \multicolumn{3}{c}{Tumor} & \multicolumn{3}{c}{Healthy} \\"
        )
        lines.append(r"\cline{2-4}\cline{5-7}")
        lines.append(r" & min & mean & max & min & mean & max \\")
        lines.append(r"\hline")
        lines.append(sar_stats_rows("Gaussian", m_g or {}))
        lines.append(sar_stats_rows("CW", m_c or {}))
        lines.append(sar_stats_rows("Optimized", m_o or {}))
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- 4.5 Temperature ---
    if has_main_runs:
        lines.append(r"\subsection{Temperature Predictions (Simplified Thermal Model)}")
        lines.append(
            "Steady-state temperature fields are shown in Figs.~\\ref{fig:maps_temp_gaussian}--\\ref{fig:maps_temp_optimize} "
            "from the simplified bioheat solve (no perfusion, Chapter~3). Because perfusion and thermoregulation are omitted, absolute "
            "temperatures are expected to overestimate in vivo heating; comparisons across runs are interpreted "
            "qualitatively as relative spatial patterns rather than clinical predictions."
        )
        lines.append("")

    extra = man.get("paper_extra_figures") or {}

    if extra.get("pipeline_phases_waterfall"):
        lines.append(r"\subsection{Pipeline phase timing}")
        lines.append(
            "Figure~\\ref{fig:pipeline_phases_waterfall} shows the relative wall-clock time spent in major "
            "pipeline stages for the CW reference run bundled with this chapter."
        )
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")
        lines.append(
            r"\includegraphics[width=\linewidth]{pipeline_phases_waterfall.png}"
        )
        lines.append(r"\caption{Stacked timeline of logged phase durations (seconds).}")
        lines.append(r"\label{fig:pipeline_phases_waterfall}")
        lines.append(r"\end{figure}")
        lines.append("")

    trips: List[Dict[str, Any]] = list(extra.get("slice_triptychs") or [])
    if trips:
        lines.append(r"\subsection{Representative slice validation}")
        lines.append(
            "The following figures compare anatomy labels, SAR, and absolute temperature "
            "($^\\circ$C) on axial slices selected from tumor-crossing slices of the simulation grid."
        )
        for ti, t in enumerate(trips):
            fn = t.get("filename", "")
            sl = str(t.get("slice_index", "?"))
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=0.92\linewidth]{{{fn}}}")
            lines.append(
                rf"\caption{{Anatomy $\mid$ SAR $\mid$ temperature triptych at axial slice index {sl}.}}"
            )
            lines.append(rf"\label{{fig:slice_triptych_{ti}}}")
            lines.append(r"\end{figure}")
            lines.append("")

    multiview_by_run: Dict[str, Any] = dict(extra.get("multiview_by_run") or {})
    if multiview_by_run:
        lines.append(r"\subsection{Multi-view projection analysis}")
        lines.append(r"\label{subsec:multiview_projection}")
        lines.append(
            "Per-run multiview exports used here are the unified "
            r"3$\times$3 SAR/temperature/geometry panels and the 15-timestep Ez and SAR max-projection timelines "
            "(axial, sagittal, and coronal). "
            "Single-view static SAR max-projection maps and duplicate standalone temperature maps are omitted; "
            "steady-state SAR and temperature for each run are shown in "
            r"Figs.~\ref{fig:maps_sar_gaussian}--\ref{fig:maps_temp_optimize}, with spatial context also in the 3$\times$3 panels below."
        )
        for run_key in ("gaussian", "cw", "optimize"):
            run_mv = multiview_by_run.get(run_key)
            if not isinstance(run_mv, dict):
                continue
            run_label = {
                "gaussian": "Gaussian",
                "cw": "CW",
                "optimize": "Optimized APA",
            }.get(run_key, run_key)
            if run_mv.get("unified_3x3"):
                lines.append(r"\begin{figure}[htbp]")
                lines.append(r"\centering")
                lines.append(
                    rf"\includegraphics[width=0.9\linewidth]{{multiview_{run_key}_unified_sar_temp_geometry_3x3.png}}"
                )
                lines.append(
                    rf"\caption{{{run_label}: unified multiview 3$\times$3 panel (SAR, temperature, and tumor geometry across axial/sagittal/coronal projections).}}"
                )
                lines.append(rf"\label{{fig:multiview_unified_{run_key}}}")
                lines.append(r"\end{figure}")
                lines.append("")

            timeline_files = list(run_mv.get("timeline") or [])
            if timeline_files:
                for modality in ("Ez", "SAR"):
                    subset = [
                        fn for fn in timeline_files if f"_timeline15_{modality}_" in fn
                    ]
                    subset = sorted(subset)
                    for fn in subset:
                        pv = _short_proj_from_filename(fn)
                        lines.append(r"\begin{figure}[htbp]")
                        lines.append(r"\centering")
                        lines.append(rf"\includegraphics[width=0.92\linewidth]{{{fn}}}")
                        lines.append(
                            rf"\caption{{{run_label}: 15-timestep {modality} max-projection timeline ({pv} view).}}"
                        )
                        lines.append(
                            rf"\label{{fig:multiview_timeline_{run_key}_{modality.lower()}_{pv}}}"
                        )
                        lines.append(r"\end{figure}")
                        lines.append("")

    # --- 4.6 Objective ---

    def obj_row(name: str, m: Dict[str, Any]) -> str:
        o = m.get("objective") or {}
        tw = _tex_escape(_format_duration_hms(_wall_time_s_meta(m)))
        return (
            f"{name} & \\num{{{float(o.get('J', 0)):.6f}}} & \\num{{{float(o.get('J_eff', 0)):.6f}}} & "
            f"{_num_latex(o.get('mean_sar_tumor_W_per_kg'))} & {_num_latex(o.get('mean_sar_healthy_W_per_kg'))} & "
            f"{_num_latex(o.get('p95_sar_healthy_W_per_kg'))} & \\texttt{{{tw}}} \\\\"
        )

    if has_main_runs:
        lines.append(r"\subsection{Objective Function and Targeted Heating Assessment}")
        lines.append(
            r"Table~\ref{tab:results_objective} reports $J=\overline{\mathrm{SAR}}_{\mathrm{tumor}}/\overline{\mathrm{SAR}}_{\mathrm{healthy}}$ "
            r"and the penalized objective $J_{\mathrm{eff}}$ (healthy P95 term, penalty weight $w=0.1$ as in Chapter~3)."
        )
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Objective metrics from the final SAR volume (each run). "
            r"$t_{\mathrm{wall}}$ is end-to-end pipeline wall clock time (segmentation, FDTD, SAR, thermal, "
            r"NIfTI/JSON export, validation figures, and animations when enabled).}"
        )
        lines.append(r"\label{tab:results_objective}")
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\hline")
        lines.append(
            r"Run & $J$ & $J_{\mathrm{eff}}$ & $\overline{\mathrm{SAR}}_t$\,(W/kg) & $\overline{\mathrm{SAR}}_h$\,(W/kg) & P95$_{h}$\,(W/kg) & $t_{\mathrm{wall}}$ (H:M:S) \\"
        )
        lines.append(r"\hline")
        lines.append(obj_row("Gaussian", m_g or {}))
        lines.append(obj_row("CW", m_c or {}))
        lines.append(obj_row("Optimized", m_o or {}))
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        jvals = [
            float(((m_g or {}).get("objective") or {}).get("J") or 0),
            float(((m_c or {}).get("objective") or {}).get("J") or 0),
            float(((m_o or {}).get("objective") or {}).get("J") or 0),
        ]
        if all(j < 1.0 for j in jvals):
            j_note = (
                "All three configurations yield $J<1$ on this case under the chosen source models. "
                "Interpretation should account for source placement, tumor geometry, and the simplified healthy-tissue mask (label~4)."
            )
        else:
            j_note = (
                "Objective values vary with excitation; interpret $J$ together with $J_{\\mathrm{eff}}$ and healthy P95 SAR "
                "when hotspot penalties are active."
            )
        lines.append(j_note)
        lines.append("")

    # --- Multi-case BraTS CW (optional manifest) ---
    if brats:
        lines.append(r"\subsection{Additional BraTS validation cases (CW)}")
        lines.append(r"\label{subsec:brats_cw_validation}")
        lines.append(
            "To assess generalization beyond the primary anatomy used for the Gaussian/CW/optimized comparison, "
            "the same CW sinusoidal excitation protocol (Chapter~3) was repeated on additional BraTS-style "
            "modality folders. Table~\\ref{tab:brats_cw_objective} summarizes grid shape, objective $J$, "
            "region-mean SAR, domain peak temperature ($^\\circ$C), and end-to-end pipeline wall time; "
            "per-case figures below show SAR and temperature maps."
        )
        lines.append("")

        def brats_summary_row(name: str, m: Dict[str, Any]) -> str:
            gs = m.get("grid_shape")
            grid_tex = _grid_shape_tex(gs) if gs and len(gs) >= 3 else "---"
            o = m.get("objective") or {}
            peak_t = _peak_temperature_C(m)
            tw = _tex_escape(_format_duration_hms(_wall_time_s_meta(m)))
            return (
                f"{name} & {grid_tex} & \\num{{{float(o.get('J', 0)):.6f}}} & "
                f"{_num_latex(o.get('mean_sar_tumor_W_per_kg'))} & "
                f"{_num_latex(o.get('mean_sar_healthy_W_per_kg'))} & "
                f"{_num_fixed(peak_t, 2)} & \\texttt{{{tw}}} \\\\"
            )

        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\footnotesize")
        lines.append(
            r"\caption{Multi-case CW validation: grid, objective, mean SAR in tumor vs.\ healthy tissue, "
            r"peak temperature in reported regions, and end-to-end pipeline wall clock time (H:M:S) "
            r"(segmentation through export and animations).}"
        )
        lines.append(r"\label{tab:brats_cw_objective}")
        lines.append(r"\begin{tabular}{@{}lcccccc@{}}")
        lines.append(r"\hline")
        lines.append(
            r"Case & $N_x\!\times\! N_y\!\times\! N_z$ & $J$ & "
            r"$\overline{\mathrm{SAR}}_t$\,(W/kg) & $\overline{\mathrm{SAR}}_h$\,(W/kg) & "
            r"$T_{\max}$ ($^\circ$C) & $t_{\mathrm{wall}}$ (H:M:S) \\"
        )
        lines.append(r"\hline")
        for ent in brats:
            cid = _tex_escape(str(ent.get("case_id", "?")))
            mb = meta_brats(ent)
            lines.append(brats_summary_row(cid, mb))
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
        for ent in brats:
            cid = _tex_escape(str(ent.get("case_id", "?")))
            sar = str(ent.get("sar", f"sar_brats_cw_{ent.get('case_id', '?')}.png"))
            tmp = str(
                ent.get("temperature", f"temp_brats_cw_{ent.get('case_id', '?')}.png")
            )
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=0.85\linewidth]{{{sar}}}")
            lines.append(rf"\caption{{BraTS CW case {cid}: SAR distribution (W/kg).}}")
            lines.append(rf"\label{{fig:brats_cw_sar_{cid}}}")
            lines.append(r"\end{figure}")
            lines.append("")
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=0.85\linewidth]{{{tmp}}}")
            lines.append(rf"\caption{{BraTS CW case {cid}: temperature ($^\circ$C).}}")
            lines.append(rf"\label{{fig:brats_cw_temp_{cid}}}")
            lines.append(r"\end{figure}")
            lines.append("")

    # --- 4.7 Optimization ---
    if has_main_runs:
        lines.append(r"\subsection{Antenna Parameter Studies and Optimization Results}")
        inc_tr = man.get("include_optimization_trace", True)
        inc_cmp = man.get("include_optimization_comparison", True)

        if inc_cmp and inc_tr:
            lines.append(
                "Figure~\\ref{fig:results_optimization} compares baseline versus optimized SAR for the APA configuration; "
                "Figure~\\ref{fig:results_optimization_trace} shows the optimization trace of $J$ over evaluations. "
                "Hotspot control is reflected in $J_{\\mathrm{eff}}$ via the healthy P95 SAR term."
            )
        elif inc_cmp and not inc_tr:
            lines.append(
                "Figure~\\ref{fig:results_optimization} compares baseline versus optimized SAR for the APA configuration. "
                "Hotspot control is reflected in $J_{\\mathrm{eff}}$ via the healthy P95 SAR term."
            )
        elif not inc_cmp and inc_tr:
            lines.append(
                "Figure~\\ref{fig:results_optimization_trace} shows the optimization trace of $J$ over evaluations "
                "from the antenna parameter search. "
                "Hotspot control is reflected in $J_{\\mathrm{eff}}$ via the healthy P95 SAR term."
            )
        else:
            lines.append(
                "The optimized APA configuration was executed using fixed quadrant amplitudes and phases "
                "from a prior optimization archive (replay mode without an additional search in this pipeline run). "
                "Spatial SAR and temperature for this configuration are shown in "
                "Figs.~\\ref{fig:maps_sar_optimize} and \\ref{fig:maps_temp_optimize}."
            )

        if inc_cmp:
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(
                r"\includegraphics[width=\linewidth]{optimization_comparison.png}"
            )
            lines.append(
                r"\caption{Baseline vs.\ optimized SAR (mid-$z$ slice) after quadrant phase/amplitude search.}"
            )
            lines.append(r"\label{fig:results_optimization}")
            lines.append(r"\end{figure}")
        if inc_tr:
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(
                r"\includegraphics[width=0.9\linewidth]{optimization_trace.png}"
            )
            lines.append(
                r"\caption{Objective $J$ versus evaluation index during antenna optimization.}"
            )
            lines.append(r"\label{fig:results_optimization_trace}")
            lines.append(r"\end{figure}")
        lines.append("")

    # --- 4.8 Scalability ---
    lines.append(r"\subsection{Computational Performance and Scalability}")
    lines.append(r"\label{subsec:scalability}")
    bench_path = (
        Path((runs.get("benchmark") or {}).get("results_dir"))
        / "data"
        / "scalability_benchmark_results.json"
        if has_benchmark
        else None
    )
    scal_text = ""
    b_fit: Optional[float] = None
    r2_fit: Optional[float] = None
    br: List[Dict[str, Any]] = []
    bj: Optional[Dict[str, Any]] = None
    if bench_path is not None and bench_path.is_file():
        bj = _load_json(bench_path)
        br = list(bj.get("runs") or [])
        if len(br) >= 2:
            n3 = [float(r["number_of_voxels"]) for r in br]
            rt = [float(r["total_wall_time_s"]) for r in br]
            a, b, r2 = _fit_power_law(n3, rt)
            if a is not None and b is not None and r2 is not None:
                b_fit, r2_fit = b, r2
                scal_text = (
                    f"A power-law fit to total wall time versus $N^3$ gives exponent $\\approx \\num{{{b:.2f}}}$ "
                    f"(R$^2=\\num{{{r2:.3f}}}$), consistent with increasing work per time step as the grid grows. "
                )
    scal_intro = (
        "Figure~\\ref{fig:results_scalability} summarizes runtime, time per FDTD step, and peak memory versus $N^3$ "
        f"for the benchmark configuration ({'full pipeline with resampled anatomy' if man.get('benchmark_full_pipeline') else 'FDTD-only'}). "
        + scal_text
    )
    if br:
        scal_intro = (
            "Table~\\ref{tab:scalability_benchmark} lists per-grid timings from the benchmark JSON; "
            + scal_intro
        )
    if has_benchmark:
        lines.append(scal_intro)
    else:
        lines.append(
            "Scalability benchmark outputs were not included in this bundle run (for example, when using --skip-benchmark)."
        )

    if br:
        default_steps = (bj or {}).get("benchmark_time_steps")
        br_sorted = sorted(br, key=lambda r: float(r.get("number_of_voxels", 0)))

        def _bench_row(r: Dict[str, Any]) -> str:
            gs = r.get("grid_shape")
            grid_tex = _grid_shape_tex(gs) if gs and len(gs) >= 3 else "---"
            nt = r.get("time_steps")
            if nt is None:
                nt = default_steps
            try:
                nt_s = str(int(round(float(nt)))) if nt is not None else "---"
            except (TypeError, ValueError):
                nt_s = "---"
            tw = r.get("total_wall_time_s")
            tps = r.get("time_per_step_ms")
            pm = r.get("peak_memory_MB")
            tw_s = (
                _tex_escape(_format_duration_hms(float(tw)))
                if tw is not None
                else "---"
            )
            tps_s = _num_fixed(float(tps), 4) if tps is not None else "---"
            pm_s = _num_fixed(float(pm), 1) if pm is not None else "---"
            return f"{grid_tex} & {nt_s} & \\texttt{{{tw_s}}} & {tps_s} & {pm_s} \\\\"

        cap_extra = ""
        if b_fit is not None and r2_fit is not None:
            cap_extra = (
                f" Least-squares fit in log--log space gives "
                f"$t_{{\\mathrm{{wall}}}} \\propto (N^3)^{{{b_fit:.2f}}}$ with $R^2={r2_fit:.3f}$."
            )
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\footnotesize")
        lines.append(
            rf"\caption{{Scalability benchmark runs (same $N_{{\mathrm{{steps}}}}$ per row when recorded).{cap_extra}}}"
        )
        lines.append(r"\label{tab:scalability_benchmark}")
        lines.append(r"\begin{tabular}{@{}lrrrr@{}}")
        lines.append(r"\hline")
        lines.append(
            r"$N_x \times N_y \times N_z$ & $N_{\mathrm{steps}}$ & "
            r"$t_{\mathrm{wall}}$ (H:M:S) & $\Delta t_{\mathrm{step}}$\,(ms) & Peak mem.\,(MB) \\"
        )
        lines.append(r"\hline")
        for r in br_sorted:
            lines.append(_bench_row(r))
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    if has_benchmark:
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=\linewidth]{scalability_summary.png}")
        lines.append(
            r"\caption{Scalability benchmark: runtime, time per step, and memory vs.\ $N^3$.}"
        )
        lines.append(r"\label{fig:results_scalability}")
        lines.append(r"\end{figure}")
        lines.append("")

    # --- 4.9 Validation ---
    lines.append(r"\subsection{Validation and Consistency Checks}")
    if has_main_runs:
        lines.append(
            "Analytical and regression-style checks were used during development to verify plane-wave phase behavior, "
            "lossy attenuation trends, CFL stability, grid-refinement consistency, and thermal-solver agreement on "
            "small reference problems. Qualitatively, the SAR and temperature maps in "
            "Figs.~\\ref{fig:maps_sar_gaussian}--\\ref{fig:maps_temp_optimize} exhibit smooth spatial variation without "
            "spurious boundary ringing, consistent with stable propagation and effective boundary absorption."
        )
    else:
        lines.append(
            "Analytical and regression-style checks were used during development to verify plane-wave phase behavior, "
            "lossy attenuation trends, CFL stability, grid-refinement consistency, and thermal-solver agreement on "
            "small reference problems. In this bundle variant, validation is interpreted primarily from the available "
            "executed-case artifacts."
        )
    lines.append("")

    out = paper_dir / "generated" / "paper_results_section.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument(
        "--paper-dir",
        type=Path,
        default=None,
        help="PAPER directory (default: parent of generated/ containing manifest)",
    )
    args = ap.parse_args()
    paper_dir = args.paper_dir
    if paper_dir is None:
        paper_dir = args.manifest.resolve().parent.parent
    generate_section(args.manifest, paper_dir)


if __name__ == "__main__":
    main()
