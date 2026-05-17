#!/usr/bin/env python3
"""
Run thesis paper figure bundle: Gaussian, CW, optimize, benchmark → figures + LaTeX.

**Run from the CODE directory** (same folder as this script) so relative ``--modalities-dir``
and ``--bundle-root`` paths match the examples::

  cd path/to/thesis/CODE
  python run_paper_bundle.py --modalities-dir ../dataset/validation_data/001

If modalities live under CODE (e.g. ``dataset/validation_data/001``)::

  cd path/to/thesis/CODE
  python run_paper_bundle.py --modalities-dir dataset/validation_data/001 --bundle-root results

Outputs (paths are relative to the thesis ``PAPER/`` tree):
  - PAPER/figures/results_three_runs/*.png  (stable names for \\graphicspath)
  - PAPER/generated/paper_bundle_manifest.json
  - PAPER/generated/paper_results_section.tex
  - Extra figures: pipeline_phases_waterfall.png, houle_style_plane_wave_validation.png (script),
    optional slice_triptych_rep{1,2,3}.png

Requires: PyYAML, numpy, matplotlib (for e-field export), full FDTD deps.
Intermediate runs (default ``--bundle-root``): ``paper_bundle_runs/{gaussian,cw,optimize,benchmark,brats_case001,...}/``
next to this script.

Also runs thesis BraTS CW validation configs (thesis_paper_brats_case*_cw.yaml) when their
modalities folders exist under CODE/ (see each YAML modalities_dir).

Partial runs (preserve existing bundle outputs; run from ``CODE/``). Re-runs only
``benchmark/`` under ``--bundle-root``; leaves ``gaussian/``, ``cw/``, ``optimize/``, ``brats_*`` intact::

  python run_paper_bundle.py --modalities-dir ../dataset/validation_data/001 \\
    --skip-main-runs --skip-brats-cases

Options:
  --skip-sim         No simulations: regenerate run ``images/*.png`` from NIfTI (then copy to PAPER),
                     run extra figures + TeX (manifest must exist). Use ``--sync-only`` to skip regeneration.
                     --bundle-root is ignored.
  --sync-only        With ``--skip-sim``: only copy existing PNGs from each ``results_dir`` (fast; stale if code changed).
  --optimize-load-from PATH  Optimize step only: ``--load-optimized-from`` for the engine (replay prior optimization).
  --skip-main-runs   Skip Gaussian/CW/optimized simulations. Existing ``bundle_root/{gaussian,cw,optimize}`` data
                     are not deleted; figures and manifest are refreshed from disk when those dirs contain outputs.
  --skip-benchmark   Skip benchmark simulation. Existing ``bundle_root/benchmark`` is not deleted; manifest keeps
                     or repairs ``results_dir`` when benchmark outputs already exist.
  --skip-brats-cases Skip re-running thesis_paper_brats_case*_cw.yaml; existing ``bundle_root/brats_*`` dirs are
                     not removed, and prior manifest BraTS entries are kept while figures are re-synced from disk.
  --paper-dir PATH   Default: thesis ``PAPER/`` (parent of CODE); relative paths resolve from cwd
  --bundle-root PATH  Default: ``paper_bundle_runs`` next to this script; relative paths resolve from cwd (use CODE/ as cwd)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parent
THESIS_ROOT = CODE_DIR.parent
DEFAULT_PAPER_DIR = THESIS_ROOT / "PAPER"
ENGINE = CODE_DIR / "fdtd_brain_simulation_engine.py"
GEN_TEX = CODE_DIR / "generate_paper_results_section.py"
PLOT_SCALING = CODE_DIR / "plot_scalability_benchmark.py"
PLOT_WATERFALL = CODE_DIR / "plot_pipeline_phases_waterfall.py"
PLOT_HOULE_VALIDATION = CODE_DIR / "plot_houle_style_plane_wave_validation.py"

# BraTS multi-case CW validation (modalities_dir read from each YAML, resolved under CODE_DIR).
BRATS_CW_CASES: list[tuple[str, str]] = [
    ("case001", "configs/thesis_paper_brats_case001_cw.yaml"),
    ("case002", "configs/thesis_paper_brats_case002_cw.yaml"),
    ("case003", "configs/thesis_paper_brats_case003_cw.yaml"),
]


def _modalities_dir_from_config(config_rel: str) -> Path:
    cfg_path = CODE_DIR / config_rel
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    md = data.get("modalities_dir")
    if not md:
        raise ValueError(f"{config_rel}: missing modalities_dir")
    p = Path(md)
    if not p.is_absolute():
        p = (CODE_DIR / p).resolve()
    return p.resolve()


def _find_output_base(data_dir: Path) -> str:
    metas = sorted(data_dir.glob("*_metadata.json"))
    if not metas:
        raise FileNotFoundError(f"No *_metadata.json under {data_dir}")
    stem = metas[0].stem
    if not stem.endswith("_metadata"):
        raise ValueError(f"Unexpected metadata name: {stem}")
    return stem[: -len("_metadata")]


def _copy_suffix(images_dir: Path, base: str, suffix: str, dest: Path) -> None:
    src = images_dir / f"{base}{suffix}"
    if not src.is_file():
        raise FileNotFoundError(f"Missing figure: {src}")
    shutil.copy2(src, dest)


def _copy_suffix_warn(images_dir: Path, base: str, suffix: str, dest: Path) -> bool:
    """Like _copy_suffix but print a warning and return False if the source is missing."""
    src = images_dir / f"{base}{suffix}"
    if not src.is_file():
        print(f"[WARN] Missing figure (sync skipped): {src}")
        return False
    shutil.copy2(src, dest)
    return True


def _sync_stable_paper_figures_from_manifest(man: dict, fig_dir: Path) -> None:
    """
    Copy stable paper filenames (geometry, SAR/temp maps, optimization, BraTS, benchmark)
    from each run's ``images/`` and ``data/`` into ``fig_dir``, using manifest paths.
    Used for ``--skip-sim`` so LaTeX includes match pipeline outputs under ``results_dir``.
    """
    runs = man.get("runs") or {}

    def _run_images_base(key: str) -> tuple[Path | None, str | None]:
        info = runs.get(key) or {}
        rdir_s = info.get("results_dir")
        if not rdir_s:
            return None, None
        rdir = Path(rdir_s)
        ob = info.get("output_base")
        if not ob and (rdir / "data").is_dir():
            try:
                ob = _find_output_base(rdir / "data")
            except (OSError, ValueError, FileNotFoundError):
                ob = None
        return rdir, ob

    gdir, base_g = _run_images_base("gaussian")
    cdir, base_c = _run_images_base("cw")
    odir, base_o = _run_images_base("optimize")

    if gdir and base_g:
        img = gdir / "images"
        _copy_suffix_warn(
            img, base_g, "_fdtd_geometry_slice.png", fig_dir / "geometry_same_case.png"
        )
        _copy_suffix_warn(
            img, base_g, "_SAR_distribution.png", fig_dir / "sar_gaussian.png"
        )
        _copy_suffix_warn(
            img, base_g, "_temperature_distribution.png", fig_dir / "temp_gaussian.png"
        )
    else:
        print(
            "[WARN] sync: gaussian results_dir or output_base missing — skip geometry + Gaussian maps"
        )

    if cdir and base_c:
        img = cdir / "images"
        _copy_suffix_warn(img, base_c, "_SAR_distribution.png", fig_dir / "sar_cw.png")
        _copy_suffix_warn(
            img, base_c, "_temperature_distribution.png", fig_dir / "temp_cw.png"
        )
    else:
        print("[WARN] sync: cw results_dir or output_base missing — skip CW maps")

    if odir and base_o:
        img = odir / "images"
        _copy_suffix_warn(
            img, base_o, "_SAR_distribution.png", fig_dir / "sar_optimized.png"
        )
        _copy_suffix_warn(
            img, base_o, "_temperature_distribution.png", fig_dir / "temp_optimized.png"
        )
        man["include_optimization_comparison"] = _copy_suffix_warn(
            img,
            base_o,
            "_antenna_opt_comparison.png",
            fig_dir / "optimization_comparison.png",
        )
        if not man["include_optimization_comparison"]:
            print(
                "[WARN] sync: baseline vs. optimized SAR comparison missing "
                f"({base_o}_antenna_opt_comparison.png)"
            )
        trace_src = img / f"{base_o}_optimization_trace.png"
        if trace_src.is_file():
            shutil.copy2(trace_src, fig_dir / "optimization_trace.png")
            man["include_optimization_trace"] = True
        else:
            print(f"[WARN] optimization trace missing: {trace_src}")
            man["include_optimization_trace"] = False
    else:
        print(
            "[WARN] sync: optimize results_dir or output_base missing — skip optimized maps"
        )

    for ent in man.get("brats_cw_validation") or []:
        if not isinstance(ent, dict):
            continue
        rdir_s = ent.get("results_dir")
        case_id = ent.get("case_id") or ""
        if not rdir_s:
            continue
        rdir = Path(rdir_s)
        base_b = ent.get("output_base")
        if not base_b and (rdir / "data").is_dir():
            try:
                base_b = _find_output_base(rdir / "data")
            except (OSError, ValueError, FileNotFoundError):
                base_b = None
        if not base_b:
            print(f"[WARN] sync: BraTS {case_id}: no output_base — skip")
            continue
        img = rdir / "images"
        _copy_suffix_warn(
            img,
            base_b,
            "_SAR_distribution.png",
            fig_dir / f"sar_brats_cw_{case_id}.png",
        )
        _copy_suffix_warn(
            img,
            base_b,
            "_temperature_distribution.png",
            fig_dir / f"temp_brats_cw_{case_id}.png",
        )

    bench = runs.get("benchmark") or {}
    br_s = bench.get("results_dir") if isinstance(bench, dict) else None
    if br_s:
        sum_src = Path(br_s) / "data" / "scalability_summary.png"
        if sum_src.is_file():
            shutil.copy2(sum_src, fig_dir / "scalability_summary.png")
        else:
            print(f"[WARN] scalability_summary.png not found: {sum_src}")

    print(
        "[INFO] Synced stable paper figures from manifest results_dir →",
        fig_dir,
        flush=True,
    )


def _refresh_paper_run_figures_from_disk(man: dict) -> None:
    """
    Rebuild ``{results_dir}/images`` distribution + multiview PNGs from NIfTI (+ frame NPZs when present)
    for Gaussian, CW, optimize, and each BraTS manifest entry.
    """
    from data_analysis_validation import regenerate_run_figures_from_disk

    for key in ("gaussian", "cw", "optimize"):
        info = man.get("runs", {}).get(key) or {}
        rdir = info.get("results_dir")
        ob = info.get("output_base")
        if not rdir or not ob:
            print(
                f"[WARN] refresh: manifest runs.{key} missing results_dir or output_base — skip",
                flush=True,
            )
            continue
        regenerate_run_figures_from_disk(rdir, ob)
    for ent in man.get("brats_cw_validation") or []:
        if not isinstance(ent, dict):
            continue
        rdir = ent.get("results_dir")
        ob = ent.get("output_base")
        cid = ent.get("case_id", "")
        if not rdir or not ob:
            print(
                f"[WARN] refresh: BraTS case {cid!r} missing results_dir or output_base — skip",
                flush=True,
            )
            continue
        regenerate_run_figures_from_disk(rdir, ob)


def _sanitize_manifest_paper_fields(man: dict) -> None:
    """Drop deprecated manifest keys (mid-$z$ E-field PNGs) from older bundle JSON."""
    man.pop("efield_caption", None)
    man.pop("efield_captions", None)
    for ent in man.get("run_figure_sets") or []:
        if isinstance(ent, dict):
            ent.pop("efield", None)
    for ent in man.get("brats_cw_validation") or []:
        if isinstance(ent, dict):
            ent.pop("efield", None)
            ent.pop("efield_caption", None)


def _resolve_existing_results_dir(name: str, configured: str | None) -> str | None:
    """Best-effort repair for stale manifest run directories during --skip-sim."""
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            CODE_DIR / "results" / name,
            CODE_DIR / "paper_bundle_runs" / name,
        ]
    )
    for cand in candidates:
        data_dir = cand / "data"
        if cand.is_dir() and data_dir.is_dir():
            return str(cand.resolve())
    return configured


def _bundle_run_dir_has_outputs(run_dir: Path) -> bool:
    """True if ``run_dir/data`` exists and contains at least one ``*_metadata.json``."""
    data_dir = run_dir / "data"
    if not run_dir.is_dir() or not data_dir.is_dir():
        return False
    return any(data_dir.glob("*_metadata.json"))


def _prepare_bundle_run_dir(run_dir: Path, wipe: bool) -> None:
    """
    If ``wipe`` is True, remove ``run_dir`` and recreate it empty (fresh simulation).
    If ``wipe`` is False, only ensure the directory exists — existing outputs are preserved.
    """
    if wipe:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)


def _main_run_manifest_fields(
    run_dir: Path, ran_this_session: bool
) -> tuple[str | None, str | None]:
    """
    Return ``(results_dir, output_base)`` for manifest / figure sync.
    When ``ran_this_session`` is False, still report paths if prior outputs exist under ``run_dir``.
    """
    if ran_this_session:
        return str(run_dir.resolve()), _find_output_base(run_dir / "data")
    if _bundle_run_dir_has_outputs(run_dir):
        return str(run_dir.resolve()), _find_output_base(run_dir / "data")
    return None, None


def _copy_brats_manifest_figures_to_paper(brats_entries: list, fig_dir: Path) -> None:
    """Copy SAR/temperature bundle PNGs for each BraTS manifest entry (used when runs are skipped)."""
    for ent in brats_entries:
        if not isinstance(ent, dict):
            continue
        case_id = ent.get("case_id")
        rdir_s = ent.get("results_dir")
        if not case_id or not rdir_s:
            continue
        rdir = Path(rdir_s)
        base_b = ent.get("output_base")
        if not base_b and (rdir / "data").is_dir():
            try:
                base_b = _find_output_base(rdir / "data")
            except (OSError, ValueError, FileNotFoundError):
                base_b = None
        if not base_b:
            print(
                f"[WARN] BraTS {case_id}: could not resolve output_base — skip figure sync"
            )
            continue
        _copy_suffix_warn(
            rdir / "images",
            base_b,
            "_SAR_distribution.png",
            fig_dir / f"sar_brats_cw_{case_id}.png",
        )
        _copy_suffix_warn(
            rdir / "images",
            base_b,
            "_temperature_distribution.png",
            fig_dir / f"temp_brats_cw_{case_id}.png",
        )


def _sync_benchmark_scaling_artifacts(bdir: Path, fig_dir: Path) -> bool:
    """
    Copy ``scalability_summary.png`` into ``fig_dir``, or build it from JSON when missing.
    Returns the ``benchmark_full_pipeline`` flag from JSON when available.
    """
    bench_json = bdir / "data" / "scalability_benchmark_results.json"
    sum_src = bdir / "data" / "scalability_summary.png"
    if sum_src.is_file():
        shutil.copy2(sum_src, fig_dir / "scalability_summary.png")
    elif bench_json.is_file():
        subprocess.run(
            [
                sys.executable,
                str(PLOT_SCALING),
                str(bench_json),
                "--out-dir",
                str(fig_dir),
            ],
            cwd=str(CODE_DIR),
            check=False,
        )
    else:
        print(f"[WARN] Missing benchmark outputs under {bdir / 'data'}")
    bf_full = False
    try:
        with open(bench_json, encoding="utf-8") as f:
            bf_full = bool(json.load(f).get("benchmark_full_pipeline"))
    except OSError:
        pass
    return bf_full


def _slice_index_from_triptych_filename(name: str) -> int:
    m = re.search(r"_slice_(\d+)_anatomy_SAR_temperature", name)
    return int(m.group(1)) if m else -1


def _pick_triptych_paths(paths: list[Path], n: int = 3) -> list[Path]:
    """Pick up to ``n`` representative slice PNGs (low / mid / high in sorted order)."""
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    ix = [0, len(paths) // 2, len(paths) - 1]
    out: list[Path] = []
    for i in dict.fromkeys(ix):
        out.append(paths[i])
    return out[:n]


def finalize_paper_extra_figures(
    fig_dir: Path,
    run_results_dirs: dict[str, Path | None],
    *,
    legacy_output_base: str = "",
) -> dict:
    """
    Generate extra figures and collect multiview exports for each run.
    CW-specific artifacts (waterfall, slice triptychs) are still sourced from CW.
    Per-run multiview includes Ez/SAR 15-timeline PNGs and SAR/temperature max-projection views.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    extra: dict = {
        "pipeline_phases_waterfall": False,
        "houle_plane_wave_validation": False,
        "slice_triptychs": [],
        "multiview_unified_3x3": False,
        "multiview_timeline": [],
        "multiview_by_run": {},
    }

    houle_out = fig_dir / "houle_style_plane_wave_validation.png"
    r_houle = subprocess.run(
        [
            sys.executable,
            str(PLOT_HOULE_VALIDATION),
            "--out",
            str(houle_out),
        ],
        cwd=str(CODE_DIR),
        check=False,
    )
    if r_houle.returncode != 0:
        print(
            f"[WARN] plot_houle_style_plane_wave_validation.py failed (exit {r_houle.returncode})"
        )
    extra["houle_plane_wave_validation"] = houle_out.is_file()

    cw_results_dir = run_results_dirs.get("cw")
    if cw_results_dir is None or not cw_results_dir.is_dir():
        print("[WARN] CW run missing; skipping CW-only extra figures.")
    else:
        data_dir = cw_results_dir / "data"
        images_dir = cw_results_dir / "images"
        try:
            cw_base = _find_output_base(data_dir)
        except (OSError, ValueError, FileNotFoundError):
            cw_base = legacy_output_base
        if not cw_base:
            print(
                "[WARN] CW run: could not resolve output_base; skipping CW-only extras."
            )
        else:
            perf_json = data_dir / f"{cw_base}_performance.json"
            if perf_json.is_file():
                wf_out = fig_dir / "pipeline_phases_waterfall.png"
                r_wf = subprocess.run(
                    [
                        sys.executable,
                        str(PLOT_WATERFALL),
                        "--performance-json",
                        str(perf_json),
                        "--out",
                        str(wf_out),
                    ],
                    cwd=str(CODE_DIR),
                    check=False,
                )
                if r_wf.returncode != 0:
                    print(
                        f"[WARN] plot_pipeline_phases_waterfall.py failed (exit {r_wf.returncode})"
                    )
                extra["pipeline_phases_waterfall"] = wf_out.is_file()
            elif images_dir.is_dir():
                print(f"[WARN] No performance JSON for waterfall: {perf_json}")

            tri_paths = sorted(
                images_dir.glob(f"{cw_base}_slice_*_anatomy_SAR_temperature.png"),
                key=lambda p: _slice_index_from_triptych_filename(p.name),
            )
            picked = _pick_triptych_paths(tri_paths, 3)
            rep_names = [
                "slice_triptych_rep1.png",
                "slice_triptych_rep2.png",
                "slice_triptych_rep3.png",
            ]
            for i, src in enumerate(picked):
                dest_name = rep_names[i]
                shutil.copy2(src, fig_dir / dest_name)
                extra["slice_triptychs"].append(
                    {
                        "slice_index": _slice_index_from_triptych_filename(src.name),
                        "filename": dest_name,
                    }
                )
            if not picked:
                print(
                    f"[WARN] No slice triptychs matched {cw_base}_slice_*_anatomy_SAR_temperature.png under {images_dir}"
                )

    # Multiview static exports (from multiview_visualization.py) for each run.
    timeline_suffixes = [
        ("Ez", "axial_maxz"),
        ("Ez", "sagittal_maxx"),
        ("Ez", "coronal_maxy"),
        ("SAR", "axial_maxz"),
        ("SAR", "sagittal_maxx"),
        ("SAR", "coronal_maxy"),
    ]
    for run_key, run_dir in run_results_dirs.items():
        if run_dir is None or not run_dir.is_dir():
            continue
        run_images = run_dir / "images"
        if not run_images.is_dir():
            continue
        try:
            run_base = _find_output_base(run_dir / "data")
        except (OSError, ValueError, FileNotFoundError):
            run_base = legacy_output_base
        if not run_base:
            print(
                f"[WARN] {run_key}: could not resolve output_base; skipping multiview exports."
            )
            continue
        by_run = {
            "unified_3x3": False,
            "timeline": [],
            "sar_maxproj": [],
            "temp_maxproj": [],
        }
        extra["multiview_by_run"][run_key] = by_run

        uni_src = run_images / f"{run_base}_unified_sar_temp_geometry_3x3.png"
        if uni_src.is_file():
            dest_name = f"multiview_{run_key}_unified_sar_temp_geometry_3x3.png"
            shutil.copy2(uni_src, fig_dir / dest_name)
            by_run["unified_3x3"] = True
            if run_key == "cw":
                shutil.copy2(
                    uni_src, fig_dir / "multiview_unified_sar_temp_geometry_3x3.png"
                )
                extra["multiview_unified_3x3"] = True

        for tag, proj in timeline_suffixes:
            src = run_images / f"{run_base}_timeline15_{tag}_{proj}.png"
            if not src.is_file():
                continue
            dest_name = f"multiview_{run_key}_timeline15_{tag}_{proj}.png"
            shutil.copy2(src, fig_dir / dest_name)
            by_run["timeline"].append(dest_name)
            if run_key == "cw":
                legacy_name = f"multiview_timeline15_{tag}_{proj}.png"
                shutil.copy2(src, fig_dir / legacy_name)
                extra["multiview_timeline"].append(legacy_name)

        # Standalone per-axis SAR/temperature max-projection PNGs are not synced to
        # PAPER: the thesis results section omits those single-view figures (unified
        # 3x3 panels and timelines suffice; SAR/temp maps remain in Sec.~4.3).

    return extra


def _simulate_command() -> list[str]:
    """Prefer installed ``hermes-simulate``; fall back to module invocation."""
    import shutil

    if shutil.which("hermes-simulate"):
        return ["hermes-simulate"]
    return [sys.executable, "-m", "hermes_drh.cli.main"]


def run_engine(
    config_rel: str,
    results_dir: Path,
    modalities_dir: str,
    extra_argv: list | None = None,
) -> None:
    cfg = CODE_DIR / config_rel
    if not cfg.is_file():
        raise FileNotFoundError(cfg)
    cmd = [
        *_simulate_command(),
        "--config",
        str(cfg),
        "--modalities-dir",
        modalities_dir,
        "--results-dir",
        str(results_dir),
    ]
    if extra_argv:
        cmd.extend(extra_argv)
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(CODE_DIR), check=True, env=os.environ.copy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modalities-dir",
        default=None,
        help=(
            "BraTS modalities folder (four NIfTIs). Required unless --skip-sim with existing manifest. "
            "Relative paths resolve from the current working directory; run from CODE/ for thesis-relative paths."
        ),
    )
    ap.add_argument(
        "--paper-dir",
        type=Path,
        default=DEFAULT_PAPER_DIR,
        help=(
            "PAPER directory (default: thesis PAPER/ next to CODE). "
            "Relative paths resolve from cwd; run from CODE/ so ../PAPER works."
        ),
    )
    ap.add_argument(
        "--bundle-root",
        type=Path,
        default=CODE_DIR / "paper_bundle_runs",
        help=(
            "Directory for gaussian/, cw/, optimize/, benchmark/, brats_*/ outputs. "
            "Default is paper_bundle_runs next to this script. If you pass a relative path, "
            "resolve it by running from CODE/ (e.g. --bundle-root results → CODE/results/)."
        ),
    )
    ap.add_argument(
        "--skip-sim",
        action="store_true",
        help="Skip simulations; refresh PNGs from NIfTI then sync to PAPER, extras, TeX (--bundle-root ignored)",
    )
    ap.add_argument(
        "--sync-only",
        action="store_true",
        help="With --skip-sim: copy existing results PNGs only (no NIfTI/matplotlib regeneration)",
    )
    ap.add_argument(
        "--optimize-load-from",
        metavar="PATH",
        default=None,
        help=(
            "Optimize bundle step only: pass --load-optimized-from PATH to fdtd_brain_simulation_engine "
            "(replay f0/quadrant/geometry from *_antenna_optimization.json or *_metadata.json)."
        ),
    )
    ap.add_argument(
        "--skip-main-runs",
        action="store_true",
        help=(
            "Skip Gaussian/CW/optimized simulations; do not delete existing "
            "gaussian/, cw/, optimize/ under --bundle-root. Refresh paper PNGs from disk when outputs exist."
        ),
    )
    ap.add_argument(
        "--skip-benchmark",
        action="store_true",
        help=(
            "Skip benchmark simulation; do not delete existing benchmark/ under --bundle-root. "
            "Refresh scalability figures from disk when benchmark JSON exists."
        ),
    )
    ap.add_argument(
        "--skip-brats-cases",
        action="store_true",
        help=(
            "Skip BraTS CW re-runs; do not delete brats_case* dirs. Keep prior manifest BraTS entries "
            "and re-sync sar/temp PNGs from those results_dir paths."
        ),
    )
    args = ap.parse_args()

    paper_dir: Path = args.paper_dir.resolve()
    fig_dir = paper_dir / "figures" / "results_three_runs"
    gen_dir = paper_dir / "generated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = gen_dir / "paper_bundle_manifest.json"

    if not args.skip_sim:
        if not args.modalities_dir:
            ap.error("--modalities-dir is required unless --skip-sim")
        modalities_abs = str(Path(args.modalities_dir).resolve())
        root = args.bundle_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        gdir = root / "gaussian"
        cdir = root / "cw"
        odir = root / "optimize"
        bdir = root / "benchmark"

        prior_manifest: dict = {}
        if manifest_path.is_file():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    prior_manifest = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError):
                prior_manifest = {}

        run_main_runs = not args.skip_main_runs
        run_benchmark = not args.skip_benchmark

        for run_dir, wipe in (
            (gdir, run_main_runs),
            (cdir, run_main_runs),
            (odir, run_main_runs),
            (bdir, run_benchmark),
        ):
            _prepare_bundle_run_dir(run_dir, wipe)

        if run_main_runs:
            run_engine("configs/paper_results_gaussian.yaml", gdir, modalities_abs)
            run_engine("configs/paper_results_cw.yaml", cdir, modalities_abs)
            opt_extra = None
            if args.optimize_load_from:
                p = Path(args.optimize_load_from)
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                opt_extra = ["--load-optimized-from", str(p)]
            run_engine(
                "configs/paper_results_optimize.yaml",
                odir,
                modalities_abs,
                extra_argv=opt_extra,
            )
        else:
            print(
                "[INFO] Skipping Gaussian/CW/optimized runs (--skip-main-runs); "
                "existing bundle outputs under gaussian/, cw/, optimize/ are preserved."
            )

        bf_full = False
        bench_results_dir: str | None = None
        if run_benchmark:
            run_engine(
                "configs/paper_results_benchmark.yaml",
                bdir,
                modalities_abs,
            )
            bench_results_dir = str(bdir.resolve())
            bf_full = _sync_benchmark_scaling_artifacts(bdir, fig_dir)
        else:
            print(
                "[INFO] Skipping benchmark (--skip-benchmark); "
                "existing bundle outputs under benchmark/ are preserved."
            )
            if _bundle_run_dir_has_outputs(bdir):
                bench_results_dir = str(bdir.resolve())
                bf_full = _sync_benchmark_scaling_artifacts(bdir, fig_dir)

        rg, base_g = _main_run_manifest_fields(gdir, run_main_runs)
        rc, base_c = _main_run_manifest_fields(cdir, run_main_runs)
        ro, base_o = _main_run_manifest_fields(odir, run_main_runs)

        output_base: str | None = None
        include_trace = False
        include_optimization_comparison = True
        run_figure_sets: list[dict] = []

        if base_g and base_c and base_o:
            if not (base_g == base_c == base_o):
                print(
                    "[INFO] Per-run output_base (file stems): "
                    f"gaussian={base_g!r}, cw={base_c!r}, optimize={base_o!r}"
                )
            output_base = base_g if base_g == base_c == base_o else base_g

            copy_fn = _copy_suffix if run_main_runs else _copy_suffix_warn
            copy_fn(
                gdir / "images",
                base_g,
                "_fdtd_geometry_slice.png",
                fig_dir / "geometry_same_case.png",
            )
            copy_fn(
                gdir / "images",
                base_g,
                "_SAR_distribution.png",
                fig_dir / "sar_gaussian.png",
            )
            copy_fn(
                gdir / "images",
                base_g,
                "_temperature_distribution.png",
                fig_dir / "temp_gaussian.png",
            )
            copy_fn(
                cdir / "images",
                base_c,
                "_SAR_distribution.png",
                fig_dir / "sar_cw.png",
            )
            copy_fn(
                cdir / "images",
                base_c,
                "_temperature_distribution.png",
                fig_dir / "temp_cw.png",
            )
            copy_fn(
                odir / "images",
                base_o,
                "_SAR_distribution.png",
                fig_dir / "sar_optimized.png",
            )
            copy_fn(
                odir / "images",
                base_o,
                "_temperature_distribution.png",
                fig_dir / "temp_optimized.png",
            )
            include_optimization_comparison = _copy_suffix_warn(
                odir / "images",
                base_o,
                "_antenna_opt_comparison.png",
                fig_dir / "optimization_comparison.png",
            )
            if run_main_runs and not include_optimization_comparison:
                print(
                    "[WARN] optimization_comparison.png missing (expected when "
                    "--load-optimized-from skips antenna optimization). "
                    "LaTeX will omit the baseline vs. optimized SAR comparison figure."
                )

            for run_key, run_label, run_dir, run_slug, run_base in (
                ("gaussian", "Gaussian", gdir, "gaussian", base_g),
                ("cw", "CW", cdir, "cw", base_c),
                ("optimize", "Optimized APA", odir, "optimized", base_o),
            ):
                sar_name = f"sar_{run_slug}.png"
                temp_name = f"temp_{run_slug}.png"
                copy_fn(
                    run_dir / "images",
                    run_base,
                    "_SAR_distribution.png",
                    fig_dir / sar_name,
                )
                copy_fn(
                    run_dir / "images",
                    run_base,
                    "_temperature_distribution.png",
                    fig_dir / temp_name,
                )
                run_figure_sets.append(
                    {
                        "run_key": run_key,
                        "run_label": run_label,
                        "sar": sar_name,
                        "temperature": temp_name,
                    }
                )
            trace_src = odir / "images" / f"{base_o}_optimization_trace.png"
            include_trace = trace_src.is_file()
            if include_trace:
                shutil.copy2(trace_src, fig_dir / "optimization_trace.png")
            elif run_main_runs:
                print(
                    "[WARN] optimization_trace.png missing — LaTeX will omit trace figure"
                )
        else:
            print(
                "[WARN] Incomplete main-run outputs under bundle_root "
                f"({root}): cannot sync Gaussian/CW/optimized paper figures "
                "(need gaussian/, cw/, optimize/ each with data/*_metadata.json)."
            )

        brats_manifest: list[dict] = []
        if not args.skip_brats_cases:
            for case_id, cfg_rel in BRATS_CW_CASES:
                try:
                    mod_brats = _modalities_dir_from_config(cfg_rel)
                except (OSError, ValueError, yaml.YAMLError) as e:
                    print(
                        f"[WARN] BraTS CW {case_id}: could not read config {cfg_rel}: {e}"
                    )
                    continue
                if not mod_brats.is_dir():
                    print(
                        f"[WARN] Skipping BraTS CW {case_id}: modalities directory not found: {mod_brats}"
                    )
                    continue
                bcase = root / f"brats_{case_id}"
                if bcase.exists():
                    shutil.rmtree(bcase)
                bcase.mkdir(parents=True)
                run_engine(cfg_rel, bcase, str(mod_brats))
                base_b = _find_output_base(bcase / "data")
                _copy_suffix(
                    bcase / "images",
                    base_b,
                    "_SAR_distribution.png",
                    fig_dir / f"sar_brats_cw_{case_id}.png",
                )
                _copy_suffix(
                    bcase / "images",
                    base_b,
                    "_temperature_distribution.png",
                    fig_dir / f"temp_brats_cw_{case_id}.png",
                )
                brats_manifest.append(
                    {
                        "case_id": case_id,
                        "modalities_dir": str(mod_brats),
                        "results_dir": str(bcase.resolve()),
                        "output_base": base_b,
                        "config": cfg_rel,
                        "sar": f"sar_brats_cw_{case_id}.png",
                        "temperature": f"temp_brats_cw_{case_id}.png",
                    }
                )
        else:
            print(
                "[INFO] Skipping BraTS CW validation (--skip-brats-cases); "
                "preserving prior manifest entries and re-syncing BraTS figures from disk."
            )
            brats_manifest = list(prior_manifest.get("brats_cw_validation") or [])
            _copy_brats_manifest_figures_to_paper(brats_manifest, fig_dir)

        manifest = {
            "modalities_dir": modalities_abs,
            "output_base": output_base,
            "runs": {
                "gaussian": {
                    "results_dir": rg,
                    **({"output_base": base_g} if base_g else {}),
                },
                "cw": {
                    "results_dir": rc,
                    **({"output_base": base_c} if base_c else {}),
                },
                "optimize": {
                    "results_dir": ro,
                    **({"output_base": base_o} if base_o else {}),
                },
                "benchmark": {"results_dir": bench_results_dir},
            },
            "main_runs_skipped": bool(args.skip_main_runs),
            "benchmark_skipped": bool(args.skip_benchmark),
            "include_optimization_trace": include_trace,
            "include_optimization_comparison": include_optimization_comparison,
            "benchmark_full_pipeline": bf_full,
            "run_figure_sets": run_figure_sets,
            "brats_cw_validation": brats_manifest,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {manifest_path}")

    # Regenerate LaTeX (also when skip-sim)
    if not manifest_path.is_file():
        print(
            f"Missing {manifest_path}; run without --skip-sim first.", file=sys.stderr
        )
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    _sanitize_manifest_paper_fields(man)
    for run_name in ("gaussian", "cw", "optimize", "benchmark"):
        run_info = man.get("runs", {}).get(run_name)
        if not isinstance(run_info, dict):
            continue
        repaired = _resolve_existing_results_dir(run_name, run_info.get("results_dir"))
        if repaired != run_info.get("results_dir"):
            print(
                f"[INFO] Repaired manifest path for {run_name}: "
                f"{run_info.get('results_dir')} -> {repaired}"
            )
            run_info["results_dir"] = repaired
        if (
            run_name in ("gaussian", "cw", "optimize")
            and run_info.get("results_dir")
            and not run_info.get("output_base")
        ):
            rdir = Path(run_info["results_dir"])
            data_d = rdir / "data"
            if data_d.is_dir():
                try:
                    run_info["output_base"] = _find_output_base(data_d)
                except (OSError, ValueError, FileNotFoundError):
                    pass
    if args.sync_only and not args.skip_sim:
        print(
            "[WARN] --sync-only applies only with --skip-sim; ignoring.",
            flush=True,
        )
    if args.skip_sim and not args.sync_only:
        _refresh_paper_run_figures_from_disk(man)
    if args.skip_sim:
        _sync_stable_paper_figures_from_manifest(man, fig_dir)
    run_results_dirs = {}
    for run_name in ("gaussian", "cw", "optimize"):
        run_dir_s = man.get("runs", {}).get(run_name, {}).get("results_dir")
        run_results_dirs[run_name] = Path(run_dir_s) if run_dir_s else None
    man["paper_extra_figures"] = finalize_paper_extra_figures(
        fig_dir,
        run_results_dirs,
        legacy_output_base=str(man.get("output_base") or ""),
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)

    subprocess.run(
        [
            sys.executable,
            str(GEN_TEX),
            str(manifest_path),
            "--paper-dir",
            str(paper_dir),
        ],
        cwd=str(CODE_DIR),
        check=True,
    )
    print(
        "Done. Compile official_paper.tex from PAPER/ (see \\input{generated/paper_results_section})."
    )


if __name__ == "__main__":
    main()
