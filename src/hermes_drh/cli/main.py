"""
3D FDTD brain segmentation simulation engine.

Orchestrates the full simulation pipeline as described in the thesis Procedures
(Sec. 3.2): Voxel-based Tissue Modeling (3.2.1), FDTD Solver (3.2.2), Source
Injection and Antenna Modeling (3.2.3), SAR Computation (3.2.4), Thermal Solver
(3.2.5), Antenna Optimization (3.2.6), Performance Evaluation (3.2.6), and Data
Analysis and Validation (3.2.8). Calculation order: grid + materials → FDTD
(D→source→E,I→H) → SAR (σ|E_rms|²/(2ρ)) → thermal (∇·(k∇T)+Q=0) → save/visualize. Implementation details live in the CODE/ modules
(fdtd_solver, sar_computation, thermal_solver, voxel_model, sources,
antenna_optimization, performance_logging, data_analysis_validation). This script
remains the single entry point: same CLI, defaults, and output layout. All
simulation outputs are written under the `results/` folder.

Usage:
  # From existing segmentation NIfTI:
  python fdtd_brain_simulation_engine.py --seg brain_segmentation.nii
  python fdtd_brain_simulation_engine.py brain_segmentation.nii   # positional still supported

  # From 4 BraTS modality NIfTIs (runs 3D U-Net segmentation first):
  python fdtd_brain_simulation_engine.py --modalities flair.nii t1.nii t1ce.nii t2.nii [--checkpoint path.pth]
  # From a single folder containing the four modalities (auto-detects filenames):
  python fdtd_brain_simulation_engine.py --modalities-dir dataset/validation_data/001 [--checkpoint path.pth]

  # Defaults from YAML (CLI overrides); keys = argparse dest names (snake_case):
  python fdtd_brain_simulation_engine.py --config configs/simulation_example.yaml --modalities-dir dataset/validation_data/001

  # Fixed output directory (reproducible paper bundle; see run_paper_bundle.py):
  python fdtd_brain_simulation_engine.py --config configs/paper_results_gaussian.yaml \\
    --modalities-dir dataset/validation_data/001 --results-dir paper_bundle_runs/gaussian

  # Antenna optimization (4-quadrant APA, Houle Ch.6 style; maximizes SAR tumor/healthy ratio):
  # With --optimize-antenna, the main FDTD run uses the optimized 4-quadrant source (Option A);
  # all SAR, temperature, E-frames, and animations are from this optimized configuration.
  python fdtd_brain_simulation_engine.py --seg brain_segmentation.nii --optimize-antenna [--f0 100e6 ...]

  # Fixed 4-quadrant APA (no optimization; faster for tests): mutually exclusive with --optimize-antenna.
  # Defaults: amplitude 1.0 and phase 0° per quadrant; --f0 carrier (default 100 MHz); ring offset npml+2;
  # z_plane at tumor centroid; same ramped CW driver as the post-optimization FDTD run.
  python fdtd_brain_simulation_engine.py --seg brain_segmentation.nii --quadrant-fixed [--f0 100e6 --fixed-quadrant-ring-offset 10]

  # Replay a prior optimization without re-running the search (YAML optimize_antenna is overridden):
  python fdtd_brain_simulation_engine.py --config configs/paper_results_optimize.yaml \\
    --modalities-dir dataset/validation_data/001 \\
    --load-optimized-from paper_bundle_runs/optimize/data/brain_segmentation_antenna_optimization.json

CLI arguments:
  seg (positional, optional)
      Path to BraTS-style segmentation NIfTI (0,1,2,3). Omit if using --modalities or --modalities-dir.
  --modalities FLAIR T1 T1CE T2
      Paths to 4 BraTS NIfTI files. Runs 3D U-Net segmentation then FDTD.
  --modalities-dir DIR
      Folder containing four BraTS modalities (flair.nii, t1.nii, t1ce.nii, t2.nii or *_flair.nii, etc.).
  --checkpoint PATH
      Path to 3D U-Net checkpoint .pth (default: best_model.pth).
  --no-normal-brain
      When using --modalities, do not add normal brain tissue (label 4); keep only tumor classes 1–3 and background 0.

  Antenna optimization (mutually exclusive with --quadrant-fixed):
  --optimize-antenna
      Run 4-quadrant APA antenna optimization to maximize J = meanSAR_tumor / meanSAR_healthy.
  --quadrant-fixed
      Skip optimization; run FDTD with fixed 4-quadrant APA (defaults: see --help). Mutually exclusive with --optimize-antenna.
  --f0 FREQ
      Carrier frequency in Hz for 4-quadrant CW source in --optimize-antenna or --quadrant-fixed (default: 100e6).
  --opt-time-steps N
      FDTD time steps per unit-quadrant run (default: 700).
  --opt-phase-steps N
      Phase grid points per quadrant in coarse sweep (default: 24).
  --opt-amp-steps N
      Amplitude grid points per quadrant in coarse sweep (default: 9).
  --opt-amp-min, --opt-amp-max
      Amplitude bounds (default: 0.2, 2.5).
  --opt-refine-iters N
      Coordinate-descent refinement iterations (default: 8).
  --opt-multi-start N
      Number of random multi-start phase offsets (default: 3).
  --opt-freq-sweep FREQ [FREQ ...]
      Frequencies to sweep; best is auto-selected (e.g. --opt-freq-sweep 70e6 100e6 130e6 170e6 200e6).
  --opt-geom-offsets OFFSET [OFFSET ...]
      Source ring offsets (cells from PML) to sweep; small offsets keep applicator in air.
      Default: 8 10 12.
  --opt-geom-zplanes Z [Z ...]
      Z-plane indices for source placement sweep (e.g. --opt-geom-zplanes 30 41 50).
  --opt-penalty-weight W
      Penalty weight for healthy P95 SAR hotspot in objective (default: 0.1).
  --opt-source-scale S
      Global scale for optimized source in final FDTD run; SAR scales with S² (default: 1.0).
  --opt-parallel N
      Parallel workers for antenna optimization (frequency/geometry sweep, multi-start). Default: 1.
  --stream-frames
      Stream E and SAR frames to disk during FDTD (no in-memory accumulation). Use with
      --stream-frame-interval for full or dense timesteps; build animations separately from saved frames.
      After the run, build_animations_from_streamed_frames.py is invoked to build MP4s; use --slice-timestep-images to also generate per-(slice, timestep) PNGs (E/SAR/T) for the dashboard.
  --no-stream-frames
      Keep E and SAR frames in memory instead of streaming to disk (disables default streaming).
  --stream-frame-interval N
      Save a frame every N timesteps when --stream-frames (default: 1 = every step).
  --skip-animations
      Do not build or save MP4 animations; frames are still saved. Use build_animations_from_streamed_frames.py later.
  --slice-timestep-images
      When using --stream-frames: generate per-(slice, timestep) PNGs for the dashboard (skipped by default).

  Standard run (pulse types: gaussian, sinusoid, sinusoid_no_ramp, modulated_gaussian, cw):
  --time-steps N
      FDTD time steps for standard run (default: 500). Ignored when --optimize-antenna.
  --max-dim N
      Maximum grid dimension; segmentation downsampled if larger (default: 120).
  --pulse-type TYPE
      Source waveform: gaussian, cw, modulated_gaussian, sinusoid, sinusoid_no_ramp (default: gaussian).
  --prop-direction DIR
      Plane-wave direction for gaussian/modulated_gaussian: +x, -x, +y, -y, +z, -z (default: +y).
  --source-x, --source-y, --source-z
      Grid indices for point source 1 (default: grid center). For cw/sinusoid/sinusoid_no_ramp.
  --pulse-amplitude A
      Amplitude of the source pulse (default: 100). SAR and temperature scale with A².
  --pulse-freq FREQ
      Frequency in Hz for modulated_gaussian, sinusoid, sinusoid_no_ramp, cw (default: 100e6).
  --cw-periods N
      For cw/sinusoid_no_ramp: set time_steps to N periods, SAR from period 10. Unset: cw/sinusoid use min 15 periods.
  --pulse-ramp-width N
      Gaussian ramp-up width (time steps) for CW and sinusoid soft start (default: 30). Ignored for sinusoid_no_ramp.
  --use-source-2, --use-source-3
      Enable 2nd/3rd point source at antenna-like positions (cw/sinusoid/sinusoid_no_ramp).
  --source-x-2, --source-y-2, --source-z-2
      Grid indices for second point source (when --use-source-2). Default: antenna-like position.
  --source-x-3, --source-y-3, --source-z-3
      Grid indices for third point source (when --use-source-3). Default: antenna-like position.
  --source-ring-offset N
      Cells from boundary for default antenna-like positions of source 2/3 (default: 10).

  Grid resolution:
  --dx-mm MM
      Grid resolution (voxel size) in mm (default: 10). Stored in meters for FDTD, SAR, thermal solver, and NIfTI affine.
      Paper recommends 1–5 mm for anatomical detail; default 10 mm preserves backward compatibility.
  --air-padding-cells N
      Add N air voxels (label 0) on every side of the segmented volume before simulation.
      Increases grid size and helps separate sources from tissue (default: 0).
  --courant-factor F
      Safety factor for time step: dt = F * dt_courant, where dt_courant follows the Courant stability condition (paper Sec. 3.2). Default: 0.99.

Input: BraTS-style segmentation (0=background, 1=necrotic, 2=edema, 3=enhancing; optional 4=normal brain).
Output: results/{timestamp}/{data|images|animations}/{base}_*.png, *.mp4, *.npy, *.json.
  Performance and scalability JSONs include a ``backend`` field (e.g. ``numpy_numba``).

When using --modalities, segmentation is performed by the 3D U-Net from
BrainTumorSegmentation-3DUNet-StreamlitApp (Apache License 2.0). See
brain_tumor_segmentation_model.py for attribution.

Benchmark mode (Objective 5 scalability):
  --benchmark-grid-sizes N [N ...]
      Run minimal FDTD-only for each grid size N³, collect timing and memory, write scalability JSON.
  --benchmark-grid-sizes-range A B S
      Grid sizes as range: min A, max B, step S (e.g. 50 200 50 → 50,100,150,200). Alternative to --benchmark-grid-sizes.
  --benchmark-time-steps N
      FDTD time steps per benchmark run (default: 500).
"""

from hermes_drh.cli.parser import parse_args
from hermes_drh.optimization.load_optimized import LoadOptimizedError, apply_load_optimized_from_to_args
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

from hermes_drh.io.validation import write_progress as write_progress_json
from hermes_drh.workflows import run_benchmark, run_brain_simulation, run_simulation
from hermes_drh._paths import package_dir
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency in some environments
    tqdm = None


class _CLIProgressManager:
    """Terminal phase loaders driven by write_progress callbacks."""

    _PHASE_LABELS = {
        "setup": "Setup",
        "segmentation": "Segmentation",
        "antenna_optimization": "Optimization",
        "fdtd_simulation": "FDTD",
        "sar_computation": "SAR",
        "thermal_solver": "Temperature",
        "saving_and_animations": "Animation/Save",
        "complete": "Complete",
    }
    _PCT_RANGES = {
        "setup": (0, 12),
        "segmentation": (0, 10),
        "antenna_optimization": (15, 25),
        "fdtd_simulation": (25, 70),
        "sar_computation": (72, 75),
        "thermal_solver": (77, 80),
        "saving_and_animations": (85, 99),
        "complete": (99, 100),
    }

    def __init__(self):
        self._bars = {}
        self._active_phase = None

    def update(self, phase, message, percent, extra=None):
        if tqdm is None:
            return
        extra = extra or {}
        if phase != self._active_phase:
            self._active_phase = phase
        bar = self._bars.get(phase)
        if bar is None:
            bar = tqdm(
                total=100,
                desc=self._PHASE_LABELS.get(phase, phase.replace("_", " ").title()),
                unit="%",
                leave=True,
            )
            self._bars[phase] = bar

        if phase == "fdtd_simulation" and extra.get("time_steps"):
            ts = int(extra.get("time_step", 0))
            ttot = int(extra["time_steps"])
            if ttot > 0:
                target = max(0, min(100, int(round(100.0 * ts / ttot))))
                bar.set_postfix_str(f"{ts}/{ttot}")
            else:
                target = self._phase_percent(phase, percent)
        else:
            target = self._phase_percent(phase, percent)

        if target > bar.n:
            bar.update(target - bar.n)
        bar.set_postfix_str(message[:80])

        if phase == "complete" or target >= 100:
            if bar.n < 100:
                bar.update(100 - bar.n)
            bar.close()

    def finalize(self):
        if tqdm is None:
            return
        for bar in self._bars.values():
            if not getattr(bar, "disable", False):
                if bar.n < 100:
                    bar.update(100 - bar.n)
                bar.close()
        self._bars.clear()

    def _phase_percent(self, phase, percent):
        lo, hi = self._PCT_RANGES.get(phase, (0, 100))
        p = max(0.0, min(100.0, float(percent)))
        if hi <= lo:
            return int(round(p))
        out = 100.0 * (p - lo) / (hi - lo)
        return max(0, min(100, int(round(out))))

def main(argv=None):
    args = parse_args(argv)
    try:
        apply_load_optimized_from_to_args(args)
    except (LoadOptimizedError, FileNotFoundError, OSError, json.JSONDecodeError) as e:
        print(f"Error: --load-optimized-from: {e}", file=sys.stderr)
        sys.exit(2)
    # Create single timestamped results directory (only in main process; workers do not run this)
    # When BENCHMARK_RESULTS_DIR is set (by parent benchmark), use it so all subprocess output goes in one dir
    now = datetime.now()
    timestamp_str = now.strftime("%d%m%y-%H%M%S")  # DDMMYY-HHMMSS
    if os.environ.get("BENCHMARK_RESULTS_DIR"):
        RESULTS_DIR = os.path.abspath(os.environ["BENCHMARK_RESULTS_DIR"])
    elif getattr(args, "results_dir", None):
        RESULTS_DIR = os.path.abspath(args.results_dir)
    else:
        RESULTS_DIR = os.path.join("results", f"{timestamp_str}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    PROGRESS_DIR = os.path.join("results", "uploads")
    PROGRESS_FILE = os.path.join(PROGRESS_DIR, "last_run_progress.json")
    # When BENCHMARK_GRID_SIZE is set (benchmark subprocess), put output in data/N, images/N, animations/N
    benchmark_grid_subdir = os.environ.get("BENCHMARK_GRID_SIZE")
    if benchmark_grid_subdir is not None:
        subdir = str(benchmark_grid_subdir)
        DATA_DIR = os.path.join(RESULTS_DIR, "data", subdir)
        IMAGES_DIR = os.path.join(RESULTS_DIR, "images", subdir)
        ANIMATIONS_DIR = os.path.join(RESULTS_DIR, "animations", subdir)
    else:
        DATA_DIR = os.path.join(RESULTS_DIR, "data")
        IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
        ANIMATIONS_DIR = os.path.join(RESULTS_DIR, "animations")
    E_FRAMES_DIR = os.path.join(DATA_DIR, "E_frames")
    SAR_FRAMES_DIR = os.path.join(DATA_DIR, "SAR_frames")
    TEMPERATURE_FRAMES_DIR = os.path.join(DATA_DIR, "Temperature_frames")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(E_FRAMES_DIR, exist_ok=True)
    os.makedirs(SAR_FRAMES_DIR, exist_ok=True)
    os.makedirs(TEMPERATURE_FRAMES_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANIMATIONS_DIR, exist_ok=True)

    engine_dir = str(package_dir())
    paths = SimpleNamespace(
        results_dir=RESULTS_DIR,
        data_dir=DATA_DIR,
        images_dir=IMAGES_DIR,
        animations_dir=ANIMATIONS_DIR,
        e_frames_dir=E_FRAMES_DIR,
        sar_frames_dir=SAR_FRAMES_DIR,
        temperature_frames_dir=TEMPERATURE_FRAMES_DIR,
        progress_dir=PROGRESS_DIR,
        progress_file=PROGRESS_FILE,
        script_dir=engine_dir,
    )
    cli_progress = _CLIProgressManager()

    def _write_progress(phase, message, percent, phases_done=None, extra=None):
        write_progress_json(
            phase,
            message,
            percent,
            PROGRESS_DIR,
            PROGRESS_FILE,
            phases_done=phases_done,
            extra=extra,
        )
        cli_progress.update(phase, message, percent, extra=extra)

    _write_progress("setup", "Output directories created", 0, [])
    print("Output directories (paper Sec. 3.2.8 Data Analysis and Validation):")
    print(f"  Root: {RESULTS_DIR}/")
    print(f"  Data: {DATA_DIR}/")
    print(f"  E-frames: {E_FRAMES_DIR}/")
    print(f"  Images: {IMAGES_DIR}/")
    print(f"  Animations: {ANIMATIONS_DIR}/")

    # -------------------------------------------------------------------------
    # Performance Evaluation (paper Sec. 3.2.6): benchmark mode.
    # -------------------------------------------------------------------------
    in_benchmark_mode = (
        args.benchmark_grid_sizes is not None
        or getattr(args, "benchmark_grid_sizes_range", None) is not None
    )
    if in_benchmark_mode:
        run_benchmark(args, paths, _write_progress)
        cli_progress.finalize()
        sys.exit(0)

    anatomy = getattr(args, "anatomy", "brain")
    if anatomy in ("breast", "cervix"):
        run_simulation(args, paths, _write_progress, anatomy=anatomy)
    else:
        run_brain_simulation(args, paths, _write_progress)
    cli_progress.finalize()


if __name__ == "__main__":
    main()
