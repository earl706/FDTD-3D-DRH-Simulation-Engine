import argparse
import os
import sys

from hermes_drh.settings.simulation import (
    argv_without_config_option,
    load_simulation_config,
    validated_defaults_for_parser,
)

# Benchmark config (Objective 5: performance evaluation)
BENCHMARK_GRID_SIZES_DEFAULT = [100, 200, 300]
BENCHMARK_TIME_STEPS_DEFAULT = 500

def _default_checkpoint():
    from hermes_drh._paths import default_checkpoint_path

    p = default_checkpoint_path()
    if p:
        return p
    return os.path.join(os.getcwd(), "best_model.pth")


_DEFAULT_CHECKPOINT = _default_checkpoint()


def _create_parser():
    parser = argparse.ArgumentParser(
        description="3D FDTD brain segmentation simulation. Provide either a segmentation NIfTI, 4 BraTS modality paths, or a modalities directory.",
        epilog=(
            "Optional: --config PATH.yaml loads defaults from YAML (PyYAML). "
            "Keys must match argparse destination names in snake_case (see simulation_config.py). "
            "Nested YAML mappings are flattened. Later CLI flags override YAML. "
            "Example: configs/simulation_example.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "seg",
        nargs="?",
        default=None,
        help="Path to BraTS-style segmentation NIfTI (0,1,2,3). For breast/cervix: path to .npy. Omit if using --modalities or --modalities-dir (brain only).",
    )
    parser.add_argument(
        "--anatomy",
        choices=("brain", "breast", "cervix"),
        default="brain",
        help="Anatomy for the shared pipeline: brain (default), breast, or cervix. Breast/cervix use --seg path to .npy.",
    )
    parser.add_argument(
        "--modalities",
        nargs=4,
        metavar=("FLAIR", "T1", "T1CE", "T2"),
        help="Paths to 4 BraTS NIfTI files (FLAIR, T1, T1CE, T2). Runs 3D U-Net segmentation then FDTD.",
    )
    parser.add_argument(
        "--modalities-dir",
        metavar="DIR",
        help="Path to a folder containing the four BraTS modalities (flair.nii, t1.nii, t1ce.nii, t2.nii or *_flair.nii, *_t1.nii, etc.). Auto-loads and runs segmentation.",
    )
    parser.add_argument(
        "--checkpoint",
        default=_DEFAULT_CHECKPOINT,
        help=f"Path to 3D U-Net checkpoint .pth (default: {_DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--no-normal-brain",
        action="store_true",
        help="When using --modalities, do not add normal brain tissue (label 4); keep only tumor classes 1–3 and background 0.",
    )
    # --- Bucket 2: Quadrant/APA (optimized or fixed; mutually exclusive mode flag) ---
    _apa_group = parser.add_mutually_exclusive_group()
    _apa_group.add_argument(
        "--optimize-antenna",
        action="store_true",
        help="Run 4-quadrant APA antenna optimization to maximize SAR tumor/healthy ratio (Houle Ch.6 style).",
    )
    _apa_group.add_argument(
        "--quadrant-fixed",
        action="store_true",
        help=(
            "Run 4-quadrant APA FDTD with fixed phases/amplitudes (skips optimization). "
            "Defaults: amplitude 1.0 and phase 0° for each of the four quadrants; carrier --f0 (default 100 MHz); "
            "ring offset npml+2 (override with --fixed-quadrant-ring-offset); "
            "dipole half-length 9 cells (--fixed-quadrant-dipole-half-len); "
            "z_plane at tumor centroid axial index (override with --fixed-quadrant-z-plane). "
            "Time steps match the optimized run: ~15 periods, SAR from period 10."
        ),
    )
    parser.add_argument(
        "--f0",
        type=float,
        default=100e6,
        help="Carrier frequency in Hz for --optimize-antenna and --quadrant-fixed 4-quadrant CW sources (default: 100 MHz).",
    )
    parser.add_argument(
        "--opt-time-steps",
        type=int,
        default=700,
        help="Number of FDTD time steps per unit-quadrant run during optimization (default: 700).",
    )
    parser.add_argument(
        "--opt-phase-steps",
        type=int,
        default=24,
        help="Number of phase grid points per quadrant in coarse sweep (default: 24).",
    )
    parser.add_argument(
        "--opt-amp-steps",
        type=int,
        default=9,
        help="Number of amplitude grid points per quadrant in coarse sweep (default: 9).",
    )
    parser.add_argument(
        "--opt-amp-min",
        type=float,
        default=0.2,
        help="Minimum amplitude bound for optimization (default: 0.2).",
    )
    parser.add_argument(
        "--opt-amp-max",
        type=float,
        default=2.5,
        help="Maximum amplitude bound for optimization (default: 2.5).",
    )
    parser.add_argument(
        "--opt-refine-iters",
        type=int,
        default=8,
        help="Number of coordinate-descent refinement iterations (default: 8).",
    )
    parser.add_argument(
        "--opt-multi-start",
        type=int,
        default=3,
        help="Number of random multi-start initial phase offsets for optimization (default: 3). "
        "Helps escape local optima.",
    )
    parser.add_argument(
        "--opt-freq-sweep",
        nargs="+",
        type=float,
        default=None,
        help="List of frequencies (Hz) to sweep before full optimization. "
        "The best f0 is auto-selected. Example: --opt-freq-sweep 70e6 100e6 130e6 170e6 200e6",
    )
    parser.add_argument(
        "--opt-geom-offsets",
        nargs="+",
        type=int,
        default=None,
        help="List of source ring offsets (cells from PML) to sweep for geometry optimization. "
        "Small offsets (e.g. 8 10 12) keep the applicator ring in the air region surrounding the head. "
        "When --optimize-antenna is used and neither this nor --opt-geom-zplanes is set, defaults to 8 10 12.",
    )
    parser.add_argument(
        "--opt-geom-zplanes",
        nargs="+",
        type=int,
        default=None,
        help="List of z-plane indices for source placement. "
        "Example: --opt-geom-zplanes 30 41 50  (default: tumor centroid z).",
    )
    parser.add_argument(
        "--opt-penalty-weight",
        type=float,
        default=0.1,
        help="Penalty weight for healthy-tissue P95 SAR hotspot in objective. "
        "Effective objective = J - weight * P95_healthy_SAR / mean_tumor_SAR. "
        "Default: 0.1 to limit P95 healthy SAR; use 0 for pure J maximization.",
    )
    parser.add_argument(
        "--opt-parallel",
        type=int,
        default=1,
        help="Number of parallel workers for antenna optimization (frequency sweep, geometry sweep, multi-start). "
        "Default: 1 (serial). Use 2–4 for sweeps on 16 GB RAM; multi-start can use more (e.g. 8 on M3).",
    )
    parser.add_argument(
        "--opt-source-scale",
        type=float,
        default=1.0,
        help="Global scale factor for the 4-quadrant source in the optimized or --quadrant-fixed FDTD run. "
        "SAR scales with scale². Use >1 for non-trivial temperature rise (e.g. 1e3--1e4). Default: 1.0.",
    )
    parser.add_argument(
        "--fixed-quadrant-ring-offset",
        type=int,
        default=None,
        help="Cells from PML for APA ring placement in --quadrant-fixed mode (default: npml+2, same as build_quadrant_sources).",
    )
    parser.add_argument(
        "--fixed-quadrant-z-plane",
        type=int,
        default=None,
        help="Axial (k) index for dipole plane in --quadrant-fixed mode (default: tumor centroid z).",
    )
    parser.add_argument(
        "--fixed-quadrant-dipole-half-len",
        type=int,
        default=9,
        help="Half-length of each dipole arm in cells for --quadrant-fixed (default: 9).",
    )
    parser.add_argument(
        "--fixed-quadrant-alphas",
        nargs=4,
        type=float,
        metavar=("A1", "A2", "A3", "A4"),
        default=None,
        help="Per-quadrant amplitude (4 values) for --quadrant-fixed (default: 1 1 1 1).",
    )
    parser.add_argument(
        "--fixed-quadrant-phases-deg",
        nargs=4,
        type=float,
        metavar=("P1", "P2", "P3", "P4"),
        default=None,
        help="Per-quadrant phase in degrees for --quadrant-fixed (default: 0 0 0 0).",
    )
    parser.add_argument(
        "--load-optimized-from",
        metavar="PATH",
        default=None,
        dest="load_optimized_from",
        help=(
            "Skip antenna optimization: load f0, quadrant amplitudes/phases (rad), ring_offset, "
            "and z_plane from a prior run's *_antenna_optimization.json or *_metadata.json. "
            "PATH may be that file or a directory containing exactly one such artifact. "
            "Overrides optimize_antenna from YAML with quadrant-fixed replay. "
            "freq_sweep.json is diagnostic-only and is not used."
        ),
    )
    # --- Bucket 3: Spacing/safety ---
    parser.add_argument(
        "--quadrant-air-margin-cells",
        type=int,
        default=18,
        help=(
            "Minimum air-to-tissue distance (in voxel cells) required for the 4 quadrant "
            "source GAP voxels. Uses the segmentation labels to estimate tissue vs air. "
            "Default: 18 (recommended range ~15-20). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--air-padding-cells",
        type=int,
        default=0,
        help=(
            "Pad the segmented label volume with this many air voxels (label 0) on all "
            "sides before building the FDTD grid. This changes grid size and can create "
            "extra source-to-tissue spacing independent of --quadrant-air-margin-cells. "
            "Default: 0 (disabled)."
        ),
    )
    # --- Bucket 4: Grid/time + benchmark ---
    parser.add_argument(
        "--pulse-amplitude",
        type=float,
        default=100.0,
        help="Amplitude of the Gaussian pulse in the standard (non-optimized) FDTD run. "
        "SAR and temperature scale with amplitude squared. (default: 100.0)",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=500,
        help="Number of FDTD time steps for standard (non-optimized, non-quadrant-fixed) run (default: 500). "
        "Ignored when --optimize-antenna or --quadrant-fixed is used (steps derived from --f0).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=120,
        help="Maximum grid dimension; segmentation is downsampled if larger (default: 120).",
    )
    parser.add_argument(
        "--dx-mm",
        type=float,
        default=10.0,
        help="Grid resolution (voxel size) in mm (default: 10). Converted to meters internally. "
        "Paper recommends 1–5 mm for anatomical detail; 10 mm is used for backward compatibility.",
    )
    parser.add_argument(
        "--courant-factor",
        type=float,
        default=0.99,
        help="Safety factor for time step: dt = factor * dt_courant, where dt_courant is from the Courant stability condition (default: 0.99).",
    )
    parser.add_argument(
        "--benchmark-grid-sizes",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help="Benchmark mode: run minimal FDTD for each N³ grid; write scalability JSON. "
        "Example: --benchmark-grid-sizes 100 200 300. Default grid sizes: 100 200 300.",
    )
    parser.add_argument(
        "--benchmark-time-steps",
        type=int,
        default=BENCHMARK_TIME_STEPS_DEFAULT,
        help=f"FDTD time steps per benchmark run (default: {BENCHMARK_TIME_STEPS_DEFAULT}).",
    )
    parser.add_argument(
        "--benchmark-grid-sizes-range",
        nargs=3,
        type=int,
        default=None,
        metavar=("A", "B", "S"),
        help="Benchmark grid sizes as range: min A, max B, step S (e.g. 50 200 50 → 50,100,150,200). "
        "Alternative to --benchmark-grid-sizes.",
    )
    parser.add_argument(
        "--results-dir",
        metavar="DIR",
        default=None,
        help="Write this run under DIR/ (data/, images/, animations/) instead of results/DDMMYY-HHMMSS. "
        "Use for reproducible paper figure bundles. Ignored when BENCHMARK_RESULTS_DIR is set (benchmark child).",
    )
    # --- Bucket 5: Output/animation ---
    parser.add_argument(
        "--stream-frames",
        action="store_true",
        default=True,
        dest="stream_frames",
        help="Stream E and SAR frames to disk during FDTD instead of keeping in memory (default: True). "
        "Use with --stream-frame-interval to control density. Enables full (or dense) timestep "
        "saving without OOM; animations can be built separately from saved frames.",
    )
    parser.add_argument(
        "--no-stream-frames",
        action="store_false",
        dest="stream_frames",
        help="Keep E and SAR frames in memory instead of streaming to disk (disables default streaming).",
    )
    parser.add_argument(
        "--stream-frame-interval",
        type=int,
        default=1,
        help="Save a frame every N timesteps when streaming (default: 1 = every step).",
    )
    parser.add_argument(
        "--sub-sample",
        type=int,
        default=1,
        dest="subsample",
        help=(
            "Use every Nth saved frame when building animations (default: 1 = all frames). "
            "Applies to both in-memory and streamed animation builders."
        ),
    )
    parser.add_argument(
        "--skip-animations",
        action="store_true",
        help="Do not build or save MP4 animations (2D/3D). Frames are still saved to E_frames/, SAR_frames/, "
        "Temperature_frames/ when applicable; use build_animations_from_streamed_frames.py later to build videos.",
    )
    parser.add_argument(
        "--slice-timestep-images",
        action="store_true",
        help="When using --stream-frames: generate per-(slice, timestep) PNGs (E/SAR/T) for the dashboard. By default they are skipped.",
    )
    # --- Bucket 6: Standard-source controls (ignored for optimized/quadrant-fixed loop) ---
    parser.add_argument(
        "--pulse-type",
        type=str,
        default="gaussian",
        choices=[
            "gaussian",
            "cw",
            "modulated_gaussian",
            "sinusoid",
            "sinusoid_no_ramp",
        ],
        help="Source waveform: gaussian, cw, modulated_gaussian, sinusoid (ramp+sin), or sinusoid_no_ramp (default: gaussian).",
    )
    parser.add_argument(
        "--prop-direction",
        type=str,
        default="+y",
        choices=[
            "+x",
            "-x",
            "+y",
            "-y",
            "+z",
            "-z",
        ],
        help="Plane-wave propagation direction for gaussian/modulated_gaussian (default: +y). Ignored for cw/sinusoid/sinusoid_no_ramp.",
    )
    parser.add_argument(
        "--source-x",
        type=int,
        default=None,
        help="Grid index i for CW point source (standard run, pulse-type cw). Default: grid center. Clamped to interior.",
    )
    parser.add_argument(
        "--source-y",
        type=int,
        default=None,
        help="Grid index j for CW point source (standard run, pulse-type cw). Default: grid center.",
    )
    parser.add_argument(
        "--source-z",
        type=int,
        default=None,
        help="Grid index k for point source 1 (cw/sinusoid/sinusoid_no_ramp). Default: grid center.",
    )
    parser.add_argument(
        "--use-source-2",
        action="store_true",
        help="Enable second point source at antenna-like position (for cw/sinusoid/sinusoid_no_ramp). Use --source-x-2 etc. to override.",
    )
    parser.add_argument(
        "--use-source-3",
        action="store_true",
        help="Enable third point source at antenna-like position (for cw/sinusoid/sinusoid_no_ramp). Use --source-x-3 etc. to override.",
    )
    parser.add_argument(
        "--source-x-2",
        type=int,
        default=None,
        help="Grid index i for second point source. Used when --use-source-2; default antenna-like position if unset.",
    )
    parser.add_argument(
        "--source-y-2",
        type=int,
        default=None,
        help="Grid index j for second point source.",
    )
    parser.add_argument(
        "--source-z-2",
        type=int,
        default=None,
        help="Grid index k for second point source.",
    )
    parser.add_argument(
        "--source-x-3",
        type=int,
        default=None,
        help="Grid index i for third point source. Used when --use-source-3; default antenna-like if unset.",
    )
    parser.add_argument(
        "--source-y-3",
        type=int,
        default=None,
        help="Grid index j for third point source.",
    )
    parser.add_argument(
        "--source-z-3",
        type=int,
        default=None,
        help="Grid index k for third point source.",
    )
    parser.add_argument(
        "--source-ring-offset",
        type=int,
        default=10,
        help="Cells from boundary for default antenna-like positions of source 2/3 when --use-source-2/3 (default: 10).",
    )
    parser.add_argument(
        "--pulse-freq",
        type=float,
        default=100e6,
        help="Frequency (Hz) for modulated_gaussian, sinusoid, sinusoid_no_ramp, and cw (default: 100e6).",
    )
    parser.add_argument(
        "--cw-periods",
        type=int,
        default=None,
        help="When pulse-type is cw or sinusoid_no_ramp: set time_steps to this many periods and SAR from period 10. If unset, cw/sinusoid use min 15 periods; sinusoid_no_ramp uses --time-steps as-is.",
    )
    parser.add_argument(
        "--pulse-ramp-width",
        type=float,
        default=30.0,
        help="Gaussian ramp-up width (time steps) for CW and sinusoid (ramp+sin) soft start (default: 30). Ignored for sinusoid_no_ramp.",
    )
    return parser


def parse_args(argv=None):
    """
    Parse command-line arguments. Pass ``argv`` for testing; default uses ``sys.argv``.

    If ``--config PATH`` is present, load YAML defaults first; CLI arguments override them.
    ``--config`` is parsed via a lightweight pre-parser so it need not be registered on the main parser.
    """
    argv_list = list(sys.argv[1:] if argv is None else argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", metavar="PATH", default=None)
    pre_ns, _ = pre.parse_known_args(argv_list)

    parser = _create_parser()
    if pre_ns.config:
        flat = load_simulation_config(pre_ns.config)
        parser.set_defaults(**validated_defaults_for_parser(parser, flat))

    argv_clean = argv_without_config_option(argv_list)
    return parser.parse_args(argv_clean)
