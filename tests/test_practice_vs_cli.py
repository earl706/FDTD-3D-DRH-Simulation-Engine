"""Compare practice_cli.py to CODE/cli.py (optional CLI drill file)."""

from __future__ import annotations

import pytest

import cli as ref
from tests.conftest import load_optional_practice_module

mod = load_optional_practice_module("practice.py")
if mod is None:
    pytest.skip(
        "Add CODE/practice.py and reimplement CODE/cli.py parse_args from memory.",
        allow_module_level=True,
    )

_CLI_SYMBOLS = [
    "BENCHMARK_GRID_SIZES_DEFAULT",
    "BENCHMARK_TIME_STEPS_DEFAULT",
    "parse_args",
]


@pytest.mark.parametrize("name", _CLI_SYMBOLS)
def test_symbol_exported(name):
    assert hasattr(mod, name), f"practice.py missing {name!r}"


def test_benchmark_constants_match_reference():
    assert mod.BENCHMARK_GRID_SIZES_DEFAULT == ref.BENCHMARK_GRID_SIZES_DEFAULT
    assert mod.BENCHMARK_TIME_STEPS_DEFAULT == ref.BENCHMARK_TIME_STEPS_DEFAULT


def _as_dict(a):
    return vars(a).copy()


def test_parse_args_defaults_match_reference_subset():
    got = _as_dict(mod.parse_args([]))
    exp = _as_dict(ref.parse_args([]))
    keys = [
        "anatomy",
        "seg",
        "modalities",
        "modalities_dir",
        "quadrant_fixed",
        "optimize_antenna",
        "f0",
        "opt_source_scale",
        "quadrant_air_margin_cells",
        "air_padding_cells",
        "stream_frames",
        "stream_frame_interval",
        "subsample",
        "pulse_type",
        "max_dim",
    ]
    for k in keys:
        assert got[k] == exp[k], f"default mismatch for {k!r}: {got[k]!r} != {exp[k]!r}"


def test_parse_args_quadrant_fixed_flags():
    argv = [
        "--quadrant-fixed",
        "--f0",
        "100e6",
        "--opt-source-scale",
        "5000",
        "--fixed-quadrant-alphas",
        "1",
        "1",
        "1",
        "1",
        "--max-dim",
        "80",
        "--quadrant-air-margin-cells",
        "5",
        "--air-padding-cells",
        "20",
        "--sub-sample",
        "10",
    ]
    got = _as_dict(mod.parse_args(argv))
    exp = _as_dict(ref.parse_args(argv))
    keys = [
        "quadrant_fixed",
        "f0",
        "opt_source_scale",
        "fixed_quadrant_alphas",
        "max_dim",
        "quadrant_air_margin_cells",
        "air_padding_cells",
        "subsample",
    ]
    for k in keys:
        assert got[k] == exp[k], f"arg mismatch for {k!r}: {got[k]!r} != {exp[k]!r}"


def test_parse_args_rejects_mutually_exclusive_apa_modes():
    argv = ["--optimize-antenna", "--quadrant-fixed", "fake.nii"]
    with pytest.raises(SystemExit):
        mod.parse_args(argv)


def _assert_keys_match(argv, keys):
    got = _as_dict(mod.parse_args(argv))
    exp = _as_dict(ref.parse_args(argv))
    for k in keys:
        assert got[k] == exp[k], f"arg mismatch for {k!r}: {got[k]!r} != {exp[k]!r}"


def test_parse_args_input_bucket_all_args():
    argv = [
        "--anatomy",
        "cervix",
        "case.npy",
        "--modalities",
        "flair.nii",
        "t1.nii",
        "t1ce.nii",
        "t2.nii",
        "--modalities-dir",
        "dataset/case001",
        "--checkpoint",
        "best_model.pth",
        "--no-normal-brain",
    ]
    _assert_keys_match(
        argv,
        [
            "anatomy",
            "seg",
            "modalities",
            "modalities_dir",
            "checkpoint",
            "no_normal_brain",
        ],
    )


def test_parse_args_quadrant_optimized_bucket_all_args():
    argv = [
        "--optimize-antenna",
        "--f0",
        "130e6",
        "--opt-time-steps",
        "600",
        "--opt-phase-steps",
        "16",
        "--opt-amp-steps",
        "7",
        "--opt-amp-min",
        "0.4",
        "--opt-amp-max",
        "2.2",
        "--opt-refine-iters",
        "5",
        "--opt-multi-start",
        "2",
        "--opt-freq-sweep",
        "70e6",
        "100e6",
        "130e6",
        "--opt-geom-offsets",
        "8",
        "10",
        "12",
        "--opt-geom-zplanes",
        "30",
        "35",
        "--opt-penalty-weight",
        "0.2",
        "--opt-parallel",
        "2",
        "--opt-source-scale",
        "1500",
    ]
    _assert_keys_match(
        argv,
        [
            "optimize_antenna",
            "f0",
            "opt_time_steps",
            "opt_phase_steps",
            "opt_amp_steps",
            "opt_amp_min",
            "opt_amp_max",
            "opt_refine_iters",
            "opt_multi_start",
            "opt_freq_sweep",
            "opt_geom_offsets",
            "opt_geom_zplanes",
            "opt_penalty_weight",
            "opt_parallel",
            "opt_source_scale",
        ],
    )


def test_parse_args_quadrant_fixed_bucket_all_args():
    argv = [
        "--quadrant-fixed",
        "--f0",
        "100e6",
        "--opt-source-scale",
        "5000",
        "--fixed-quadrant-ring-offset",
        "12",
        "--fixed-quadrant-z-plane",
        "41",
        "--fixed-quadrant-dipole-half-len",
        "11",
        "--fixed-quadrant-alphas",
        "1.0",
        "0.8",
        "1.2",
        "1.1",
        "--fixed-quadrant-phases-deg",
        "0",
        "90",
        "180",
        "270",
    ]
    _assert_keys_match(
        argv,
        [
            "quadrant_fixed",
            "f0",
            "opt_source_scale",
            "fixed_quadrant_ring_offset",
            "fixed_quadrant_z_plane",
            "fixed_quadrant_dipole_half_len",
            "fixed_quadrant_alphas",
            "fixed_quadrant_phases_deg",
        ],
    )


def test_parse_args_spacing_safety_bucket_all_args():
    argv = [
        "--quadrant-air-margin-cells",
        "5",
        "--air-padding-cells",
        "20",
    ]
    _assert_keys_match(argv, ["quadrant_air_margin_cells", "air_padding_cells"])


def test_parse_args_grid_time_benchmark_bucket_all_args():
    argv = [
        "--pulse-amplitude",
        "250.0",
        "--time-steps",
        "900",
        "--max-dim",
        "96",
        "--dx-mm",
        "8.0",
        "--courant-factor",
        "0.95",
        "--benchmark-grid-sizes",
        "64",
        "96",
        "128",
        "--benchmark-time-steps",
        "200",
        "--benchmark-grid-sizes-range",
        "60",
        "120",
        "30",
    ]
    _assert_keys_match(
        argv,
        [
            "pulse_amplitude",
            "time_steps",
            "max_dim",
            "dx_mm",
            "courant_factor",
            "benchmark_grid_sizes",
            "benchmark_time_steps",
            "benchmark_grid_sizes_range",
        ],
    )


def test_parse_args_output_animation_bucket_all_args():
    argv = [
        "--no-stream-frames",
        "--stream-frame-interval",
        "3",
        "--sub-sample",
        "10",
        "--skip-animations",
        "--slice-timestep-images",
    ]
    _assert_keys_match(
        argv,
        [
            "stream_frames",
            "stream_frame_interval",
            "subsample",
            "skip_animations",
            "slice_timestep_images",
        ],
    )


def test_parse_args_standard_source_bucket_all_args():
    argv = [
        "--pulse-type",
        "sinusoid",
        "--prop-direction",
        "+z",
        "--source-x",
        "20",
        "--source-y",
        "22",
        "--source-z",
        "24",
        "--use-source-2",
        "--use-source-3",
        "--source-x-2",
        "18",
        "--source-y-2",
        "19",
        "--source-z-2",
        "20",
        "--source-x-3",
        "50",
        "--source-y-3",
        "48",
        "--source-z-3",
        "24",
        "--source-ring-offset",
        "12",
        "--pulse-freq",
        "150e6",
        "--cw-periods",
        "20",
        "--pulse-ramp-width",
        "45",
    ]
    _assert_keys_match(
        argv,
        [
            "pulse_type",
            "prop_direction",
            "source_x",
            "source_y",
            "source_z",
            "use_source_2",
            "use_source_3",
            "source_x_2",
            "source_y_2",
            "source_z_2",
            "source_x_3",
            "source_y_3",
            "source_z_3",
            "source_ring_offset",
            "pulse_freq",
            "cw_periods",
            "pulse_ramp_width",
        ],
    )
