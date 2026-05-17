#!/usr/bin/env python3
"""
Bibliographic-style validation figure: lossy homogeneous dielectric, plane wave +y.

Material parameters (εr=4, σ=0.05 S/m) follow Houle & Sullivan, Ch.~1
(lossy dielectric example; 1D sinusoid script in the book uses f=700 MHz — here
f0=100 MHz matches the thesis DRH band while keeping the same loss tangent scale).

Runs the same 3D uniform-grid + TFSF-style path as CODE/tests/test_validation_fdtd.py,
extracts accumulated Ex phasor along the propagation line, and overlays the classical
plane-wave attenuation magnitude exp(-α y) with α from the lossy-dielectric formula.

Usage (from CODE/):
  python plot_houle_style_plane_wave_validation.py [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = (
    CODE_DIR.parent
    / "PAPER"
    / "figures"
    / "results_three_runs"
    / "houle_style_plane_wave_validation.png"
)


def _load_build_state():
    path = CODE_DIR / "tests" / "test_validation_fdtd.py"
    spec = importlib.util.spec_from_file_location("_tv_fdtd", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._build_uniform_fdtd_state


def attenuation_constant_np_per_m(
    f0_hz: float, eps_r: float, sigma_s_per_m: float
) -> float:
    """Plane-wave attenuation α (Np/m) in a homogeneous lossy dielectric."""
    eps0 = 8.8541878128e-12
    mu0 = 4.0e-7 * np.pi
    omega = 2.0 * np.pi * f0_hz
    eps_prime = eps_r * eps0
    ratio = sigma_s_per_m / (omega * eps_prime)
    inner = np.sqrt(1.0 + ratio**2)
    return float(omega * np.sqrt((mu0 * eps_prime / 2.0) * (inner - 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output PNG path",
    )
    args = ap.parse_args()

    _build_uniform_fdtd_state = _load_build_state()
    from core.fdtd.loops import run_fdtd_standard_loop

    # Houle Ch.1 lossy slab parameters (εr, σ); thesis DRH carrier for wavelength context.
    f0_hz = 100e6
    eps_r = 4.0
    sigma = 0.05
    dx_m = 0.01
    courant_factor = 0.99
    nx, ny, nz = 60, 120, 40
    time_steps = 1400

    state, npml = _build_uniform_fdtd_state(
        nx=nx,
        ny=ny,
        nz=nz,
        dx_m=dx_m,
        courant_factor=courant_factor,
        f0_hz=f0_hz,
        eps_r=eps_r,
        sigma_s_per_m=sigma,
        pulse_type="modulated_gaussian",
        time_steps=time_steps,
        prop_direction="+y",
    )
    run_fdtd_standard_loop(state)

    cx = state.simulation_size_x // 2
    z = state.source_z
    y0 = npml + 6
    y1 = state.simulation_size_y - npml - 6
    phasor = state.real_pt[0, cx, y0:y1, z] + 1j * state.imag_pt[0, cx, y0:y1, z]
    amp = np.abs(phasor)
    y_m = np.arange(amp.size, dtype=np.float64) * dx_m + y0 * dx_m

    alpha = attenuation_constant_np_per_m(f0_hz, eps_r, sigma)

    # Past the source peak: compare shape to exp(-α Δy) normalized to 1 at a reference cell.
    peak_idx = int(np.argmax(amp))
    y_ref_idx = min(peak_idx + 10, amp.size - 3)
    y_ref_m = y_m[y_ref_idx]
    norm_fdtd = amp / (amp[y_ref_idx] + 1e-30)
    norm_theory = np.exp(-alpha * (y_m - y_ref_m))
    norm_theory /= norm_theory[y_ref_idx] + 1e-30

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150)

    ax1.plot(y_m * 100.0, amp, color="C0", lw=1.2)
    ax1.set_xlabel("Distance along $+y$ (cm)")
    ax1.set_ylabel(r"$|E_x|$ (Fourier magnitude, arb.\ units)")
    ax1.set_title("HERMES FDTD (uniform lossy medium)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        y_m * 100.0,
        norm_fdtd,
        label=r"HERMES $|E_x|$ (norm.\ at $y_{\mathrm{ref}}$)",
        color="C0",
        lw=1.2,
    )
    ax2.plot(
        y_m * 100.0,
        norm_theory,
        "--",
        color="C1",
        lw=1.5,
        label=r"Analytic $e^{-\alpha (y-y_{\mathrm{ref}})}$",
    )
    ax2.set_xlabel("Distance along $+y$ (cm)")
    ax2.set_ylabel("Normalized magnitude")
    ax2.set_title(r"Overlay vs.\ lossy plane-wave attenuation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Houle-style lossy dielectric ($\varepsilon_r={eps_r:g}$, "
        rf"$\sigma={sigma:g}$ S/m, $f_0={f0_hz/1e6:.0f}$ MHz); "
        rf"$\alpha \approx {alpha:.3f}$ Np/m",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
