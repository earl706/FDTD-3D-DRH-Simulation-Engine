# HERMES → PyPI packaging refactor checklist

Target distribution name (from thesis materials): **`hermes-drh`**  
Suggested import package: **`hermes_drh`** (avoid top-level names `core`, `config`, `cli` on PyPI—they are ambiguous and collide with other projects).

**Scope:** The PyPI wheel/sdist contains only the **simulation engine and its runtime tooling** (FDTD pipeline, segmentation hook, optimization, validation exports, animations, optional dashboard). It does **not** include thesis/paper automation that writes LaTeX or syncs figures into `PAPER/`.

---

## Handoff status (2026-05-15)

**Done (agent):**

- [x] `src/hermes_drh/` package with simulation, settings, workflows, CLI, I/O, optimization, visualization, dashboard, compat facades
- [x] `pyproject.toml`, `LICENSE` (MIT), `README.md`, `CHANGELOG.md`, `.github/workflows/ci.yml`
- [x] Console scripts: `hermes-simulate`, `hermes-build-animations`, `hermes-dashboard`
- [x] Root shims: `fdtd_brain_simulation_engine.py`, `build_animations_from_streamed_frames.py`
- [x] `run_paper_bundle.py` calls `hermes-simulate` when on PATH
- [x] `tests/test_packaging_smoke.py` (4 tests pass in `.venv`)
- [x] Wheel build verified; paper/LaTeX scripts **not** in wheel

**You still need to:**

- [ ] Confirm PyPI name + university MIT approval
- [ ] Create PyPI / TestPyPI accounts and upload (`twine upload`)
- [ ] Host `best_model.pth` and document `HERMES_CHECKPOINT`
- [ ] Run full thesis smoke: `hermes-simulate` + `run_paper_bundle.py` on your data
- [ ] (Optional) Remove duplicate legacy `core/` / flat modules after migration period

**Dev install (from `CODE/`):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_packaging_smoke.py -q
hermes-simulate --help
```

Nested repos such as `BC_MRI_SEG/` stay **out of the wheel** unless explicitly split into optional extras.

---

## In scope vs out of scope

### In scope (package `hermes-drh`)

| Area | Current files / dirs |
|------|----------------------|
| Simulation engine | `fdtd_brain_simulation_engine.py`, `cli.py`, `workflows/`, `simulation_config.py` |
| Physics / numerics | `core/` (→ `hermes_drh.simulation`) |
| Settings | `config/` (→ `hermes_drh.settings`) |
| Optimization | `antenna_optimization.py`, `load_optimized_config.py` |
| Post-processing & exports | `data_analysis_validation.py`, `performance_logging.py`, `progress.py` |
| Visualization (simulation outputs) | `multiview_visualization.py`, `build_animations_from_streamed_frames.py` |
| Segmentation (optional extra) | `brain_tumor_segmentation_model.py`, `segmentation_loader.py` |
| Dashboard (optional extra) | `fdtd_dashboard/` |
| Public facades (deprecate later) | `fdtd_solver.py`, `sar_computation.py`, `thermal_solver.py`, `voxel_model.py`, `sources.py` |
| Example configs | `configs/simulation_example.yaml`, `configs/thesis_paper_*.yaml` (runtime YAML only—no LaTeX) |
| Tests | `tests/` (engine, FDTD validation, CLI; not paper-bundle tests) |

### Out of scope — keep in thesis repo only (do not ship on PyPI)

| Script | Reason |
|--------|--------|
| `run_paper_bundle.py` | Orchestrates thesis figure bundle + manifest + `PAPER/` sync |
| `generate_paper_results_section.py` | Writes `paper_results_section.tex` and LaTeX table fragments |
| `plot_scalability_benchmark.py` | Paper scalability figures (invoked by paper bundle) |
| `plot_pipeline_phases_waterfall.py` | Paper waterfall figure → `PAPER/figures/` |
| `plot_houle_style_plane_wave_validation.py` | Paper validation figure → `PAPER/figures/` |
| `configs/paper_results_*.yaml` | Paper-bundle-only run profiles (optional: keep in git, exclude from wheel via `package-data` filter) |

Breast/cervix engines (`fdtd_breast_simulation_engine.py`, `build_breast_animations_from_streamed_frames.py`, …) — decide for v1: include as `hermes_drh.anatomy.*` or defer.

Thesis reproduction: `cd CODE && python run_paper_bundle.py ...` remains valid **from the cloned thesis repo**, importing the installed `hermes-drh` engine via console scripts—not by packaging the paper scripts themselves.

---

## Phase 0 — Decisions (blockers)

- [/] **Confirm PyPI name availability** — search [pypi.org](https://pypi.org) for `hermes-drh`, `hermes-drh-fdtd`, etc.; reserve name on TestPyPI first.
- [ ] **License (preferred: MIT)** — add `LICENSE` at repo root using the **MIT License** (simple, PyPI-friendly, common for academic OSS). Before publishing, confirm with your university/thesis IP office that MIT is acceptable for code derived from the thesis. If MIT is not allowed, fallback is **BSD-3-Clause** (similar permissiveness). Set `license = { text = "MIT" }` in `pyproject.toml` and include `License :: OSI Approved :: MIT License` classifier.
- [ ] **Supported Python versions** — align packaging with how the project is actually run today:
  - **Reference / CI Python:** **3.11** (matches `.devcontainer/devcontainer.json`: `python:1-3.11-bookworm`; use this for thesis reproducibility and release CI).
  - **`requires-python` in `pyproject.toml`:** `>=3.11,<3.14` (supports 3.11, 3.12, 3.13; excludes 3.10 and 3.14+ until tested).
  - **Classifiers:** `Programming Language :: Python :: 3.11`, `3.12`, `3.13`.
  - **Local dev:** If your Mac venv is 3.13, it remains supported; for strict thesis parity, develop/release-test on **3.11** (e.g. `pyenv local 3.11` or the devcontainer).
- [ ] **Install surfaces (recommended extras)** — keep the default install small; users opt in to GPU/UI stacks:

  | Extra | Install command | Purpose |
  |-------|-----------------|--------|
  | *(none)* | `pip install hermes-drh` | **Core:** FDTD, SAR, thermal, voxel materials, optimization, NIfTI/PNG export, `hermes-simulate`, `hermes-build-animations`. No PyTorch, no Streamlit. |
  | `segmentation` | `pip install hermes-drh[segmentation]` | BraTS 3D U-Net path: `torch`, `monai`, loaders. User supplies or downloads `best_model.pth` via `--checkpoint`. |
  | `dashboard` | `pip install hermes-drh[dashboard]` | Streamlit HERMES Dashboard (`hermes-dashboard`). |
  | `all` | `pip install hermes-drh[all]` | Convenience meta-extra: `segmentation` + `dashboard` (typical full-workflow install). |
  | `dev` | `pip install hermes-drh[dev]` | Contributors: `pytest`, `pytest-cov`, `ruff`, `black`, `build`, `twine`, `mypy` (optional). |

  **Suggested dependency pins (ranges, not frozen):**

  ```toml
  [project]
  requires-python = ">=3.11,<3.14"
  dependencies = [
    "numpy>=1.26,<3",
    "scipy>=1.11",
    "nibabel>=5.2",
    "matplotlib>=3.8",
    "PyYAML>=6.0",
    "tqdm>=4.66",
  ]

  [project.optional-dependencies]
  segmentation = [
    "torch>=2.2,<2.11",
    "monai>=1.3",
  ]
  dashboard = [
    "streamlit>=1.30",
  ]
  # Meta-extra: repeat segmentation + dashboard deps (avoid self-referential extras)
  all = [
    "torch>=2.2,<2.11",
    "monai>=1.3",
    "streamlit>=1.30",
  ]
  dev = [
    "pytest>=8",
    "pytest-cov>=4",
    "ruff>=0.4",
    "black>=24",
    "build>=1.2",
    "twine>=5",
  ]
  ```

  **Recommended installs for users:**

  - FDTD from existing segmentation NIfTI only: `pip install hermes-drh`
  - MRI modalities → segment → simulate: `pip install hermes-drh[segmentation]`
  - GUI workflow: `pip install hermes-drh[all]` (needs **ffmpeg** on PATH for MP4 export)

- [ ] **Model weights policy (recommended)** — do **not** bundle `best_model.pth` in the wheel. **v1 approach:** document `--checkpoint /path/to/best_model.pth` and ship a future console helper `hermes-download-weights` under the `segmentation` extra (Zenodo/Hugging Face URL in docs). Default engine behavior when segmentation is requested but no checkpoint is set: clear error with download instructions, not a silent fallback.

---

## Phase 1 — Repository layout (`src/` layout)

Current state: flat `CODE/` tree, `pytest.ini` uses `pythonpath = .`, scripts import `core`, `config`, `simulation_config` from CWD.

- [ ] Create standard layout:

  ```text
  CODE/
    pyproject.toml
    README.md
    LICENSE
    src/
      hermes_drh/
        __init__.py          # __version__
        py.typed             # PEP 561 (optional)
        cli/                 # argparse + engine entry
        simulation/          # was core/ (fdtd, metrics, materials)
        settings/            # was config/ (tissues, anatomy, viz)
        workflows/
        optimization/        # antenna_optimization, load_optimized_config
        io/                  # data_analysis_validation, performance_logging
        segmentation/        # brain model, segmentation_loader
        visualization/       # multiview_visualization, animation builder
        dashboard/           # optional extra: streamlit app
    tests/
    configs/                 # package-data: example + thesis_paper_*.yaml (not paper_results_*)
    # Thesis-only (not in src/):
    run_paper_bundle.py
    generate_paper_results_section.py
    plot_*.py                # paper figure scripts listed above
  ```

- [ ] **Rename `core/` → `hermes_drh.simulation`** — `core` is not a safe public package name on PyPI.
- [ ] **Rename `config/` → `hermes_drh.settings`**
- [ ] Move **in-scope** top-level modules into packages:

  | Current | Target |
  |---------|--------|
  | `fdtd_brain_simulation_engine.py` | `hermes_drh.cli.main` (+ console script) |
  | `cli.py` | `hermes_drh.cli.parser` |
  | `simulation_config.py` | `hermes_drh.settings.simulation` |
  | `antenna_optimization.py` | `hermes_drh.optimization.antenna` |
  | `data_analysis_validation.py` | `hermes_drh.io.validation` |
  | `build_animations_from_streamed_frames.py` | `hermes_drh.visualization.animations` |
  | `multiview_visualization.py` | `hermes_drh.visualization.multiview` |
  | `fdtd_dashboard/streamlit_app.py` | `hermes_drh.dashboard.app` (extra) |
  | Facades: `fdtd_solver.py`, … | `hermes_drh.compat` or remove after one release |

- [ ] **Do not move** `run_paper_bundle.py`, `generate_paper_results_section.py`, or `plot_{scalability,pipeline_phases_waterfall,houle_style_*}.py` into `src/hermes_drh/`.
- [ ] Keep **`BC_MRI_SEG/`**, **`dataset/`**, **`results/`**, **`paper_bundle_runs/`**, **`../PAPER/`** out of the wheel.
- [ ] **`package-data`**: include `configs/simulation_example.yaml` and user-facing `configs/thesis_paper_*.yaml`; **exclude** `configs/paper_results_*.yaml` from the wheel (thesis bundle only).

---

## Phase 2 — `pyproject.toml` (PEP 621)

- [ ] Add **`pyproject.toml`** with `[build-system]` (`setuptools` or `hatchling`).
- [ ] Metadata: `name = "hermes-drh"`, description, readme, license, authors, classifiers, `requires-python`, URLs.
- [ ] **Core dependencies** (from engine imports, not full `requirements.txt` freeze):
  - [ ] `numpy`, `scipy`, `nibabel`, `matplotlib`, `PyYAML`, `tqdm`
- [ ] **Optional dependencies**:
  - [ ] `segmentation = ["torch", "monai", ...]`
  - [ ] `dashboard = ["streamlit", ...]`
  - [ ] `dev = ["pytest", "pytest-cov", "ruff", "build", "twine", ...]`
- [ ] **`[project.scripts]`** — simulation-only entry points:

  ```toml
  [project.scripts]
  hermes-simulate = "hermes_drh.cli.main:main"
  hermes-build-animations = "hermes_drh.visualization.animations:main"
  hermes-dashboard = "hermes_drh.dashboard.app:main"
  ```

  No `hermes-paper-bundle` or LaTeX-related scripts.

- [ ] **`[tool.setuptools.package-data]`**:

  ```toml
  "hermes_drh" = ["configs/simulation_example.yaml", "configs/thesis_paper_*.yaml", "py.typed"]
  ```

- [ ] Split **`requirements.txt`**: published deps in `pyproject.toml`; optional `requirements-thesis.txt` in repo root for `run_paper_bundle` + matplotlib extras used only by paper scripts.

---

## Phase 3 — Import and path refactor (largest effort)

- [ ] Replace flat imports with `hermes_drh.*` throughout **in-scope** modules and tests.
- [ ] Remove **CWD / `CODE_DIR` assumptions** in packaged code:
  - [ ] Engine default checkpoint → `HERMES_CHECKPOINT` or download helper, not `best_model.pth` beside source.
  - [ ] Config loading → `importlib.resources.files("hermes_drh")` for bundled YAML.
  - [ ] **`run_paper_bundle.py`** (thesis-only): may keep `subprocess` + `cwd=CODE` calling `hermes-simulate` after install, or call `hermes_drh` APIs directly—outside package refactor.
- [ ] Update **`pytest.ini`** → `[tool.pytest.ini_options]` with `pythonpath = ["src"]`; exclude paper-only scripts from coverage targets.
- [ ] Add **`tests/test_packaging_smoke.py`**: `import hermes_drh`, `hermes-simulate --help`, bundled example config exists; assert `generate_paper_results_section` is **not** importable from installed package.

---

## Phase 4 — Runtime data directories (not in the package)

- [ ] Env vars: `HERMES_RESULTS_DIR`, `HERMES_DATA_DIR`, `HERMES_CHECKPOINT`.
- [ ] Document that **`results/`**, **`dataset/`** are user/runtime dirs, never installed into `site-packages`.
- [ ] `paper_bundle_runs/` is thesis-repro only (used by `run_paper_bundle.py`, not documented as part of the PyPI product).

---

## Phase 5 — Dashboard (Streamlit, optional extra)

- [ ] Move `fdtd_dashboard/` → `src/hermes_drh/dashboard/`.
- [ ] `hermes-dashboard` entry point; `streamlit` only in `[dashboard]` extra.
- [ ] Dashboard invokes **`hermes-simulate`** (installed script) instead of `python fdtd_brain_simulation_engine.py` with hardcoded `CODE` paths.

---

## Phase 6 — Segmentation / PyTorch (optional extra)

- [ ] Lazy-import torch in segmentation code so `pip install hermes-drh` works without GPU stack.
- [ ] `hermes-download-weights` or documented manual `--checkpoint` path.
- [ ] Do not bundle `best_model.pth` in the wheel.

---

## Phase 7 — Quality, CI, and release engineering

- [ ] `ruff` / `black` / optional `mypy` in `pyproject.toml`.
- [ ] CI: `pytest` (exclude tests that require paper bundle), `python -m build`, `twine check`.
- [ ] TestPyPI publish on tag; SemVer in `hermes_drh/__version__`.
- [ ] `CHANGELOG.md`: note that paper/LaTeX workflow remains in thesis repo, not in `hermes-drh`.

---

## Phase 8 — Documentation and PyPI page

- [ ] **README.md**: `pip install hermes-drh[segmentation,dashboard]`; quickstart `hermes-simulate --modalities-dir ...`; `hermes-build-animations` for MP4s from streamed frames.
- [ ] Migrate **`docs/FDTD_BRAIN_SIMULATION_DOCUMENTATION.md`** to user docs (engine + animations only).
- [ ] **Separate short doc** in thesis repo: “Reproducing paper figures” → points to `run_paper_bundle.py` (not installed from PyPI).

---

## Phase 9 — Thesis repo integration (outside the package)

- [ ] **`run_paper_bundle.py`** continues to live under `CODE/`; depends on installed `hermes-simulate` (or editable install) for simulations.
- [ ] **`generate_paper_results_section.py`** and **`plot_*.py`** remain thesis-only; write into `PAPER/generated/` and `PAPER/figures/`.
- [ ] `PAPER/` reproducibility README: `pip install hermes-drh` + `python run_paper_bundle.py` (second step not from PyPI).
- [ ] Defense `script.md` PyPI claims refer to **engine + dashboard**, not LaTeX automation.

---

## Phase 10 — Publish checklist

- [ ] `python -m build` → wheel + sdist; `twine check dist/*`
- [ ] TestPyPI install; smoke: `hermes-simulate --help`, tiny `simulation_example.yaml` run
- [ ] Verify wheel **does not** contain: `run_paper_bundle.py`, `generate_paper_results_section.py`, `plot_scalability_benchmark.py`, `plot_pipeline_phases_waterfall.py`, `plot_houle_style_plane_wave_validation.py`
- [ ] Production PyPI upload + GitHub release

---

## Phase 11 — Post-release shims (in-scope scripts only)

- [ ] Thin deprecation shims for **`fdtd_brain_simulation_engine.py`** and **`build_animations_from_streamed_frames.py`** pointing to console scripts.
- [ ] **No shims** for paper/LaTeX scripts (they stay as plain thesis scripts).
- [ ] Move `practice*.py` drills to `tests/drills/` or exclude from wheel.

---

## Suggested implementation order

1. Phase 0 (scope lock: no LaTeX in wheel)  
2. Phase 1 + 2 (`src/hermes_drh`, `pyproject.toml`, simulation entry points only)  
3. Phase 3 imports + tests  
4. Phase 4 env / results dirs  
5. Extras: segmentation, dashboard  
6. Phase 7–8 CI + docs  
7. Phase 10 release  
8. Phase 9 thesis repo wiring (`run_paper_bundle` calls installed `hermes-simulate`)

---

## Known risks / gotchas

| Issue | Mitigation |
|-------|------------|
| Generic name `core` | Rename to `hermes_drh.simulation` |
| Frozen `requirements.txt` | Core deps in `pyproject.toml` only |
| `run_paper_bundle` subprocess `cwd=CODE` | Thesis script; call `hermes-simulate` on PATH |
| `fdtd==0.3.5` dependency | Verify no import clash with `hermes_drh` |
| Paper configs `paper_results_*.yaml` | Exclude from package-data; keep in git for thesis |
| Streamlit + **ffmpeg** for animations | Document system deps; `hermes-build-animations` |

---

## Acceptance criteria

- [ ] `pip install hermes-drh` works in a fresh venv (macOS ARM64).
- [ ] `hermes-simulate --config ...` runs using bundled example YAML.
- [ ] `hermes-build-animations <results_dir>` produces MP4s from streamed frames.
- [ ] Installed package has **no** LaTeX writers and **no** `run_paper_bundle` module.
- [ ] `pytest` passes for engine tests when installed editable from `src/`.
- [ ] Wheel &lt;50 MB without segmentation weights.
- [ ] TestPyPI verified before production upload.
