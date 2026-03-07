"""
Local runner for FDTD simulation.

Listens on localhost for run configs sent from the Streamlit app (when user
chooses "Run on my machine"). When POST /run is received, builds argv for
fdtd_brain_simulation_engine.py and runs it in a subprocess so the simulation
runs on the user's machine, not on Streamlit's server.

Usage:
  From repo root (CODE/):
    python fdtd_dashboard/local_runner.py [--port 8765] [--repo-root .]

  Then in the deployed Streamlit app: choose "Run on my machine", download the
  run package ZIP, extract it, and use the Run page (run_local.html) to send
  the config to this runner. Set run_package_dir in that page to the extracted
  folder path.
"""

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _repo_root(script_dir):
    """Repo root = parent of fdtd_dashboard."""
    return script_dir.parent


def _build_argv_from_config(repo_root_path, run_package_dir, config):
    """Build argv for fdtd_brain_simulation_engine.py from a config dict.
    Paths in config (seg_path, modalities_dir) are relative to run_package_dir.
    """
    repo_root_path = Path(repo_root_path)
    run_package_dir = Path(run_package_dir)
    engine_path = repo_root_path / "fdtd_brain_simulation_engine.py"
    if not engine_path.exists():
        return None, f"Engine not found: {engine_path}"

    argv = [sys.executable, "-u", str(engine_path)]

    input_mode = config.get("input_mode", "modalities_dir")
    seg_path_rel = config.get("seg_path")
    modalities_dir_rel = config.get("modalities_dir")

    if input_mode == "seg" and seg_path_rel:
        seg_path = run_package_dir / seg_path_rel
        if not seg_path.exists():
            return None, f"Segmentation file not found: {seg_path}"
        argv.append(str(seg_path))
    elif modalities_dir_rel:
        modalities_dir = run_package_dir / modalities_dir_rel
        if not modalities_dir.is_dir():
            return None, f"Modalities dir not found: {modalities_dir}"
        argv.extend(["--modalities-dir", str(modalities_dir)])
        checkpoint = config.get("checkpoint")
        if checkpoint:
            argv.extend(["--checkpoint", str(checkpoint)])
        if config.get("no_normal_brain"):
            argv.append("--no-normal-brain")
    else:
        return None, "Config must have seg_path or modalities_dir"

    time_steps = config.get("time_steps", 500)
    max_dim = config.get("max_dim", 120)
    argv.extend(["--time-steps", str(time_steps)])
    argv.extend(["--max-dim", str(max_dim)])
    argv.append("--stream-frames")

    if config.get("optimize_antenna"):
        argv.append("--optimize-antenna")
        argv.extend(["--f0", str(config.get("f0_hz", 100e6))])
        argv.extend(["--opt-time-steps", str(config.get("opt_time_steps", 700))])
        argv.extend(["--opt-phase-steps", str(config.get("opt_phase_steps", 24))])
        argv.extend(["--opt-amp-steps", str(config.get("opt_amp_steps", 9))])
        argv.extend(["--opt-amp-min", str(config.get("opt_amp_min", 0.2))])
        argv.extend(["--opt-amp-max", str(config.get("opt_amp_max", 2.5))])
        argv.extend(["--opt-refine-iters", str(config.get("opt_refine_iters", 8))])
        argv.extend(["--opt-multi-start", str(config.get("opt_multi_start", 3))])
        argv.extend(
            ["--opt-penalty-weight", str(config.get("opt_penalty_weight", 0.1))]
        )
        opt_freq_sweep = config.get("opt_freq_sweep")
        if opt_freq_sweep:
            argv.append("--opt-freq-sweep")
            argv.extend([str(f) for f in opt_freq_sweep])
        opt_geom_offsets = config.get("opt_geom_offsets")
        if opt_geom_offsets:
            argv.append("--opt-geom-offsets")
            argv.extend([str(o) for o in opt_geom_offsets])
        opt_geom_zplanes = config.get("opt_geom_zplanes")
        if opt_geom_zplanes:
            argv.append("--opt-geom-zplanes")
            argv.extend([str(z) for z in opt_geom_zplanes])
    else:
        argv.extend(["--pulse-amplitude", str(config.get("pulse_amplitude", 100.0))])

    return argv, None


class LocalRunnerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>FDTD Local Runner</h1>"
                b"<p>Runner is running. Use the Run page from the Streamlit app and "
                b"click Run to start a simulation on this machine.</p>"
                b"<p>POST /run with JSON body: run_package_dir + engine config.</p></body></html>"
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/run":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            run_package_dir = data.get("run_package_dir")
            if not run_package_dir or not run_package_dir.strip():
                self._send_json(400, {"error": "run_package_dir is required"})
                return

            run_package_dir = run_package_dir.strip()
            if not Path(run_package_dir).is_dir():
                self._send_json(
                    400,
                    {"error": f"run_package_dir is not a directory: {run_package_dir}"},
                )
                return

            repo_root_path = self.server.repo_root
            config = {k: v for k, v in data.items() if k != "run_package_dir"}
            argv, err = _build_argv_from_config(repo_root_path, run_package_dir, config)
            if err:
                self._send_json(400, {"error": err})
                return

            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(repo_root_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                self._send_json(200, {"status": "started", "pid": proc.pid})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_repo_root = _repo_root(script_dir)

    ap = argparse.ArgumentParser(
        description="Local runner for FDTD (receives config from Streamlit app)."
    )
    ap.add_argument(
        "--port", type=int, default=8765, help="Port to listen on (default: 8765)"
    )
    ap.add_argument(
        "--repo-root",
        type=str,
        default=str(default_repo_root),
        help="Repo root (engine directory)",
    )
    args = ap.parse_args()

    repo_root_path = Path(args.repo_root)
    if not (repo_root_path / "fdtd_brain_simulation_engine.py").exists():
        print(
            f"Error: fdtd_brain_simulation_engine.py not found under {repo_root_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    server = HTTPServer(("127.0.0.1", args.port), LocalRunnerHandler)
    server.repo_root = repo_root_path
    print(f"Local runner listening on http://127.0.0.1:{args.port}")
    print(
        "Use the Streamlit app (Run on my machine) and the Run page to start simulations."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
