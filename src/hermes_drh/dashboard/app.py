"""Console entry: ``hermes-dashboard`` → Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError as e:
        raise SystemExit(
            "Streamlit is required for the dashboard. Install with: pip install 'hermes-drh[dashboard]'"
        ) from e

    app = Path(__file__).resolve().parent / "streamlit_app.py"
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
