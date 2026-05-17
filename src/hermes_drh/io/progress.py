"""
Progress reporting for the simulation dashboard.

Single responsibility: write progress JSON (phase, message, percent, phases_done)
so the dashboard can show live status. Used by the engine and data_analysis_validation.
"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def write_progress(
    phase,
    message,
    percent,
    progress_dir,
    progress_file,
    phases_done=None,
    extra=None,
):
    """Write progress JSON for dashboard. phases_done = list of completed phase names."""
    try:
        os.makedirs(progress_dir, exist_ok=True)
        payload = {
            "phase": phase,
            "message": message,
            "percent": min(100, max(0, percent)),
            "phases_done": list(phases_done) if phases_done else [],
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            payload.update(extra)
        with open(progress_file, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
    except Exception:
        logger.warning(
            "Could not write progress JSON (phase=%r, path=%s)",
            phase,
            progress_file,
            exc_info=True,
        )
