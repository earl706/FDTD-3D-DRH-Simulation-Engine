"""
Performance Evaluation (thesis: Performance Evaluation).

Peak memory reporting and optional helpers for performance metrics and scalability JSON.
"""

import platform

try:
    import resource
except ImportError:
    resource = None


def get_peak_memory_mb():
    """Return peak resident set size in MB, or None if unavailable (e.g. on Windows)."""
    if resource is None:
        return None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss: Linux = KB, macOS = bytes (see getrusage(2))
        if platform.system() == "Darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0  # Linux: KB -> MB
    except Exception:
        return None
