
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("NL2SQL")

_HERE        = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_BUILD_SCRIPT = _PROJECT_ROOT / "scripts" / "build_master_cache.py"
_CHECK_INTERVAL_SECS = 6 * 60 * 60   # check staleness every 6 hours


def _is_cache_stale() -> bool:
    """Returns True if the cache file is missing or past its expiry."""
    from core.response_translator import _cache_expires_at
    if _cache_expires_at is None:
        return True
    return datetime.utcnow() > _cache_expires_at


def _rebuild_cache() -> bool:
    """
    Runs build_master_cache.py as a subprocess.
    After success, hot-reloads the new cache into memory without restarting.
    Returns True on success.
    """
    log.info("[scheduler] Translation cache is stale — rebuilding …")
    try:
        result = subprocess.run(
            [sys.executable, str(_BUILD_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=300,   # 5-minute hard limit
        )
        if result.returncode == 0:
            log.info("[scheduler] build_master_cache.py completed OK.")
            # Hot-reload into memory
            from core.response_translator import load_translation_cache
            load_translation_cache()
            return True
        else:
            log.error(
                "[scheduler] build_master_cache.py failed (rc=%d):\n%s",
                result.returncode, result.stderr[:1000],
            )
            return False
    except subprocess.TimeoutExpired:
        log.error("[scheduler] build_master_cache.py timed out after 5 min.")
        return False
    except Exception as exc:
        log.error("[scheduler] Unexpected error running build_master_cache.py: %s", exc)
        return False


def _scheduler_loop() -> None:
    """Main loop — runs forever in background thread."""
    log.info(
        "[scheduler] Translation cache refresh scheduler started "
        "(check interval: %dh).",
        _CHECK_INTERVAL_SECS // 3600,
    )
    while True:
        try:
            if _is_cache_stale():
                _rebuild_cache()
            else:
                from core.response_translator import _cache_expires_at
                if _cache_expires_at:
                    remaining = (_cache_expires_at - datetime.utcnow()).days
                    log.info(
                        "[scheduler] Translation cache is fresh — expires in %d day(s).",
                        remaining,
                    )
        except Exception as exc:
            log.error("[scheduler] Unhandled error in scheduler loop: %s", exc)

        time.sleep(_CHECK_INTERVAL_SECS)


def start_cache_refresh_scheduler() -> threading.Thread:
    """
    Start the background refresh scheduler.
    Safe to call multiple times — only starts one thread.
    Returns the thread object.
    """
    t = threading.Thread(
        target=_scheduler_loop,
        name="translation-cache-refresher",
        daemon=True,
    )
    t.start()
    log.info("[scheduler] Translation cache refresh thread started (daemon).")
    return t
