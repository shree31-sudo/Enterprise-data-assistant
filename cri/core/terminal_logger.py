

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

# ── ANSI colour palette ───────────────────────────────────────────────────────
_NO_COLOR: bool = bool(os.getenv("NO_COLOR", ""))

_C: dict[str, str] = {
    "reset":   "" if _NO_COLOR else "\033[0m",
    "bold":    "" if _NO_COLOR else "\033[1m",
    "dim":     "" if _NO_COLOR else "\033[2m",
    "green":   "" if _NO_COLOR else "\033[92m",
    "cyan":    "" if _NO_COLOR else "\033[96m",
    "yellow":  "" if _NO_COLOR else "\033[93m",
    "red":     "" if _NO_COLOR else "\033[91m",
    "blue":    "" if _NO_COLOR else "\033[94m",
    "grey":    "" if _NO_COLOR else "\033[90m",
    "white":   "" if _NO_COLOR else "\033[97m",
    "magenta": "" if _NO_COLOR else "\033[95m",
}

# Level → colour mapping
_LEVEL_COLOR: dict[int, str] = {
    logging.DEBUG:    "blue",
    logging.INFO:     "green",
    logging.WARNING:  "yellow",
    logging.ERROR:    "red",
    logging.CRITICAL: "red",
}

# Level → short badge text
_LEVEL_BADGE: dict[int, str] = {
    logging.DEBUG:    "DBG",
    logging.INFO:     "INF",
    logging.WARNING:  "WRN",
    logging.ERROR:    "ERR",
    logging.CRITICAL: "CRT",
}

# Standard LogRecord attributes — stripped from the extras dict
_STANDARD_ATTRS: frozenset = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "taskName",
    "message", "asctime",
})


# ── Formatter ─────────────────────────────────────────────────────────────────

class _TerminalFormatter(logging.Formatter):
    """
    Compact, coloured single-line terminal format.

    Normal log lines:
        10:42:07  [INF]  [NL2SQL]  query received  key=val

    _vprint()-sourced lines (extra['vprint'] == True):
        10:42:07  [REQUEST]   magenta badge  message  detail
        10:42:07  [BM25]      magenta badge  scoring result
        10:42:07  [PIPELINE]  magenta badge  pipeline start

    For ERROR / CRITICAL the full traceback is appended on subsequent lines.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        ts = f"{_C['dim']}{datetime.fromtimestamp(record.created).strftime('%H:%M:%S')}{_C['reset']}"

        # Level badge (for normal lines)
        colour = _LEVEL_COLOR.get(record.levelno, "white")
        badge_text = _LEVEL_BADGE.get(record.levelno, record.levelname[:3])
        badge = f"{_C[colour]}{_C['bold']}[{badge_text}]{_C['reset']}"

        # Logger name
        logger_name = record.name.split(".")[-1]
        name_part = f"{_C['cyan']}[{logger_name}]{_C['reset']}"

        # Message
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            msg = f"{_C[colour]}{msg}{_C['reset']}"
        else:
            msg = f"{_C['white']}{msg}{_C['reset']}"

        # Collect extra fields beyond standard LogRecord keys
        extras: dict = {
            k: v for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        }

        # Pop _vprint()-injected keys before rendering extras
        # chatbot_backend._vprint() sets: vprint=True, vlabel=LABEL, vdetail=detail
        is_vprint = extras.pop("vprint", False)
        vlabel    = extras.pop("vlabel", "")
        vdetail   = extras.pop("vdetail", "")

        # Remaining extras as trailing key=value pairs
        extra_str = ""
        if extras:
            parts = [f"{k}={v}" for k, v in extras.items()]
            extra_str = f"  {_C['grey']}{' | '.join(parts)}{_C['reset']}"

        if is_vprint and vlabel:
            # _vprint line: magenta badge, no level badge, no logger name
            # This makes REQUEST / BM25 / PIPELINE events visually distinct
            vbadge     = f"{_C['magenta']}{_C['bold']}[{vlabel}]{_C['reset']}"
            detail_str = f"  {_C['grey']}{vdetail}{_C['reset']}" if vdetail else ""
            line = f"{ts}  {vbadge}  {msg}{detail_str}{extra_str}"
        else:
            # Normal structured log line
            line = f"{ts}  {badge}  {name_part}  {msg}{extra_str}"

        # Traceback for errors
        if record.exc_info:
            tb = self.formatException(record.exc_info)
            line += f"\n{_C['grey']}{tb}{_C['reset']}"

        return line


# ── Public API ────────────────────────────────────────────────────────────────

_ATTACHED: bool = False   # guard against double-attach


def attach_terminal_handler(level: int | None = None) -> None:
    """
    Add a coloured StreamHandler to the root logger.

    Parameters
    ----------
    level
        Logging level for the terminal handler.  Defaults to the value of
        TERMINAL_LOG_LEVEL env var, then LOG_LEVEL, then INFO.
        Pass logging.DEBUG / logging.INFO etc. to override.
    """
    global _ATTACHED
    if _ATTACHED:
        return  # idempotent — safe to call multiple times

    if level is None:
        raw = (
            os.getenv("TERMINAL_LOG_LEVEL")
            or os.getenv("LOG_LEVEL", "INFO")
        ).strip().upper()
        level = getattr(logging, raw, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_TerminalFormatter())
    handler.setLevel(level)

    root = logging.getLogger()
    root.addHandler(handler)

    _ATTACHED = True

    # Emit a confirmation line visible in both file and terminal
    logging.getLogger("NL2SQL").info(
        "Terminal logging active",
        extra={"level": logging.getLevelName(level), "no_color": _NO_COLOR},
    )
