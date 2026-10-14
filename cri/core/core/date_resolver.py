from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz

log = logging.getLogger("DateResolver")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimeClause:
    """
    Represents a time-of-day filter that is kept *separate* from the date
    range so that month / year / week ranges are never mutated.

    qualifier  : "after" | "before" | "at" | "between" | "morning" |
                 "afternoon" | "evening" | "night" | "rolling_minutes" |
                 "rolling_hours"
    hour_start : 24-h start hour  (inclusive lower bound)
    minute_start: minute for start
    hour_end   : 24-h end   hour  (exclusive upper bound for "between")
    minute_end : minute for end
    rolling_n  : number of minutes/hours for rolling window qualifiers
    """
    qualifier:     str
    hour_start:    int   = 0
    minute_start:  int   = 0
    hour_end:      int   = 0
    minute_end:    int   = 0
    rolling_n:     int   = 0

    def sql_fragment(self, table: str, col: str) -> str:
        """
        Build a generic SQL time-of-day fragment using:
          CAST(col AS TIME)
          DATEPART(HOUR, col)
          DATENAME(WEEKDAY, col)          ← available if needed by callers
        """
        tc = f"CAST({table}.{col} AS TIME)"
        h_expr = f"DATEPART(HOUR, {table}.{col})"

        if self.qualifier == "after":
            t = f"'{self.hour_start:02d}:{self.minute_start:02d}:00'"
            return f"{tc} >= {t}"

        if self.qualifier == "before":
            t = f"'{self.hour_start:02d}:{self.minute_start:02d}:00'"
            return f"{tc} < {t}"

        if self.qualifier == "at":
            t = f"'{self.hour_start:02d}:{self.minute_start:02d}:00'"
            t1 = f"'{self.hour_end:02d}:{self.minute_end:02d}:00'"
            return f"{tc} >= {t} AND {tc} < {t1}"

        if self.qualifier == "between":
            t_lo = f"'{self.hour_start:02d}:{self.minute_start:02d}:00'"
            t_hi = f"'{self.hour_end:02d}:{self.minute_end:02d}:00'"
            return f"{tc} >= {t_lo} AND {tc} < {t_hi}"

        if self.qualifier == "morning":
            return f"{h_expr} >= 5 AND {h_expr} < 12"

        if self.qualifier == "afternoon":
            return f"{h_expr} >= 12 AND {h_expr} < 17"

        if self.qualifier == "evening":
            return f"{h_expr} >= 17 AND {h_expr} < 21"

        if self.qualifier == "night":
            return f"({h_expr} >= 21 OR {h_expr} < 5)"

        if self.qualifier == "rolling_minutes":
            # DATEADD-free: caller passes the window as a literal range via
            # DateRange; this fragment is a fallback label only — real rolling
            # ranges come through DateRange.sql_fragment.
            t = f"'{self.hour_start:02d}:{self.minute_start:02d}:00'"
            return f"{tc} >= {t}"

        if self.qualifier == "rolling_hours":
            t = f"'{self.hour_start:02d}:{self.minute_start:02d}:00'"
            return f"{tc} >= {t}"

        return ""


@dataclass
class DateRange:
    """Half-open interval [start, end) — both timezone-aware datetimes."""
    start:      datetime
    end:        datetime
    grain:      str          # "minute"|"hour"|"day"|"week"|"month"|"quarter"|"year"|"range"|"custom"
    time_clause: Optional[TimeClause] = field(default=None)

    def sql_fragment(self, table: str, col: str) -> str:
        s = self.start.strftime("%Y-%m-%d %H:%M:%S")
        e = self.end.strftime("%Y-%m-%d %H:%M:%S")
        return f"{table}.{col} >= '{s}' AND {table}.{col} < '{e}'"

    def full_sql_fragment(self, table: str, col: str) -> str:
        """Date range + optional time clause combined."""
        base = self.sql_fragment(table, col)
        if self.time_clause:
            tc_sql = self.time_clause.sql_fragment(table, col)
            if tc_sql:
                return f"{base} AND {tc_sql}"
        return base


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_NAMES: Dict[str, int] = {
    "january": 1,  "jan": 1,
    "february": 2, "feb": 2,
    "march": 3,    "mar": 3,
    "april": 4,    "apr": 4,
    "may": 5,
    "june": 6,     "jun": 6,
    "july": 7,     "jul": 7,
    "august": 8,   "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MONTH_PATTERN = "|".join(sorted(_MONTH_NAMES.keys(), key=len, reverse=True))

_DAY_NAMES: Dict[str, int] = {
    "monday": 0,   "mon": 0,
    "tuesday": 1,  "tue": 1, "tues": 1,
    "wednesday": 2,"wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4,   "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6,   "sun": 6,
}

_DAY_PATTERN = "|".join(sorted(_DAY_NAMES.keys(), key=len, reverse=True))

_TEMPORAL_TRIGGERS = set(_MONTH_NAMES.keys()) | set(_DAY_NAMES.keys()) | {
    "today", "yesterday", "tomorrow",
    "day after tomorrow", "day before yesterday",
    "this week", "last week", "next week",
    "this month", "last month", "next month",
    "this year", "last year", "next year",
    "this quarter", "last quarter", "next quarter",
    "recent", "latest", "recently",
    "ytd", "year to date", "mtd", "month to date",
    "since", "before", "after", "between", "until", "till",
    "ago", "past", "last", "within",
    "q1", "q2", "q3", "q4",
    "week", "month", "year", "day", "hour", "minute",
    "fy", "fiscal", "financial year", "financial", "fiscal year",
    "now", "current", "currently", "real-time", "realtime", "live",
    "right now", "at this moment", "as of now", "ongoing",
    "this hour", "last hour", "last minute", "last few minutes",
    # time-of-day qualifiers
    "am", "pm", "midnight", "noon",
    "morning", "afternoon", "evening", "night",
}

_EFFECTIVE_TRIGGERS = {
    "effective", "valid", "active", "in effect",
    "validity", "currently active", "as of", "live",
}

# ── Fiscal year config ───────────────────────────────────────────────────────
FISCAL_YEAR_START_MONTH: int = int(os.getenv("FISCAL_YEAR_START_MONTH", "4").strip())

# ── "recent/latest" window in days ───────────────────────────────────────────
RECENT_DAYS: int = int(os.getenv("RECENT_DAYS", "30").strip())

# ── "current/now" window in minutes ─────────────────────────────────────────
CURRENT_WINDOW_MINUTES: int = int(os.getenv("CURRENT_WINDOW_MINUTES", "60").strip())

_DATE_FUNC_RE = re.compile(
    r'\b(dateadd|getdate|datediff|convert|datepart|month|year|day)\s*\(',
    re.IGNORECASE
)

# ── n-units-ago ──────────────────────────────────────────────────────────────
_N_AGO_RE = re.compile(
    r'(?:last|past|previous|within|in\s+the\s+last)\s+(\d+)\s+'
    r'(day|week|month|year|hour|minute)s?',
    re.IGNORECASE
)

_N_AGO_BARE_RE = re.compile(
    r'\b(?:last|past|previous)\s+(hour|minute|day|week|month|year)\b',
    re.IGNORECASE
)

# ── Rolling window: "last N minutes / hours" (pure time, no date) ────────────
_ROLLING_TIME_RE = re.compile(
    r'\b(?:last|past|within)\s+(\d+)\s+(minute|hour)s?\b',
    re.IGNORECASE
)

# ── Exact / bounded time patterns ────────────────────────────────────────────
# Matches: "after 12:30", "before 3pm", "from 9:00 am to 5pm",
#          "at 14:00", "by 5pm", "until 18:30", "since 08:00",
#          "after noon", "before midnight", "between 9am and 5pm"
_TOD_RE = re.compile(
    r'\b(?:after|before|from|since|until|till|at|by|around|during)\s+'
    r'(?:'
        r'(noon|midnight)'
        r'|(?P<hm_h>\d{1,2})[:.](?P<hm_m>\d{2})\s*(?P<hm_mer>am|pm)?'
        r'|(?P<h_only>\d{1,2})\s*(?P<h_mer>am|pm)'
        r'|(?P<h_bare>\d{1,2})(?!\s*(?:am|pm|:\d|\d))(?=\s*(?:$|\b))'  # bare hour e.g. "after 12"
    r')',
    re.IGNORECASE,
)

# "between 9am and 5pm" / "between 09:00 and 17:00"
_TOD_BETWEEN_RE = re.compile(
    r'\bbetween\s+'
    r'(?:(noon|midnight)|(?P<ah>\d{1,2})(?:[:.](?P<am>\d{2}))?\s*(?P<amer>am|pm)?)'
    r'\s+and\s+'
    r'(?:(noon|midnight)|(?P<bh>\d{1,2})(?:[:.](?P<bm>\d{2}))?\s*(?P<bmer>am|pm)?)',
    re.IGNORECASE,
)

# Generic time-of-day words
_TOD_WORDS_RE = re.compile(
    r'\b(morning|afternoon|evening|night)\b',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TIME PARSING  (pure, no date mutation)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_hm(
    named_group: Optional[str],
    h_str: Optional[str],
    m_str: Optional[str],
    mer: str,
) -> Tuple[int, int]:
    """Convert raw regex groups → (hour24, minute)."""
    if named_group:
        lc = named_group.lower()
        return (12, 0) if lc == "noon" else (0, 0)
    h = int(h_str)
    minute = int(m_str) if m_str else 0
    mer = (mer or "").lower()
    if mer == "pm" and h != 12:
        h += 12
    elif mer == "am" and h == 12:
        h = 0
    # Bare hour with no am/pm: treat 1-6 as PM (manufacturing shift hours),
    # 0 and 7-23 as-is (24h). "after 12" → 12:00, "after 3" → 15:00 (shift default).
    elif not mer and m_str is None and 1 <= h <= 6:
        h += 12
    return h, minute


def parse_time_clause(ql: str) -> Optional[TimeClause]:
    """
    Extract a time-of-day qualifier from the lowercased question string.

    Resolution order:
      1. "between <time> and <time>"
      2. after / before / from / since / until / till / at / by / around <time>
         (handles HH:MM, H am/pm, bare hour like "after 12")
      3. morning / afternoon / evening / night
      4. rolling minutes/hours — only when NO date range keyword is present
      5. dateparser fallback — catches anything regex missed

    Returns TimeClause or None.
    """
    # ── 1. between <time> and <time> ─────────────────────────────────────────
    m = _TOD_BETWEEN_RE.search(ql)
    if m:
        h_a, min_a = _parse_hm(m.group(1), m.group("ah"), m.group("am"), m.group("amer") or "")
        h_b, min_b = _parse_hm(m.group(5), m.group("bh"), m.group("bm"), m.group("bmer") or "")
        log.info("[DateResolver] time-clause between %02d:%02d and %02d:%02d", h_a, min_a, h_b, min_b)
        return TimeClause(
            qualifier="between",
            hour_start=h_a, minute_start=min_a,
            hour_end=h_b,   minute_end=min_b,
        )

    # ── 2. exact / bounded time ───────────────────────────────────────────────
    m = _TOD_RE.search(ql)
    if m:
        matched_text   = m.group(0)
        qualifier_word = matched_text.strip().split()[0].lower()

        if qualifier_word in ("after", "from", "since", "around"):
            qual = "after"
        elif qualifier_word in ("before", "until", "till", "by"):
            qual = "before"
        else:
            qual = "at"

        named = m.group(1)  # noon / midnight
        # h_bare: bare number like "after 12" — no am/pm, no minutes
        h_bare = m.group("h_bare") if "h_bare" in m.groupdict() else None
        h_str  = m.group("hm_h") or m.group("h_only") or h_bare
        m_str  = m.group("hm_m") if not h_bare else None
        mer    = (m.group("hm_mer") or m.group("h_mer") or "") if not h_bare else ""

        h, minute = _parse_hm(named, h_str, m_str, mer)

        if qual == "at":
            end_minute = minute + 1
            end_hour   = h + (end_minute // 60)
            end_minute = end_minute % 60
            log.info("[DateResolver] time-clause at %02d:%02d", h, minute)
            return TimeClause(
                qualifier="at",
                hour_start=h, minute_start=minute,
                hour_end=end_hour, minute_end=end_minute,
            )

        log.info("[DateResolver] time-clause %s %02d:%02d", qual, h, minute)
        return TimeClause(qualifier=qual, hour_start=h, minute_start=minute)

    # ── 3. generic time-of-day words ─────────────────────────────────────────
    m = _TOD_WORDS_RE.search(ql)
    if m:
        word = m.group(1).lower()
        log.info("[DateResolver] time-clause generic: %s", word)
        return TimeClause(qualifier=word)

    # ── 4. rolling minutes / hours  (only when not the primary date range) ───
    _DATE_ANCHORS = re.compile(
        r'\b(today|yesterday|tomorrow|this\s+week|last\s+week|this\s+month|'
        r'last\s+month|this\s+year|last\s+year|ytd|mtd|q[1-4]|'
        r'jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|'
        r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|'
        r'oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|'
        r'(19|20)\d{2})\b',
        re.IGNORECASE,
    )
    rm = _ROLLING_TIME_RE.search(ql)
    if rm and not _DATE_ANCHORS.search(ql):
        n    = int(rm.group(1))
        unit = rm.group(2).lower()
        qual = "rolling_minutes" if unit == "minute" else "rolling_hours"
        log.info("[DateResolver] time-clause rolling %d %ss", n, unit)
        return TimeClause(qualifier=qual, rolling_n=n)

    # ── 5. dateparser fallback — catches "after 3pm", "post noon", etc. ──────
    try:
        import dateparser
        # Scan for "after/before/at/from/since/until/by" + anything
        _DP_TRIGGER = re.compile(
            r'\b(after|before|from|since|until|till|at|by|around)\s+([\w:. ]+)',
            re.IGNORECASE,
        )
        dm = _DP_TRIGGER.search(ql)
        if dm:
            qualifier_word = dm.group(1).lower()
            fragment       = dm.group(2).strip().split()[0]  # first token after qualifier
            parsed = dateparser.parse(
                fragment,
                settings={"PREFER_DAY_OF_MONTH": "first", "RETURN_TIME_AS_PERIOD": True},
            )
            if parsed and re.match(r'^\d{1,2}', fragment):
                h, minute = parsed.hour, parsed.minute
                if qualifier_word in ("after", "from", "since", "around"):
                    qual = "after"
                elif qualifier_word in ("before", "until", "till", "by"):
                    qual = "before"
                else:
                    qual = "at"
                log.info("[DateResolver] dateparser fallback time-clause %s %02d:%02d", qual, h, minute)
                return TimeClause(qualifier=qual, hour_start=h, minute_start=minute)
    except Exception as exc:
        log.debug("[DateResolver] dateparser fallback skipped: %s", exc)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATE PARSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_local(dt: datetime, tz: pytz.BaseTzInfo) -> datetime:
    if dt.tzinfo is None:
        dt = tz.localize(dt)
    return dt


_to_utc = _to_local   # legacy alias


def _month_start(year: int, month: int, tz: pytz.BaseTzInfo) -> datetime:
    return _to_local(datetime(year, month, 1, 0, 0, 0), tz)


def _month_end(year: int, month: int, tz: pytz.BaseTzInfo) -> datetime:
    if month == 12:
        return _to_local(datetime(year + 1, 1, 1, 0, 0, 0), tz)
    return _to_local(datetime(year, month + 1, 1, 0, 0, 0), tz)


def _year_start(year: int, tz: pytz.BaseTzInfo) -> datetime:
    return _to_local(datetime(year, 1, 1, 0, 0, 0), tz)


def _year_end(year: int, tz: pytz.BaseTzInfo) -> datetime:
    return _to_local(datetime(year + 1, 1, 1, 0, 0, 0), tz)


def _quarter_bounds(year: int, q: int, tz: pytz.BaseTzInfo) -> Tuple[datetime, datetime]:
    start_month = (q - 1) * 3 + 1
    end_month   = start_month + 3
    s = _to_local(datetime(year, start_month, 1), tz)
    e = _year_end(year, tz) if end_month > 12 else _to_local(datetime(year, end_month, 1), tz)
    return s, e


def _day_range(dt: datetime, tz: pytz.BaseTzInfo) -> Tuple[datetime, datetime]:
    s = _to_local(dt.replace(hour=0, minute=0, second=0, microsecond=0), tz)
    return s, s + timedelta(days=1)


def _hour_range(dt: datetime, tz: pytz.BaseTzInfo) -> Tuple[datetime, datetime]:
    s = _to_local(dt.replace(minute=0, second=0, microsecond=0), tz)
    return s, s + timedelta(hours=1)


def _midnight_tomorrow(now: datetime, tz: pytz.BaseTzInfo) -> datetime:
    return _to_local(
        now.replace(hour=0, minute=0, second=0, microsecond=0), tz
    ) + timedelta(days=1)


def _now_truncated(now: datetime, tz: pytz.BaseTzInfo) -> datetime:
    dt = now.replace(microsecond=0)
    if dt.tzinfo is None:
        dt = tz.localize(dt)
    return dt


def _current_quarter(now: datetime) -> int:
    return (now.month - 1) // 3 + 1


def _fiscal_year_bounds(year: int, tz: pytz.BaseTzInfo) -> Tuple[datetime, datetime]:
    start_yr = year - 1 if FISCAL_YEAR_START_MONTH > 1 else year
    s = _to_local(datetime(start_yr, FISCAL_YEAR_START_MONTH, 1), tz)
    end_yr = start_yr + 1
    e = _to_local(datetime(end_yr, FISCAL_YEAR_START_MONTH, 1), tz)
    return s, e


def _current_fiscal_year(now: datetime) -> int:
    if now.month >= FISCAL_YEAR_START_MONTH:
        return now.year + 1
    return now.year


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WEEKDAY PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_named_day(
    ql: str, now: datetime, tz: pytz.BaseTzInfo
) -> Optional[Tuple[datetime, datetime]]:
    """
    Parse a named weekday reference and return (start, end) day bounds.

    "last monday"  → most recent Monday before today
    "this friday"  → this week's Friday
    "next friday"  → next occurrence of Friday
    bare / "on"    → most recent past occurrence
    """
    m = re.search(
        rf'\b(last|past|previous|this|on|next)?\s*({_DAY_PATTERN})\b',
        ql, re.IGNORECASE
    )
    if not m:
        return None

    prefix   = (m.group(1) or "").lower().strip()
    day_name = m.group(2).lower().strip()
    target_wd = _DAY_NAMES.get(day_name)
    if target_wd is None:
        return None

    today_wd = now.weekday()

    if prefix == "next":
        days_ahead = (target_wd - today_wd) % 7 or 7
        target_date = now + timedelta(days=days_ahead)
    elif prefix == "this":
        target_date = now + timedelta(days=(target_wd - today_wd))
    else:
        days_back = (today_wd - target_wd) % 7 or 7
        target_date = now - timedelta(days=days_back)

    return _day_range(target_date, tz)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def has_temporal_intent(question: str) -> bool:
    ql = question.lower()
    for kw in sorted(_TEMPORAL_TRIGGERS, key=len, reverse=True):
        if kw in ql:
            return True
    if re.search(r'\b(19|20)\d{2}\b', ql):
        return True
    if re.search(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', ql):
        return True
    if re.search(r'\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b', ql):
        return True
    if _TOD_RE.search(ql) or _TOD_WORDS_RE.search(ql):
        return True
    return False


def has_effective_intent(question: str) -> bool:
    ql = question.lower()
    return any(t in ql for t in _EFFECTIVE_TRIGGERS)


def has_current_intent(question: str) -> bool:
    ql = question.lower()
    _NOW_TRIGGERS = {
        "now", "right now", "at this moment", "as of now",
        "real-time", "realtime", "ongoing", "this hour",
        "last hour", "last minute", "last few minutes",
    }
    if re.search(r'\b(current|currently)\b', ql) and "currently active" not in ql:
        return True
    return any(t in ql for t in _NOW_TRIGGERS)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — INDIVIDUAL DATE PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_current_now(
    ql: str, now: datetime, tz: pytz.BaseTzInfo
) -> Optional[DateRange]:
    """Handle "current", "right now", "live", "this hour", etc."""
    _NOW_KWS = {
        "right now", "at this moment", "as of now",
        "real-time", "realtime", "ongoing", "last few minutes",
    }
    _CURRENT_KWS = {"current", "currently", "now", "live"}

    def _kw_in(kw: str, text: str) -> bool:
        if " " in kw or "-" in kw:
            return kw in text
        return bool(re.search(rf'\b{re.escape(kw)}\b', text))

    if not any(_kw_in(k, ql) for k in _NOW_KWS | _CURRENT_KWS):
        return None
    if "currently active" in ql:
        return None

    end = _now_truncated(now, tz)

    if "this hour" in ql:
        s, _ = _hour_range(now, tz)
        log.info("[DateResolver] current-now (this hour): %s → %s", s, end)
        return DateRange(start=s, end=end, grain="hour")

    if "last minute" in ql or "last few minutes" in ql:
        start = end - timedelta(minutes=CURRENT_WINDOW_MINUTES)
        log.info("[DateResolver] current-now (last %d min): %s → %s", CURRENT_WINDOW_MINUTES, start, end)
        return DateRange(start=start, end=end, grain="minute")

    start = end - timedelta(minutes=CURRENT_WINDOW_MINUTES)
    log.info("[DateResolver] current-now (rolling %d min): %s → %s", CURRENT_WINDOW_MINUTES, start, end)
    return DateRange(start=start, end=end, grain="minute")


def _parse_endpoint(
    s: str, now: datetime, tz: pytz.BaseTzInfo, prefer_start: bool
) -> Optional[Tuple[datetime, datetime]]:
    """Parse a single date endpoint string → (day_start, day_end)."""
    s = s.strip()

    m = re.fullmatch(r'(20\d{2})', s)
    if m:
        y = int(m.group(1))
        return _year_start(y, tz), _year_end(y, tz)

    m = re.search(rf'({_MONTH_PATTERN})\s*(20\d{{2}})?', s)
    if m:
        mn = _MONTH_NAMES[m.group(1)]
        yr = int(m.group(2)) if m.group(2) else now.year
        return _month_start(yr, mn, tz), _month_end(yr, mn, tz)

    m = re.search(r'q([1-4])\s*(20\d{2})?', s)
    if m:
        q, yr = int(m.group(1)), int(m.group(2)) if m.group(2) else now.year
        return _quarter_bounds(yr, q, tz)

    _rel = _parse_relative(s, now, tz)
    if _rel:
        return _rel.start, _rel.end
    _nd = _parse_named_day(s, now, tz)
    if _nd:
        return _nd

    dt = _try_parse_date_str(s, now, tz)
    if dt:
        return _day_range(dt, tz)

    return None


def _parse_explicit_range(
    ql: str, now: datetime, tz: pytz.BaseTzInfo
) -> Optional[DateRange]:
    """Handle "between X and Y" / "from X to Y"."""
    m = re.search(
        r'\bbetween\s+(.+?)\s+and\s+(.+?)(?=\s*(?:$|;|\.|,|\band\b))', ql
    ) or re.search(r'\bbetween\s+(.+?)\s+and\s+(.+)$', ql)

    if not m:
        m = re.search(
            r'\bfrom\s+(.+?)\s+(?:to|until|till)\s+(.+?)(?=\s*(?:$|;|\.|,))', ql
        ) or re.search(r'\bfrom\s+(.+?)\s+(?:to|until|till)\s+(.+)$', ql)

    if not m:
        return None

    raw_a, raw_b = m.group(1).strip(), m.group(2).strip()
    b = _parse_endpoint(raw_b, now, tz, prefer_start=False)
    if not b:
        return None

    # If raw_a has no explicit year, resolve it against the end year so that
    # "from march to june 2023" gives [2023-03, 2023-07) not [2026-03, 2023-07).
    _has_year = bool(re.search(r'\b20\d{2}\b', raw_a))
    if not _has_year:
        anchor_now = now.replace(year=b[0].year)
        a = _parse_endpoint(raw_a, anchor_now, tz, prefer_start=True)
    else:
        a = _parse_endpoint(raw_a, now, tz, prefer_start=True)

    if not a:
        return None
    start, end = a[0], b[1]
    if start >= end:
        return None
    log.info("[DateResolver] explicit-range: %s -> %s", start, end)
    return DateRange(start=start, end=end, grain="range")


def _parse_n_ago(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    """Handle "last 7 days", "past 3 months", "last 2 hours", etc."""
    m = _N_AGO_RE.search(ql)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    end = _midnight_tomorrow(now, tz)

    if unit == "minute":
        end   = _now_truncated(now, tz)
        start = end - timedelta(minutes=n)
        grain = "minute"
    elif unit == "hour":
        end   = _now_truncated(now, tz)
        start = end - timedelta(hours=n)
        grain = "hour"
    elif unit == "day":
        start = end - timedelta(days=n)
        grain = "day"
    elif unit == "week":
        start = end - timedelta(weeks=n)
        grain = "week"
    elif unit == "month":
        month, year = now.month - n, now.year
        while month <= 0:
            month += 12
            year  -= 1
        start = _month_start(year, month, tz)
        grain = "month"
    elif unit == "year":
        start = _year_start(now.year - n, tz)
        grain = "year"
    else:
        return None

    log.info("[DateResolver] n-ago (%d %ss): %s -> %s", n, unit, start, end)
    return DateRange(start=start, end=end, grain=grain)


def _parse_bare_unit_ago(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    """Handle "last hour", "last day", etc. with no explicit number."""
    m = _N_AGO_BARE_RE.search(ql)
    if not m:
        return None
    unit = m.group(1).lower()
    end  = _now_truncated(now, tz)

    if unit == "minute":
        start, grain = end - timedelta(minutes=1), "minute"
    elif unit == "hour":
        start, grain = end - timedelta(hours=1), "hour"
    elif unit == "day":
        start, end = _day_range(now - timedelta(days=1), tz)
        grain = "day"
    elif unit == "week":
        start, grain = end - timedelta(weeks=1), "week"
    elif unit == "month":
        first = datetime(now.year, now.month, 1)
        lm    = first - timedelta(days=1)
        start = _month_start(lm.year, lm.month, tz)
        end   = _month_end(lm.year, lm.month, tz)
        grain = "month"
    elif unit == "year":
        start = _year_start(now.year - 1, tz)
        end   = _year_end(now.year - 1, tz)
        grain = "year"
    else:
        return None

    log.info("[DateResolver] bare-unit-ago (1 %s): %s -> %s", unit, start, end)
    return DateRange(start=start, end=end, grain=grain)


def _parse_relative(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    """Handle today / yesterday / tomorrow / this week / last month / etc."""
    if re.search(r'\bday\s+after\s+tomorrow\b', ql):
        s, e = _day_range(now + timedelta(days=2), tz)
        return DateRange(start=s, end=e, grain="day")
    if re.search(r'\bday\s+before\s+yesterday\b', ql):
        s, e = _day_range(now - timedelta(days=2), tz)
        return DateRange(start=s, end=e, grain="day")
    if re.search(r'\btoday\b', ql):
        s, e = _day_range(now, tz)
        return DateRange(start=s, end=e, grain="day")
    if re.search(r'\byesterday\b', ql):
        s, e = _day_range(now - timedelta(days=1), tz)
        return DateRange(start=s, end=e, grain="day")
    if re.search(r'\btomorrow\b', ql):
        s, e = _day_range(now + timedelta(days=1), tz)
        return DateRange(start=s, end=e, grain="day")

    if re.search(r'\bthis\s+week\b', ql):
        monday = now - timedelta(days=now.weekday())
        s = _to_local(monday.replace(hour=0, minute=0, second=0, microsecond=0), tz)
        return DateRange(start=s, end=s + timedelta(weeks=1), grain="week")
    if re.search(r'\blast\s+week\b', ql):
        monday = now - timedelta(days=now.weekday() + 7)
        s = _to_local(monday.replace(hour=0, minute=0, second=0, microsecond=0), tz)
        return DateRange(start=s, end=s + timedelta(weeks=1), grain="week")
    if re.search(r'\bnext\s+week\b', ql):
        monday = now - timedelta(days=now.weekday()) + timedelta(weeks=1)
        s = _to_local(monday.replace(hour=0, minute=0, second=0, microsecond=0), tz)
        return DateRange(start=s, end=s + timedelta(weeks=1), grain="week")

    if re.search(r'\bthis\s+month\b', ql):
        s, e = _month_start(now.year, now.month, tz), _month_end(now.year, now.month, tz)
        return DateRange(start=s, end=e, grain="month")
    if re.search(r'\blast\s+month\b', ql):
        first = datetime(now.year, now.month, 1)
        lm    = first - timedelta(days=1)
        s, e  = _month_start(lm.year, lm.month, tz), _month_end(lm.year, lm.month, tz)
        return DateRange(start=s, end=e, grain="month")
    if re.search(r'\bnext\s+month\b', ql):
        nm = now.month + 1 if now.month < 12 else 1
        ny = now.year if now.month < 12 else now.year + 1
        s, e = _month_start(ny, nm, tz), _month_end(ny, nm, tz)
        return DateRange(start=s, end=e, grain="month")

    if re.search(r'\bthis\s+year\b', ql):
        s, e = _year_start(now.year, tz), _year_end(now.year, tz)
        return DateRange(start=s, end=e, grain="year")
    if re.search(r'\blast\s+year\b', ql):
        s, e = _year_start(now.year - 1, tz), _year_end(now.year - 1, tz)
        return DateRange(start=s, end=e, grain="year")
    if re.search(r'\bnext\s+year\b', ql):
        s, e = _year_start(now.year + 1, tz), _year_end(now.year + 1, tz)
        return DateRange(start=s, end=e, grain="year")

    if re.search(r'\bthis\s+hour\b', ql):
        s, _ = _hour_range(now, tz)
        end  = _now_truncated(now, tz)
        return DateRange(start=s, end=end, grain="hour")

    if re.search(r'\b(recent|recently|latest)\b', ql):
        end = _midnight_tomorrow(now, tz)
        return DateRange(start=end - timedelta(days=RECENT_DAYS), end=end, grain="day")

    return None


def _parse_named_day_range(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    """Wrapper that returns a DateRange from _parse_named_day."""
    result = _parse_named_day(ql, now, tz)
    if result is None:
        return None
    s, e = result
    log.info("[DateResolver] named-day: %s -> %s", s, e)
    return DateRange(start=s, end=e, grain="day")


def _parse_fiscal_year(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    cfy = _current_fiscal_year(now)

    if re.search(r"\b(this\s+)?(fy|financial\s+year|fiscal\s+year)\b", ql):
        yr = cfy - 1 if re.search(r"\blast\b", ql) else cfy
        s, e = _fiscal_year_bounds(yr, tz)
        log.info("[DateResolver] fiscal-year this/last: %s → %s", s, e)
        return DateRange(start=s, end=e, grain="year")

    m = re.search(r"\bfy\s*(20\d{2})\b", ql)
    if m:
        s, e = _fiscal_year_bounds(int(m.group(1)), tz)
        return DateRange(start=s, end=e, grain="year")

    m = re.search(r"\bfy(20\d{2})[-](\d{2,4})\b", ql)
    if m:
        s, e = _fiscal_year_bounds(int(m.group(1)) + 1, tz)
        return DateRange(start=s, end=e, grain="year")

    m = re.search(r"\b(?:financial|fiscal)\s+year\s+(20\d{2})\b", ql)
    if m:
        s, e = _fiscal_year_bounds(int(m.group(1)), tz)
        return DateRange(start=s, end=e, grain="year")

    return None


def _parse_quarter(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    cq = _current_quarter(now)

    if re.search(r'\bthis\s+quarter\b', ql):
        s, e = _quarter_bounds(now.year, cq, tz)
        return DateRange(start=s, end=e, grain="quarter")
    if re.search(r'\blast\s+quarter\b', ql):
        lq, yr = cq - 1, now.year
        if lq == 0:
            lq, yr = 4, yr - 1
        s, e = _quarter_bounds(yr, lq, tz)
        return DateRange(start=s, end=e, grain="quarter")
    if re.search(r'\bnext\s+quarter\b', ql):
        nq, yr = cq + 1, now.year
        if nq > 4:
            nq, yr = 1, yr + 1
        s, e = _quarter_bounds(yr, nq, tz)
        return DateRange(start=s, end=e, grain="quarter")

    m = re.search(r'\bq([1-4])\b(?:\s+(?:of\s+|in\s+)?(20\d{2}))?', ql)
    if m:
        q, yr = int(m.group(1)), int(m.group(2)) if m.group(2) else now.year
        s, e  = _quarter_bounds(yr, q, tz)
        return DateRange(start=s, end=e, grain="quarter")

    return None


def _parse_ytd_mtd(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    end = _now_truncated(now, tz)
    if re.search(r'\b(ytd|year[- ]to[- ]date)\b', ql):
        return DateRange(start=_year_start(now.year, tz), end=end, grain="year")
    if re.search(r'\b(mtd|month[- ]to[- ]date)\b', ql):
        return DateRange(start=_month_start(now.year, now.month, tz), end=end, grain="month")
    return None


def _parse_before_after_since(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    """Handle since X / before X / after X."""
    m = re.search(r'\bsince\s+(.+?)(?=\s*(?:$|;|\.|,|\band\b|\bor\b))', ql) \
        or re.search(r'\bsince\s+(.+)$', ql)
    if m:
        ep = _parse_endpoint(m.group(1).strip(), now, tz, prefer_start=True)
        if ep:
            end = _now_truncated(now, tz)
            log.info("[DateResolver] since: %s -> %s", ep[0], end)
            return DateRange(start=ep[0], end=end, grain="range")

    m = re.search(r'\b(before|until|till)\s+(.+?)(?=\s*(?:$|;|\.|,|\band\b|\bor\b))', ql) \
        or re.search(r'\b(before|until|till)\s+(.+)$', ql)
    if m:
        ep = _parse_endpoint(m.group(2).strip(), now, tz, prefer_start=True)
        if ep:
            far_past = _to_local(datetime(1900, 1, 1), tz)
            log.info("[DateResolver] before: %s -> %s", far_past, ep[0])
            return DateRange(start=far_past, end=ep[0], grain="range")

    m = re.search(r'\bafter\s+(.+?)(?=\s*(?:$|;|\.|,|\band\b|\bor\b))', ql) \
        or re.search(r'\bafter\s+(.+)$', ql)
    if m:
        raw = m.group(1).strip()
        # Guard: bare number (e.g. "after 12") is a time reference, not a date.
        # Let parse_time_clause handle it — skip here.
        if re.fullmatch(r'\d{1,2}', raw):
            pass
        else:
            ep = _parse_endpoint(raw, now, tz, prefer_start=False)
            if ep:
                end = _now_truncated(now, tz)
                log.info("[DateResolver] after: %s -> %s", ep[1], end)
                return DateRange(start=ep[1], end=end, grain="range")

    return None


def _parse_month_year(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    m = re.search(rf'\blast\s+({_MONTH_PATTERN})\b', ql)
    if m:
        mn = _MONTH_NAMES[m.group(1)]
        _s, _e = _month_start(now.year - 1, mn, tz), _month_end(now.year - 1, mn, tz)
        return DateRange(start=_s, end=_e, grain="month")

    m = re.search(rf'(?:in\s+|during\s+|for\s+|of\s+)?({_MONTH_PATTERN})\s*(20\d{{2}})?', ql)
    if m:
        mn = _MONTH_NAMES[m.group(1)]
        yr = int(m.group(2)) if m.group(2) else (now.year - 1 if mn > now.month else now.year)
        _s, _e = _month_start(yr, mn, tz), _month_end(yr, mn, tz)
        log.info("[DateResolver] month-year (%s %d): %s -> %s", m.group(1), yr, _s, _e)
        return DateRange(start=_s, end=_e, grain="month")

    return None


def _parse_year_only(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    m = re.search(r'\b(20\d{2})\b', ql)
    if m:
        yr = int(m.group(1))
        return DateRange(start=_year_start(yr, tz), end=_year_end(yr, tz), grain="year")
    return None


_DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y",
    "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
    "%B %d, %Y", "%b %d, %Y", "%d %B", "%d %b", "%B %d", "%b %d",
]


def _try_parse_date_str(s: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[datetime]:
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s.strip(), fmt)
            if dt.year == 1900:
                dt = dt.replace(year=now.year)
            return dt
        except ValueError:
            continue
    return None


def _parse_specific_date(ql: str, now: datetime, tz: pytz.BaseTzInfo) -> Optional[DateRange]:
    for pat in [
        r'\b(\d{4}-\d{2}-\d{2})\b',
        r'\b(\d{2}/\d{2}/\d{4})\b',
        r'\b(\d{2}-\d{2}-\d{4})\b',
    ]:
        m = re.search(pat, ql)
        if m:
            dt = _try_parse_date_str(m.group(1), now, tz)
            if dt:
                s, e = _day_range(dt, tz)
                return DateRange(start=s, end=e, grain="day")

    m = re.search(rf'(?:on\s+|at\s+)?(\d{{1,2}})\s+({_MONTH_PATTERN})(?:\s+(20\d{{2}}))?', ql)
    if m:
        try:
            dt = datetime(
                int(m.group(3)) if m.group(3) else now.year,
                _MONTH_NAMES[m.group(2)], int(m.group(1))
            )
            s, e = _day_range(dt, tz)
            return DateRange(start=s, end=e, grain="day")
        except ValueError:
            pass

    m = re.search(rf'({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+(20\d{{2}}))?', ql)
    if m:
        try:
            dt = datetime(
                int(m.group(3)) if m.group(3) else now.year,
                _MONTH_NAMES[m.group(1)], int(m.group(2))
            )
            s, e = _day_range(dt, tz)
            return DateRange(start=s, end=e, grain="day")
        except ValueError:
            pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CORE DATE RANGE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_date_range(question: str, tz_name: str = "UTC") -> Optional[DateRange]:
    """
    Parse a date range from a natural-language question.

    Date parsing and time-of-day parsing are fully separated:
      • Date range  → parse_date_range()  (this function)
      • Time clause → parse_time_clause() / attach via time_clause field

    Resolution order (first match wins):
      1.  current / now / live
      2.  explicit range  (between X and Y / from X to Y)
      3.  n-unit ago      (last 7 days, past 3 months …)
      4.  bare unit ago   (last hour, last minute)
      5.  relative        (today, yesterday, this week …)
      6.  named weekday   (last monday, next friday …)
      7.  fiscal year
      8.  calendar quarter
      9.  YTD / MTD
     10.  before / after / since
     11.  month + optional year
     12.  year only
     13.  specific date string

    A TimeClause (if found) is attached to the returned DateRange as
    `time_clause` WITHOUT mutating start/end boundaries.
    """
    if not has_temporal_intent(question):
        return None
    try:
        tz  = pytz.timezone(tz_name)
        now = datetime.now(tz)
        ql  = question.lower().strip()

        dr = (
            _parse_current_now(ql, now, tz)
            or _parse_explicit_range(ql, now, tz)
            or _parse_n_ago(ql, now, tz)
            or _parse_bare_unit_ago(ql, now, tz)
            or _parse_relative(ql, now, tz)
            or _parse_named_day_range(ql, now, tz)
            or _parse_fiscal_year(ql, now, tz)
            or _parse_quarter(ql, now, tz)
            or _parse_ytd_mtd(ql, now, tz)
            or _parse_before_after_since(ql, now, tz)
            or _parse_month_year(ql, now, tz)
            or _parse_year_only(ql, now, tz)
            or _parse_specific_date(ql, now, tz)
        )

        if dr is not None:
            # Attach time clause without touching date boundaries.
            # For rolling minute/hour grains: only suppress if the detected time
            # clause is itself a rolling window (not an explicit bounded qualifier).
            tc = parse_time_clause(ql)
            if tc is not None:
                _rolling_grains   = {"minute", "hour"}
                _rolling_qualifiers = {"rolling_minutes", "rolling_hours"}
                if dr.grain in _rolling_grains and tc.qualifier in _rolling_qualifiers:
                    pass  # suppress — the DateRange IS the time window
                else:
                    dr.time_clause = tc

        return dr

    except Exception as exc:
        log.warning("[DateResolver] parse_date_range failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SQL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _has_outer_where(sql: str) -> bool:
    depth, i = 0, 0
    while i < len(sql):
        ch = sql[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and sql[i:i + 5].upper() == 'WHERE':
            before = sql[i - 1] if i > 0 else ' '
            after  = sql[i + 5] if i + 5 < len(sql) else ' '
            if not (before.isalnum() or before == '_') and \
               not (after.isalnum() or after == '_'):
                return True
        i += 1
    return False


def _is_exists_pattern(sql: str) -> bool:
    return bool(re.match(r'^\s*SELECT\s+CASE\s+WHEN\s+EXISTS\s*\(', sql.strip(), re.I))


def _append_clause(sql: str, clause: str) -> str:
    """
    Inject a WHERE/AND condition into the SQL — always placed BEFORE
    GROUP BY / ORDER BY / HAVING.
    """
    sql = sql.strip().rstrip(";")
    connector = "AND" if _has_outer_where(sql) else "WHERE"

    _TAIL_KW = re.compile(r'\b(GROUP\s+BY|ORDER\s+BY|HAVING)\b', re.IGNORECASE)
    insert_pos = None
    for m in _TAIL_KW.finditer(sql):
        pos = m.start()
        d = sum(1 if c == '(' else -1 if c == ')' else 0 for c in sql[:pos])
        if d == 0:
            insert_pos = pos
            break

    if insert_pos is not None:
        before = sql[:insert_pos].rstrip()
        after  = sql[insert_pos:]
        result = f"{before} {connector} {clause} {after}"
    else:
        result = f"{sql} {connector} {clause}"

    return result.strip() + ";"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — STRIP LLM-GENERATED DATE FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def _find_outer_where_pos(sql: str) -> int:
    depth = 0
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and sql[i:i + 5].upper() == 'WHERE':
            before = sql[i - 1] if i > 0 else ' '
            after  = sql[i + 5] if i + 5 < len(sql) else ' '
            if not (before.isalnum() or before == '_') and \
               not (after.isalnum() or after == '_'):
                return i
        i += 1
    return -1


def strip_llm_date_filters(sql: str, date_cols: set) -> str:
    """
    Remove any date/time WHERE conditions the LLM generated.
    Catches: DATEADD/GETDATE/DATEDIFF/CONVERT/DATEPART/MONTH()/YEAR()/DAY()
    and direct comparisons like date_col >= 'YYYY-...'
    Also strips CAST(col AS TIME), DATEPART(HOUR, ...) time-of-day fragments.
    """
    date_cols_lower = {c.lower() for c in date_cols}

    _TIME_FUNC_RE = re.compile(
        r'\b(cast\s*\(.+?\s+as\s+time\)|datepart\s*\(\s*hour|datename\s*\(\s*weekday)',
        re.IGNORECASE
    )

    def _is_bad(cond: str) -> bool:
        cl = cond.lower().strip()
        if _DATE_FUNC_RE.search(cl) or _TIME_FUNC_RE.search(cl):
            return True
        m = re.search(r'\b(\w+\.)?(\w+)\s*(?:>=|<=|>|<|=)\s*\'(\d{4}[^\']*?)\'', cl)
        if m and m.group(2).strip() in date_cols_lower:
            return True
        return False

    where_pos = _find_outer_where_pos(sql)
    if where_pos == -1:
        return sql

    after_where = sql[where_pos + 5:]
    tail_m = re.search(r'\b(ORDER\s+BY|GROUP\s+BY|HAVING)\b', after_where, re.I)
    body   = after_where[:tail_m.start()].strip().rstrip(";") if tail_m else after_where.strip().rstrip(";")

    parts = re.split(r'\b(AND|OR)\b', body, flags=re.I)
    out, skip = [], False

    for p in parts:
        if p.strip().upper() in ("AND", "OR"):
            if not skip:
                out.append(p)
            skip = False
            continue
        if _is_bad(p):
            log.warning("[DateResolver] Stripped LLM date filter: %r", p.strip())
            if out and out[-1].strip().upper() in ("AND", "OR"):
                out.pop()
            skip = True
        else:
            out.append(p)

    new_where = "".join(out).strip()
    new_where = re.sub(r'^(AND|OR)\s+', '', new_where, flags=re.I)
    new_where = re.sub(r'\s+(AND|OR)\s*$', '', new_where, flags=re.I).strip()

    where_pos = _find_outer_where_pos(sql)
    if where_pos == -1:
        return sql.rstrip(";") + ";"

    prefix      = sql[:where_pos].rstrip()
    after_where = sql[where_pos:]
    tail_m      = re.search(r'\b(ORDER\s+BY|GROUP\s+BY|HAVING)\b', after_where, re.I)
    tail        = (" " + after_where[tail_m.start():].strip()) if tail_m else ""

    if not new_where or new_where.upper() in ("AND", "OR"):
        sql = prefix + tail
    else:
        sql = f"{prefix} WHERE {new_where}{tail}"

    return sql.rstrip(";") + ";"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — PUBLIC INJECTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def inject_date_filter(
    sql:           str,
    table:         str,
    question:      str,
    date_col:      Optional[str],
    all_date_cols: Optional[set] = None,
    tz_name:       str = "UTC",
) -> str:
    """
    Strip LLM date noise then inject the correct date range from the question.

    If a time-of-day clause is present it is appended as a *separate*
    AND condition using generic SQL functions:
      CAST(col AS TIME) >= 'HH:MM:00'
      DATEPART(HOUR, col) >= N
    This keeps the date boundary (start/end) untouched for month/year/week ranges.

    Parameters
    ----------
    sql           : SQL to modify
    table         : primary table name
    question      : original user question
    date_col      : creation/transaction date column (e.g. "Cdate")
    all_date_cols : all date columns — used for stripping
    tz_name       : user timezone string (e.g. "Asia/Kolkata")

    Returns modified SQL, or original if no date intent / no column.
    Never raises.
    """
    if not sql or not date_col:
        return sql
    if not has_temporal_intent(question):
        return sql
    if _is_exists_pattern(sql):
        log.info("[DateResolver] Skipping date inject — EXISTS pattern.")
        return sql

    cols_to_strip = (all_date_cols or set()) | {date_col}
    sql = strip_llm_date_filters(sql, cols_to_strip)

    dr = parse_date_range(question, tz_name=tz_name)
    if dr is None:
        log.info("[DateResolver] No date range resolved — SQL unchanged.")
        return sql

    # Build one combined clause: date range + optional time-of-day
    clause = dr.sql_fragment(table, date_col)
    if dr.time_clause:
        tc_sql = dr.time_clause.sql_fragment(table, date_col)
        if tc_sql:
            clause = f"{clause} AND {tc_sql}"
    log.info("[DateResolver] Injecting date filter: %s", clause)
    sql = _append_clause(sql, clause)

    return sql


def inject_time_filter(
    sql:      str,
    table:    str,
    question: str,
    date_col: Optional[str],
) -> str:
    """
    Inject a time-of-day filter WITHOUT touching any date boundaries.
    Useful when the caller has already injected the date range and only
    wants to layer a time clause on top.

    Uses generic SQL:
      CAST(col AS TIME) >= 'HH:MM:00'
      DATEPART(HOUR, col) BETWEEN N AND M
      DATENAME(WEEKDAY, col) = 'Monday'   ← if weekday context is needed

    Parameters
    ----------
    sql      : SQL to modify (should already have a date range injected)
    table    : primary table name
    question : original user question
    date_col : the datetime column to filter on

    Returns modified SQL, or original if no time clause detected.
    Never raises.
    """
    if not sql or not date_col:
        return sql
    try:
        ql = question.lower().strip()
        tc = parse_time_clause(ql)
        if tc is None:
            return sql
        tc_sql = tc.sql_fragment(table, date_col)
        if not tc_sql:
            return sql
        log.info("[DateResolver] inject_time_filter: %s", tc_sql)
        return _append_clause(sql, tc_sql)
    except Exception as exc:
        log.warning("[DateResolver] inject_time_filter failed: %s", exc)
        return sql


def inject_effective_filter(
    sql:            str,
    table:          str,
    question:       str,
    effective_from: Optional[str],
    effective_to:   Optional[str],
    tz_name:        str = "UTC",
    reference_date: Optional[datetime] = None,
) -> str:
    """
    Inject Effective_From/To validity window filter.
    Only fires when the user explicitly uses validity keywords.
    """
    if not sql or not effective_from or not effective_to:
        return sql
    if not has_effective_intent(question):
        log.info("[DateResolver] Skipping effective filter — no validity keyword.")
        return sql
    if _is_exists_pattern(sql):
        return sql

    try:
        tz  = pytz.timezone(tz_name)
        ref = reference_date or datetime.now(tz)
        if ref.tzinfo is None:
            ref = tz.localize(ref)
        ru = ref if ref.tzinfo else tz.localize(ref)
        clause = (
            f"{table}.{effective_from} <= '{ru:%Y-%m-%d %H:%M:%S}' "
            f"AND {table}.{effective_to} > '{ru:%Y-%m-%d %H:%M:%S}'"
        )
        log.info("[DateResolver] Injecting effective filter: %s", clause)
        return _append_clause(sql, clause)
    except Exception as exc:
        log.warning("[DateResolver] inject_effective_filter failed: %s", exc)
        return sql
