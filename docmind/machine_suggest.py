"""
machine_suggest.py
────────────────────────────────────────────────────────────────────────────
Machine-name typeahead suggestion API for DocBot.

What this is for
-----------------
As the end user types a question ("How to calibrate J..."), the frontend
wants to suggest known machine names that match what's been typed so far,
narrowing on every keystroke:

    "J"            -> JJAPQ1425, JJAPQ1430
    "JJAP"         -> JJAPQ1425, JJAPQ1430
    "JJAPQ142"     -> JJAPQ1425

Once the user picks a suggestion, the frontend sends that machine_name back
explicitly on /enduser/chat instead of leaving the backend to guess it out
of free-text. That's a separate change made in docbot_backend.py (ChatRequest
gained an optional `machine_name` field) -- this file is only the suggestion
endpoint itself.

Why a separate file
--------------------
This is pure string matching against a list docbot_backend.py already
builds and caches per module (get_known_machine_names) -- no LLM, no
embeddings, no Qdrant calls of its own. Keeping it out of the (already
large) docbot_backend.py keeps the matching logic easy to read and unit
test in isolation.

Wiring
------
docbot_backend.py already owns the machine-name cache, so rather than have
this file import docbot_backend (which imports this file to register the
router -- a circular import), the router is built by a small factory that
takes get_known_machine_names as a plain argument:

    from machine_suggest import create_machine_router
    app.include_router(create_machine_router(get_known_machine_names))
"""
import re
from typing import Awaitable, Callable, List

from fastapi import APIRouter, Query

# Same normalization convention used for machine-name matching elsewhere in
# DocBot (see docbot_backend._normalize_machine_token): case, spaces,
# hyphens, and underscores are all irrelevant to whether two machine names
# "match" -- only the alphanumeric characters count.
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def create_machine_router(
    get_known_machine_names: Callable[[str], Awaitable[List[str]]],
) -> APIRouter:
    """
    Build the /machines router.

    `get_known_machine_names(module)` must be an async callable returning
    the distinct machine names on file for that module (docbot_backend.py's
    cached version, which already folds in "ALL"-tagged common documents,
    is what's passed in practice).
    """
    router = APIRouter()

    @router.get("/machines")
    async def suggest_machines(
        module: str = Query(..., description="Module id, e.g. 'CNC'"),
        query: str = Query("", description="Fragment typed so far"),
        limit: int = Query(10, ge=1, le=50),
    ):
        known_names = await get_known_machine_names(module)

        q_norm = _normalize(query)
        if not q_norm:
            # Nothing typed yet -- hand back the known list (capped below)
            # so the frontend can use it as an initial browse list too.
            matches = known_names
        else:
            # Prefix match on the normalized name is what makes the
            # progressive narrowing in the spec work -- each extra
            # character typed only narrows the candidate set, it never
            # changes it unpredictably the way a plain substring match
            # could (e.g. a fragment matching the *middle* of one name and
            # the *start* of another).
            matches = [m for m in known_names if _normalize(m).startswith(q_norm)]
            if not matches:
                # Nothing starts with it -- fall back to substring match so
                # a mid-name fragment (a suffix/code the user remembers)
                # still surfaces something instead of an empty list.
                matches = [m for m in known_names if q_norm in _normalize(m)]

        # Unique, keep source order (already sorted alphabetically by
        # get_known_machine_names), capped to `limit`.
        seen = set()
        unique_matches = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                unique_matches.append(m)
        unique_matches = unique_matches[:limit]

        return [{"machine_name": m} for m in unique_matches]

    return router
