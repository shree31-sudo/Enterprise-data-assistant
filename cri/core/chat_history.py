
import logging
import threading
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config.db import get_db_connection

log = logging.getLogger("chat_history")

router = APIRouter(prefix="/chat", tags=["chat-history"])

# ── Retention ─────────────────────────────────────────────────────────────
#
# Conversations older than this (by Updated_At — i.e. last activity, not
# creation time) are hard-deleted, messages included. Not a soft delete:
# the point is to stop old, heavy result-set payloads from piling up in
# CHAT_MESSAGE.Content forever.
CHAT_HISTORY_RETENTION_DAYS = 7
_CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60  # re-check every 6 hours


# ── Identity resolution ──────────────────────────────────────────────────
#
# Every endpoint below calls this instead of trusting a client-supplied
# username/plant. Swap ONLY the body of this function when the main app's
# login is integrated — the rest of chat_history.py, and any pipeline code
# that calls save_message(), stays exactly as-is.

def get_chat_identity(request: Request) -> dict:
    # ── CURRENT (temporary) ──────────────────────────────────────────────
    return {"username": "DEV_USER", "plant": "DEV_PLANT"}

    # ── LATER (once main-app login/session is wired in) ─────────────────
    # return {
    #     "username": request.state.authenticated_username,
    #     "plant": request.state.selected_plant,
    # }


# ── Models ────────────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: Optional[str] = None  # usually left null; auto-set from first question


class ConversationOut(BaseModel):
    conversation_id: int
    title: Optional[str]
    created_at: str
    updated_at: str


class MessageIn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class MessageOut(BaseModel):
    message_id: int
    role: str
    content: str
    created_at: str


# ── Internal helpers (import save_message directly into chatbot_backend.py
#    to call it inline from _run_query_pipeline()) ─────────────────────────

def _verify_ownership(cur, conversation_id: int, username: str, plant: str) -> None:
    cur.execute(
        """
        SELECT 1 FROM CHAT_CONVERSATION
        WHERE Conversation_ID = ? AND Username = ? AND Plant = ? AND Is_Active = 1
        """,
        (conversation_id, username, plant),
    )
    if cur.fetchone() is None:
        # Deliberately vague — don't confirm whether the ID exists at all.
        raise HTTPException(status_code=404, detail="Conversation not found.")


def save_message(
    conversation_id: int,
    username: str,
    plant: str,
    role: str,
    content: str,
    title_override: Optional[str] = None,
) -> int:
    """
    Call this directly from your pipeline code (e.g. inside
    _run_query_pipeline in chatbot_backend.py) to save a message without
    a round-trip through the HTTP layer. `username`/`plant` here should
    come from get_chat_identity(request) at the call site — this function
    itself stays identity-agnostic so it's easy to unit test.

    `title_override`, if given, is used as the conversation's auto-title
    instead of the plain first-80-chars-of-content fallback below — e.g.
    a short LLM-generated topic ("Show Item Types") in place of the raw
    question. This function stays model-agnostic: the caller decides how
    (or whether) to generate one and just hands over a plain string.
    Ignored (safely) unless the message is from the user and the
    conversation doesn't already have a title — same guard as the
    fallback, so a stale/wrong is_new_chat flag from the client can never
    overwrite a real, already-set title.

    Returns the new Message_ID. Raises HTTPException(404) if the
    conversation doesn't belong to this username+plant.
    """
    if role not in ("user", "assistant"):
        raise ValueError("role must be 'user' or 'assistant'")

    with get_db_connection() as conn:
        cur = conn.cursor()
        _verify_ownership(cur, conversation_id, username, plant)

        cur.execute(
            """
            INSERT INTO CHAT_MESSAGE (Conversation_ID, Role, Content)
            OUTPUT INSERTED.Message_ID
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content),
        )
        message_id = cur.fetchone()[0]

        # bump Updated_At so the sidebar re-sorts this conversation to top
        cur.execute(
            "UPDATE CHAT_CONVERSATION SET Updated_At = GETDATE() WHERE Conversation_ID = ?",
            (conversation_id,),
        )

        # auto-title from the first user message if none set yet
        if role == "user":
            cur.execute(
                "SELECT Title FROM CHAT_CONVERSATION WHERE Conversation_ID = ?",
                (conversation_id,),
            )
            row = cur.fetchone()
            if row and not row[0]:
                auto_title = (title_override or content).strip()[:40]
                cur.execute(
                    "UPDATE CHAT_CONVERSATION SET Title = ? WHERE Conversation_ID = ?",
                    (auto_title, conversation_id),
                )

        conn.commit()
        return message_id


# ── Retention / cleanup ───────────────────────────────────────────────────

def purge_old_conversations(retention_days: int = CHAT_HISTORY_RETENTION_DAYS) -> int:
    """
    Hard-deletes any conversation (and all of its messages) whose
    Updated_At is older than `retention_days`. This is deliberately a
    real DELETE, not the Is_Active soft-delete used by the user-facing
    "delete chat" action — the goal is to stop old result-set payloads
    (rows/SQL JSON) from accumulating in CHAT_MESSAGE indefinitely.

    Idempotent / safe to call repeatedly or from multiple workers: each
    run only touches rows that are already past the cutoff, and a run
    that finds nothing stale is a cheap no-op.

    Returns the number of conversations removed.
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT Conversation_ID FROM CHAT_CONVERSATION
            WHERE Updated_At < DATEADD(day, -?, GETDATE())
            """,
            (retention_days,),
        )
        stale_ids = [r[0] for r in cur.fetchall()]

        if not stale_ids:
            return 0

        placeholders = ",".join("?" for _ in stale_ids)

        # Messages first (no FK-cascade assumption made either way).
        cur.execute(
            f"DELETE FROM CHAT_MESSAGE WHERE Conversation_ID IN ({placeholders})",
            stale_ids,
        )
        cur.execute(
            f"DELETE FROM CHAT_CONVERSATION WHERE Conversation_ID IN ({placeholders})",
            stale_ids,
        )
        conn.commit()

    log.info(
        "[chat_history][cleanup] purged %d conversation(s) older than %d day(s): %s",
        len(stale_ids), retention_days, stale_ids,
    )
    return len(stale_ids)


def start_chat_history_cleanup_scheduler(
    retention_days: int = CHAT_HISTORY_RETENTION_DAYS,
    interval_seconds: int = _CLEANUP_INTERVAL_SECONDS,
) -> None:
    """
    Fire-and-forget background loop. Call this once at app startup
    (see lifespan() in chatbot_backend.py, right alongside the existing
    start_cache_refresh_scheduler() call) — it runs purge_old_conversations()
    immediately, then again every `interval_seconds`, for the lifetime of
    the process. A daemon thread, so it never blocks shutdown.
    """
    def _loop():
        while True:
            try:
                purge_old_conversations(retention_days)
            except Exception:
                log.exception("[chat_history][cleanup] purge run failed")
            time.sleep(interval_seconds)

    threading.Thread(target=_loop, name="chat-history-cleanup", daemon=True).start()
    log.info(
        "[chat_history] Cleanup scheduler started (retention=%dd, check every %ds)",
        retention_days, interval_seconds,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/conversations", response_model=ConversationOut)
async def create_conversation(req: ConversationCreate, request: Request):
    identity = get_chat_identity(request)
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO CHAT_CONVERSATION (Username, Plant, Title)
            OUTPUT INSERTED.Conversation_ID, INSERTED.Created_At, INSERTED.Updated_At
            VALUES (?, ?, ?)
            """,
            (identity["username"], identity["plant"], req.title),
        )
        row = cur.fetchone()
        conn.commit()
        return ConversationOut(
            conversation_id=row[0],
            title=req.title,
            created_at=str(row[1]),
            updated_at=str(row[2]),
        )


@router.get("/conversations", response_model=List[ConversationOut])
async def list_conversations(request: Request):
    identity = get_chat_identity(request)
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT Conversation_ID, Title, Created_At, Updated_At
            FROM CHAT_CONVERSATION
            WHERE Username = ? AND Plant = ? AND Is_Active = 1
            ORDER BY Updated_At DESC
            """,
            (identity["username"], identity["plant"]),
        )
        rows = cur.fetchall()
        return [
            ConversationOut(
                conversation_id=r[0], title=r[1], created_at=str(r[2]), updated_at=str(r[3])
            )
            for r in rows
        ]


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageOut])
async def get_conversation_messages(conversation_id: int, request: Request):
    identity = get_chat_identity(request)
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.Message_ID, m.Role, m.Content, m.Created_At
            FROM CHAT_MESSAGE m
            INNER JOIN CHAT_CONVERSATION c ON m.Conversation_ID = c.Conversation_ID
            WHERE c.Conversation_ID = ? AND c.Username = ? AND c.Plant = ? AND c.Is_Active = 1
            ORDER BY m.Created_At ASC
            """,
            (conversation_id, identity["username"], identity["plant"]),
        )
        rows = cur.fetchall()
        if not rows:
            # Could be: empty convo (valid) OR wrong owner (invalid).
            # Disambiguate with the ownership check so we return the right
            # status code either way.
            _verify_ownership(cur, conversation_id, identity["username"], identity["plant"])
        return [
            MessageOut(message_id=r[0], role=r[1], content=r[2], created_at=str(r[3]))
            for r in rows
        ]


@router.post("/conversations/{conversation_id}/messages", response_model=MessageOut)
async def post_message(conversation_id: int, req: MessageIn, request: Request):
    identity = get_chat_identity(request)
    message_id = save_message(
        conversation_id=conversation_id,
        username=identity["username"],
        plant=identity["plant"],
        role=req.role,
        content=req.content,
    )
    return MessageOut(message_id=message_id, role=req.role, content=req.content, created_at="")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, request: Request):
    """Soft delete — sets Is_Active = 0 rather than actually deleting rows."""
    identity = get_chat_identity(request)
    with get_db_connection() as conn:
        cur = conn.cursor()
        _verify_ownership(cur, conversation_id, identity["username"], identity["plant"])
        cur.execute(
            "UPDATE CHAT_CONVERSATION SET Is_Active = 0 WHERE Conversation_ID = ?",
            (conversation_id,),
        )
        conn.commit()
    return {"status": "deleted", "conversation_id": conversation_id}