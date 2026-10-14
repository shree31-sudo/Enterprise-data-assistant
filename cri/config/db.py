from dotenv import load_dotenv
load_dotenv()  # Load .env file into os.environ

import os
import queue
import threading
import logging
import pyodbc
from contextlib import contextmanager

log = logging.getLogger("NL2SQL")

# =========================
# DATABASE CONFIG — from .env
# =========================
# Required keys in your .env:
#   DB_DRIVER=ODBC Driver 17 for SQL Server
#   DB_SERVER=10.225.30.26
#   DB_NAME=LAMI_CNC_PROD0302
#   DB_USER=sa
#   DB_PASSWORD=sql@123
#   DB_TRUST_CERT=yes
#   DB_POOL_SIZE=5        (optional, default 5)
#   DB_POOL_TIMEOUT=10    (optional, default 10s)

_POOL_SIZE    = int(os.getenv("DB_POOL_SIZE", "5"))
_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "10"))


def get_connection_string() -> str:
    driver     = os.getenv("DB_DRIVER",     "ODBC Driver 17 for SQL Server")
    server     = os.getenv("DB_SERVER",     "")
    database   = os.getenv("DB_NAME",       "")
    username   = os.getenv("DB_USER",       "")
    password   = os.getenv("DB_PASSWORD",   "")
    trust_cert = os.getenv("DB_TRUST_CERT", "yes")

    if not server or not database or not username or not password:
        raise RuntimeError(
            "Missing DB credentials in .env — "
            "DB_SERVER, DB_NAME, DB_USER and DB_PASSWORD are all required."
        )

    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate={trust_cert};"
    )


# =========================
# CONNECTION POOL
# =========================

class _ConnectionPool:
    """
    Simple fixed-size pyodbc connection pool.
    Connections are created once at startup and reused across requests,
    eliminating the per-request TCP + ODBC handshake cost (20–80ms each).
    """

    def __init__(self, size: int) -> None:
        self._size        = size
        self._conn_string = get_connection_string()
        self._pool: queue.Queue = queue.Queue(maxsize=size)
        self._lock        = threading.Lock()

        for _ in range(size):
            self._pool.put(self._new_conn())

        log.info("[db_pool] Initialised — pool_size=%d", size)

    def _new_conn(self):
        conn = pyodbc.connect(self._conn_string, timeout=5)
        conn.autocommit = False
        return conn

    def _is_alive(self, conn) -> bool:
        try:
            conn.cursor().execute("SELECT 1")
            return True
        except Exception:
            return False

    def acquire(self) -> pyodbc.Connection:
        try:
            conn = self._pool.get(timeout=_POOL_TIMEOUT)
        except queue.Empty:
            raise TimeoutError(
                f"[db_pool] No free connection after {_POOL_TIMEOUT}s "
                f"(pool_size={self._size}). Consider increasing DB_POOL_SIZE."
            )
        # Health-check — silently replace a dead connection
        if not self._is_alive(conn):
            log.warning("[db_pool] Stale connection detected — replacing.")
            try:
                conn.close()
            except Exception:
                pass
            conn = self._new_conn()
        return conn

    def release(self, conn) -> None:
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            # Shouldn't happen in normal use — just discard
            try:
                conn.close()
            except Exception:
                pass

    def status(self) -> dict:
        available = self._pool.qsize()
        return {
            "pool_size": self._size,
            "available": available,
            "in_use":    self._size - available,
        }


# Singleton — created once when this module is first imported
_pool: "_ConnectionPool | None" = None
_pool_init_lock = threading.Lock()


def _get_pool() -> _ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_init_lock:
            if _pool is None:
                _pool = _ConnectionPool(_POOL_SIZE)
    return _pool


# =========================
# PUBLIC API — unchanged interface
# =========================

@contextmanager
def get_db_connection():
    """
    Context manager — checks out a pooled connection, yields it,
    then returns it to the pool.

    Usage (identical to before):
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(...)
    """
    pool = _get_pool()
    conn = pool.acquire()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        pool.release(conn)


def get_pool_status() -> dict:
    """
    Returns current pool stats — plug into /health endpoint.
    """
    return _get_pool().status()


def test_db_connection() -> bool:
    """
    Simple connectivity test — unchanged behaviour.
    """
    try:
        with get_db_connection() as conn:
            conn.cursor().execute("SELECT 1").fetchone()
        return True
    except Exception as e:
        print("DB Connection Failed:", e)
        return False