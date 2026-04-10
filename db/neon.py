from __future__ import annotations
import psycopg2
import psycopg2.extras
from datetime import date
from config import settings


def get_conn():
    """Return a new psycopg2 connection to Neon."""
    return psycopg2.connect(settings.database_url, sslmode="require")


def init_db():
    """Create tables if they don't exist. Called on app startup."""
    if not settings.database_url:
        print("⚠ No DATABASE_URL — skipping Neon init")
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        clerk_id   TEXT PRIMARY KEY,
                        email      TEXT,
                        name       TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS queries (
                        id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        clerk_id     TEXT NOT NULL REFERENCES users(clerk_id) ON DELETE CASCADE,
                        query        TEXT NOT NULL,
                        hyp_count    INT  DEFAULT 0,
                        paper_count  INT  DEFAULT 0,
                        health_score INT  DEFAULT 0,
                        top_hyp      TEXT DEFAULT '',
                        created_at   TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_queries_clerk_id ON queries(clerk_id);
                """)
            conn.commit()
        print("✓ Neon DB initialised")
    except Exception as e:
        print(f"⚠ Neon init error: {e}")


def upsert_user(clerk_id: str, email: str, name: str):
    """Insert or update a user row."""
    if not settings.database_url:
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (clerk_id, email, name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (clerk_id) DO UPDATE
                      SET email = EXCLUDED.email,
                          name  = EXCLUDED.name;
                """, (clerk_id, email, name))
            conn.commit()
    except Exception as e:
        print(f"⚠ upsert_user error: {e}")


def save_query(clerk_id: str, query: str, hyp_count: int,
               paper_count: int, health_score: int, top_hyp: str):
    """Persist a completed query run."""
    if not settings.database_url:
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO queries
                      (clerk_id, query, hyp_count, paper_count, health_score, top_hyp)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (clerk_id, query, hyp_count, paper_count, health_score, top_hyp))
            conn.commit()
    except Exception as e:
        print(f"⚠ save_query error: {e}")


def get_usage_today(clerk_id: str) -> int:
    """Return how many queries the user has run today (UTC)."""
    if not settings.database_url:
        return 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM queries
                    WHERE clerk_id = %s
                      AND created_at::date = CURRENT_DATE;
                """, (clerk_id,))
                row = cur.fetchone()
                return row[0] if row else 0
    except Exception as e:
        print(f"⚠ get_usage_today error: {e}")
        return 0


def get_history(clerk_id: str, limit: int = 20) -> list[dict]:
    """Return the user's recent query history."""
    if not settings.database_url:
        return []
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, query, hyp_count, paper_count, health_score,
                           top_hyp, created_at
                    FROM queries
                    WHERE clerk_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (clerk_id, limit))
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"⚠ get_history error: {e}")
        return []
