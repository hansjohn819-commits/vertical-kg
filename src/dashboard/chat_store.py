"""Chat persistence backed by a single sqlite file (§16.7 R4 / §16.9).

Schema rationale:
- `users` table reserves a uuid slot per role so future multi-user mode is
  a UI/auth question, not a schema migration. 2026-05-06 seeds two
  hardcoded rows (one per role) — capstone scope has no auth.
- `conversations.mode` is denormalized from `users.role` so a future
  user-with-multiple-roles use case doesn't break historic conversations.
  Currently always equals the user's role.
- `messages.tool_calls_json` stores the OpenAI tool_calls list verbatim
  (JSON-serialized) so reload reproduces the original assistant message
  shape that GraphAgent expects to feed back into the model.

Public surface (5 functions): list_chats / load_chat / save_chat / new_chat
/ delete_chat. All writes are single transactions; per-conversation save
is `INSERT OR REPLACE` style — full overwrite of message rows for the
conversation, simpler than diff-merging and the row count is small.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Default-user uuids (written into MEMORY.md too if needed). Fixed strings
# so chats.db is portable across machines without re-seeding.
INTERNAL_DEFAULT_USER_ID = "00000000-0000-0000-0000-00000000aaaa"
EXTERNAL_DEFAULT_USER_ID = "00000000-0000-0000-0000-00000000bbbb"

ROLE_INTERNAL = "internal"
ROLE_EXTERNAL = "external"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id           TEXT PRIMARY KEY,
    role         TEXT NOT NULL,
    display_name TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    mode       TEXT NOT NULL,
    name       TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_user_updated
    ON conversations(user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    conv_id          TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    idx              INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT,
    tool_calls_json  TEXT,
    ts               TEXT NOT NULL,
    PRIMARY KEY (conv_id, idx)
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema_and_seed(conn: sqlite3.Connection) -> None:
    """Create tables + seed default users if missing. Idempotent."""
    conn.executescript(_SCHEMA_SQL)
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        ts = _now_iso()
        conn.executemany(
            "INSERT INTO users(id, role, display_name, created_at) VALUES (?, ?, ?, ?)",
            [
                (INTERNAL_DEFAULT_USER_ID, ROLE_INTERNAL, "Internal Researcher", ts),
                (EXTERNAL_DEFAULT_USER_ID, ROLE_EXTERNAL, "External Viewer", ts),
            ],
        )
        conn.commit()


class ChatStore:
    """Façade over the sqlite file. Cheap to construct; safe to share across
    Streamlit reruns via `@st.cache_resource`. Each call opens its own
    short-lived connection — sqlite handles serialization internally and
    we avoid worrying about cross-thread reuse."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        with _connect(self.db_path) as conn:
            _ensure_schema_and_seed(conn)

    # --- Conversations ---

    def new_chat(self, user_id: str, mode: str, name: str | None = None) -> str:
        """Create an empty conversation, return its uuid."""
        cid = str(uuid.uuid4())
        ts = _now_iso()
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations(id, user_id, mode, name, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (cid, user_id, mode, name, ts, ts),
            )
            conn.commit()
        return cid

    def list_chats(self, user_id: str) -> list[dict]:
        """Most-recently-active first."""
        with _connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, name, mode, created_at, updated_at FROM conversations"
                " WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_chat(self, conv_id: str) -> None:
        with _connect(self.db_path) as conn:
            # ON DELETE CASCADE on messages handles the row sweep.
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()

    # --- Messages ---

    def load_chat(self, conv_id: str) -> tuple[dict | None, list[dict]]:
        """Return (conversation_meta, messages_in_order). Tool calls are
        deserialized back to native lists so callers see the same dict
        shape they'd get from the OpenAI client."""
        with _connect(self.db_path) as conn:
            meta_row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,),
            ).fetchone()
            if meta_row is None:
                return None, []
            msg_rows = conn.execute(
                "SELECT idx, role, content, tool_calls_json, ts FROM messages"
                " WHERE conv_id = ? ORDER BY idx ASC",
                (conv_id,),
            ).fetchall()
        messages: list[dict] = []
        for r in msg_rows:
            m: dict = {"role": r["role"], "content": r["content"] or ""}
            if r["tool_calls_json"]:
                try:
                    m["tool_calls"] = json.loads(r["tool_calls_json"])
                except json.JSONDecodeError:
                    pass
            messages.append(m)
        return dict(meta_row), messages

    def save_chat(
        self,
        conv_id: str,
        user_id: str,
        mode: str,
        messages: list[dict],
        name: str | None = None,
    ) -> None:
        """Full overwrite of the conversation's message list. The conversation
        row is upserted (created if missing). Auto-derives a name from the
        first user message if not provided and the conversation has no name yet.
        """
        ts = _now_iso()
        with _connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT name FROM conversations WHERE id = ?", (conv_id,),
            ).fetchone()
            derived_name = name
            if existing is None:
                if derived_name is None:
                    derived_name = _derive_name(messages)
                conn.execute(
                    "INSERT INTO conversations(id, user_id, mode, name, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (conv_id, user_id, mode, derived_name, ts, ts),
                )
            else:
                # Keep existing name unless caller explicitly overrides; if
                # the existing name is empty and there's now content, derive.
                final_name = name if name is not None else (
                    existing["name"] or _derive_name(messages)
                )
                conn.execute(
                    "UPDATE conversations SET name = ?, mode = ?, updated_at = ?"
                    " WHERE id = ?",
                    (final_name, mode, ts, conv_id),
                )
            # Rewrite all messages — small rows, simpler than diff merge.
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
            for idx, m in enumerate(messages):
                tool_calls_json = (
                    json.dumps(m["tool_calls"]) if m.get("tool_calls") else None
                )
                conn.execute(
                    "INSERT INTO messages(conv_id, idx, role, content, tool_calls_json, ts)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        conv_id, idx, m.get("role", "user"),
                        m.get("content"), tool_calls_json, ts,
                    ),
                )
            conn.commit()


def _derive_name(messages: list[dict]) -> str:
    """First user message's first ~60 chars, single-line."""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = " ".join(str(m["content"]).split())
            return text[:60] + ("…" if len(text) > 60 else "")
    return "(empty chat)"
