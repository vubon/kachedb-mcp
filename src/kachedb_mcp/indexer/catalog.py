"""Local publication catalog for scoped KacheDB records."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from .worktree import WorktreeError

if TYPE_CHECKING:
    from collections.abc import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    namespace TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    indexed INTEGER NOT NULL DEFAULT 0,
    head TEXT,
    indexed_dirty_files TEXT NOT NULL DEFAULT '[]',
    ignored_tracked_files TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS files (
    path BLOB PRIMARY KEY,
    digest TEXT,
    owner TEXT,
    parse_failed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS definitions (
    path BLOB NOT NULL REFERENCES files ON DELETE CASCADE,
    name TEXT NOT NULL,
    record TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS definitions_by_name ON definitions(name);
CREATE INDEX IF NOT EXISTS definitions_by_path ON definitions(path);
CREATE TABLE IF NOT EXISTS documents (
    topic TEXT PRIMARY KEY,
    ref TEXT NOT NULL UNIQUE,
    sources TEXT NOT NULL,
    expires REAL
);
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT PRIMARY KEY,
    document TEXT NOT NULL REFERENCES documents(ref) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS code_vectors (
    path BLOB NOT NULL REFERENCES files ON DELETE CASCADE,
    profile TEXT NOT NULL,
    definition TEXT NOT NULL REFERENCES definitions(record) ON DELETE CASCADE,
    owner TEXT NOT NULL,
    PRIMARY KEY (definition, profile)
);
CREATE INDEX IF NOT EXISTS code_vectors_by_profile ON code_vectors(profile);
CREATE INDEX IF NOT EXISTS code_vectors_by_path ON code_vectors(path);
CREATE TABLE IF NOT EXISTS objects (
    record TEXT NOT NULL,
    index_name TEXT NOT NULL,
    owner TEXT NOT NULL,
    attempt TEXT NOT NULL,
    state TEXT NOT NULL,
    deadline REAL NOT NULL,
    PRIMARY KEY (index_name, record)
);
CREATE INDEX IF NOT EXISTS objects_by_owner ON objects(owner);
CREATE INDEX IF NOT EXISTS objects_by_attempt ON objects(attempt);
CREATE INDEX IF NOT EXISTS objects_by_deadline ON objects(state, deadline);
"""


@dataclass(frozen=True)
class State:
    generation: int
    indexed: bool
    head: str | None
    dirty: set[str]
    ignored: set[str]


@dataclass(frozen=True)
class Record:
    owner: str
    key: str
    index: str = ""


class CatalogNotReadyError(Exception):
    """A read-only caller reached the catalog before schema publication."""


def catalog_path(root: Path, workspace: str, host: str, port: int) -> Path:
    custom = os.getenv("KACHEDB_MCP_STATE_DIR")
    if custom:
        base = Path(custom).expanduser()
    elif os.name == "nt" and os.getenv("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "kachedb-mcp"
    else:
        state_home = os.getenv("XDG_STATE_HOME")
        base = (Path(state_home) if state_home else Path.home() / ".local/state") / "kachedb-mcp"
    identity = json.dumps([os.path.normcase(str(root)), workspace.strip(), host, port])
    return base / (hashlib.sha256(identity.encode()).hexdigest() + ".sqlite3")


class Catalog:
    def __init__(self, path: Path, timeout: float, *, read_only: bool = False) -> None:
        if not read_only and not path.exists():
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Create the catalog with restrictive permissions before SQLite opens it.
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
            except FileExistsError:
                pass
        mode = "ro" if read_only else "rw"
        try:
            self.db = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode={mode}",
                uri=True,
                timeout=timeout,
                isolation_level=None,
            )
        except sqlite3.OperationalError as exc:
            if read_only and not path.exists():
                raise CatalogNotReadyError from exc
            raise
        try:
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA foreign_keys = ON")
            version = self._version()
            if version > 1:
                raise sqlite3.DatabaseError("Unsupported catalog schema version")
            if version == 0:
                if read_only:
                    raise CatalogNotReadyError
                self._initialize(time.monotonic() + timeout)
            self.namespace: str = self.db.execute(
                "SELECT namespace FROM state WHERE id = 1"
            ).fetchone()[0]
        except BaseException:
            self.db.close()
            raise

    def _version(self) -> int:
        return int(self.db.execute("PRAGMA user_version").fetchone()[0])

    def _initialize(self, deadline: float) -> None:
        # The initial journal-mode switch can report SQLITE_BUSY immediately if
        # another process is opening the same new database.
        while True:
            try:
                mode = self.db.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if mode != "wal":
                    raise sqlite3.DatabaseError("The catalog could not enable WAL")
                break
            except sqlite3.OperationalError as exc:
                remaining = deadline - time.monotonic()
                if "locked" not in str(exc).lower() or remaining <= 0:
                    raise
                time.sleep(min(0.01, remaining))
        with self.transaction(write=True):
            version = self._version()
            if version == 1:
                return
            if version != 0:
                raise sqlite3.DatabaseError("Unsupported catalog schema version")
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    self.db.execute(statement)
            columns = {row[1] for row in self.db.execute("PRAGMA table_info(files)")}
            if "parse_failed" not in columns:
                self.db.execute(
                    "ALTER TABLE files ADD COLUMN parse_failed INTEGER NOT NULL DEFAULT 0"
                )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS unparsed_files ON files(path) WHERE parse_failed = 1"
            )
            self.db.execute(
                "INSERT OR IGNORE INTO state (id, namespace) VALUES (1, ?)", (uuid4().hex,)
            )
            self.db.execute("PRAGMA user_version = 1")

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def transaction(self, *, write: bool = False) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def state(self) -> State:
        row = self.db.execute(
            "SELECT generation, indexed, head, indexed_dirty_files, ignored_tracked_files"
            " FROM state WHERE id = 1"
        ).fetchone()
        return State(row[0], bool(row[1]), row[2], set(json.loads(row[3])), set(json.loads(row[4])))

    def files(self) -> dict[str, tuple[str | None, str | None]]:
        rows = self.db.execute("SELECT path, digest, owner FROM files")
        return {os.fsdecode(row[0]): (row[1], row[2]) for row in rows}

    def stage(self, attempt: str, records: list[Record], deadline: float) -> None:
        with self.transaction(write=True):
            self.db.executemany(
                "INSERT INTO objects VALUES (?, ?, ?, ?, 'pending', ?)",
                [(entry.key, entry.index, entry.owner, attempt, deadline) for entry in records],
            )

    @contextmanager
    def publish(self, base: int, attempt: str | None = None) -> Iterator[None]:
        with self.transaction(write=True):
            if self.state().generation != base:
                raise WorktreeError("RETRY", "Another operation published a newer revision.")
            if attempt:
                invalid = self.db.execute(
                    "SELECT 1 FROM objects WHERE attempt = ?"
                    " AND (state != 'pending' OR deadline <= ?) LIMIT 1",
                    (attempt, time.time()),
                ).fetchone()
                if invalid:
                    raise WorktreeError("RETRY", "The staged revision is no longer available.")
                self.db.execute("UPDATE objects SET state = 'active' WHERE attempt = ?", (attempt,))
            yield
            self.db.execute("UPDATE state SET generation = generation + 1 WHERE id = 1")

    def abandon(self, attempt: str) -> None:
        self.db.execute(
            "UPDATE objects SET state = 'retired', deadline = deadline + 2"
            " WHERE attempt = ? AND state = 'pending'",
            (attempt,),
        )

    def retire(self, owner: str | None) -> None:
        if owner:
            self.db.execute(
                "UPDATE objects SET state = 'retired', deadline = 0"
                " WHERE owner = ? AND state = 'active'",
                (owner,),
            )

    def garbage(self) -> list[Record]:
        now = time.time()
        # Atomically claim expired attempts before allowing their records to be deleted.
        self.db.execute(
            "UPDATE objects SET state = 'retired', deadline = deadline + 2"
            " WHERE state = 'pending' AND deadline <= ?",
            (now,),
        )
        rows = self.db.execute(
            "SELECT owner, record, index_name FROM objects"
            " WHERE state = 'retired' AND deadline <= ? LIMIT 256",
            (now,),
        )
        return [Record(*row) for row in rows]

    def collected(self, record: Record) -> None:
        self.db.execute(
            "DELETE FROM objects WHERE index_name = ? AND record = ? AND state = 'retired'",
            (record.index, record.key),
        )

    def replace_file(
        self,
        path: str,
        digest: str | None,
        owner: str | None,
        definitions: list[tuple[str, str, int]],
        *,
        eligible: bool,
        parse_failed: bool = False,
    ) -> None:
        encoded = os.fsencode(path)
        row = self.db.execute("SELECT owner FROM files WHERE path = ?", (encoded,)).fetchone()
        if row:
            self.retire(row[0])
        for row in self.db.execute("SELECT owner FROM code_vectors WHERE path = ?", (encoded,)):
            self.retire(row[0])
        self.db.execute("DELETE FROM files WHERE path = ?", (encoded,))
        if eligible:
            self.db.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?)", (encoded, digest, owner, parse_failed)
            )
            self.db.executemany(
                "INSERT INTO definitions VALUES (?, ?, ?, ?)",
                [(encoded, name, key, ordinal) for name, key, ordinal in definitions],
            )

    def parse_warnings(self) -> dict[str, int | list[str]]:
        count = self.db.execute("SELECT count(*) FROM files WHERE parse_failed = 1").fetchone()[0]
        if not count:
            return {}
        paths = self.db.execute(
            "SELECT path FROM files WHERE parse_failed = 1 ORDER BY path LIMIT 20"
        )
        return {
            "unparsed_file_count": count,
            "unparsed_files": [os.fsdecode(row[0]) for row in paths],
        }

    def publish_worktree(
        self, head: str | None, dirty: frozenset[str], ignored: frozenset[str]
    ) -> None:
        self.db.execute(
            "UPDATE state SET indexed = 1, head = ?, indexed_dirty_files = ?,"
            " ignored_tracked_files = ? WHERE id = 1",
            (head, json.dumps(sorted(dirty)), json.dumps(sorted(ignored))),
        )

    def definitions(self, name: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT files.path, files.digest, definitions.record"
            " FROM definitions JOIN files USING(path)"
            " WHERE name = ? ORDER BY files.path, definitions.ordinal",
            (name,),
        ).fetchall()

    def definitions_without_vectors(self, profile: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT files.path, definitions.record"
            " FROM definitions JOIN files USING(path)"
            " LEFT JOIN code_vectors ON code_vectors.definition = definitions.record"
            " AND code_vectors.profile = ? WHERE code_vectors.definition IS NULL"
            " ORDER BY files.path, definitions.ordinal LIMIT 33",
            (profile,),
        ).fetchall()

    def publish_vector(self, path: bytes, profile: str, definition: str, owner: str) -> None:
        self.db.execute(
            "INSERT INTO code_vectors VALUES (?, ?, ?, ?)", (path, profile, definition, owner)
        )

    def active_vector(self, index: str, key: str) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM objects WHERE index_name = ? AND record = ? AND state = 'active'",
                (index, key),
            ).fetchone()
            is not None
        )

    def code_vector_count(self, profile: str) -> int:
        return int(
            self.db.execute(
                "SELECT count(*) FROM code_vectors WHERE profile = ?", (profile,)
            ).fetchone()[0]
        )

    def retire_code_profile(self, profile: str) -> None:
        self.db.execute(
            "UPDATE objects SET state = 'retired', deadline = 0 WHERE owner IN"
            " (SELECT owner FROM code_vectors WHERE profile = ?) AND state = 'active'",
            (profile,),
        )
        self.db.execute("DELETE FROM code_vectors WHERE profile = ?", (profile,))

    def document(self, *, ref: str = "", topic: str = "") -> sqlite3.Row | None:
        condition, value = ("ref", ref) if ref else ("topic", topic)
        return cast(
            "sqlite3.Row | None",
            self.db.execute(f"SELECT * FROM documents WHERE {condition} = ?", (value,)).fetchone(),
        )

    def exact_document(self, alias: str) -> sqlite3.Row | None:
        return cast(
            "sqlite3.Row | None",
            self.db.execute(
                "SELECT documents.* FROM aliases JOIN documents ON documents.ref = aliases.document"
                " WHERE alias = ?",
                (alias,),
            ).fetchone(),
        )

    def remove_document(self, document: sqlite3.Row) -> None:
        self.retire(document["ref"])
        self.db.execute("DELETE FROM documents WHERE ref = ?", (document["ref"],))

    def publish_document(
        self,
        topic: str,
        ref: str,
        sources: list[dict[str, str]],
        expires: float | None,
        aliases: set[str],
    ) -> None:
        previous = self.document(topic=topic)
        if previous:
            self.remove_document(previous)
        self.db.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?)", (topic, ref, json.dumps(sources), expires)
        )
        self.db.executemany(
            "INSERT OR REPLACE INTO aliases VALUES (?, ?)", [(alias, ref) for alias in aliases]
        )

    def retire_expired_documents(self) -> None:
        expired = self.db.execute(
            "SELECT * FROM documents WHERE expires <= ? LIMIT 64", (time.time(),)
        ).fetchall()
        if expired:
            with self.transaction(write=True):
                for entry in expired:
                    self.remove_document(entry)
                self.db.execute("UPDATE state SET generation = generation + 1 WHERE id = 1")
