"""Worktree-scoped exact symbol index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from kachedb import KacheClient

from ..config import settings
from .catalog import Catalog, Record, State
from .parser import CodeParser
from .worktree import Observation, Worktree, WorktreeError

if TYPE_CHECKING:
    from collections.abc import Iterator

ARG_BYTES = 2 * 1024 * 1024
# A conservative staging batch fits the RESP, AOF, argument count, and key limits.
KV_BATCH_BYTES = 4 * 1024 * 1024
KV_BATCH_COUNT = 128


@dataclass
class FileUpdate:
    path: str
    eligible: bool
    digest: str | None
    owner: str | None
    definitions: list[tuple[str, str, int]]
    values: dict[str, str]
    parse_failed: bool


def checked_value(value: str) -> str:
    if len(value.encode("utf-8")) > ARG_BYTES:
        raise WorktreeError("PAYLOAD_TOO_LARGE", "An encoded record exceeds 2 MiB.")
    return value


@lru_cache(maxsize=16)
def _supports_atomic_mset(client: KacheClient, time_bucket: int) -> bool:
    """Probe once per pooled endpoint and time bucket."""
    info = client.info("server")
    version = re.search(r"(?m)^kachedb_version:(\d+)\.(\d+)\.(\d+)", info)
    return bool(version and tuple(map(int, version.groups())) >= (0, 5, 0))


class ScopedIndex:
    def __init__(self, worktree: Worktree, catalog: Catalog, client: KacheClient) -> None:
        self.worktree = worktree
        self.catalog = catalog
        self.client = client

    @contextmanager
    def connection(self) -> Iterator[KacheClient]:
        remaining = self.worktree.remaining()
        if remaining >= 1.0:
            yield self.client
            return
        client = KacheClient(
            host=self.client.host,
            port=self.client.port,
            max_connections=1,
            socket_timeout=remaining,
        )
        try:
            yield client
        finally:
            client.disconnect_all()

    def key(self, kind: str, revision: str) -> str:
        return f"svam:{self.catalog.namespace}:{kind}:{revision}"

    def stage(self, attempt: str, records: list[Record]) -> None:
        self.catalog.stage(attempt, records, time.time() + self.worktree.remaining())

    def cleanup(self) -> None:
        self.catalog.retire_expired_documents()
        for record in self.catalog.garbage():
            # Leave time for the caller's result. Maintenance resumes on the next scoped operation.
            if self.worktree.remaining() < 0.25:
                break
            with self.connection() as client:
                if record.index:
                    client.vdel(record.index, record.key)
                else:
                    client.delete(record.key)
            self.catalog.collected(record)

    def put_values(self, values: dict[str, str]) -> None:
        if not values:
            return
        supported = False
        if len(values) > 1:
            with self.connection() as client:
                supported = _supports_atomic_mset(client, int(time.monotonic()) // 60)
        if not supported:
            for key, value in values.items():
                with self.connection() as client:
                    if not client.set(key, value):
                        raise WorktreeError("STORAGE_ERROR", "KacheDB rejected a staged record.")
            return
        batch: dict[str | bytes, str | bytes] = {}
        size = 32
        for key, value in values.items():
            encoded = len(key.encode()) + len(value.encode()) + 32
            if size + encoded > KV_BATCH_BYTES or len(batch) == KV_BATCH_COUNT:
                self._put_batch(batch)
                batch = {}
                size = 32
            batch[key] = value
            size += encoded
        self._put_batch(batch)

    def _put_batch(self, batch: dict[str | bytes, str | bytes]) -> None:
        try:
            with self.connection() as client:
                if not client.mset(batch):
                    raise WorktreeError("STORAGE_ERROR", "KacheDB rejected a staged batch.")
        except Exception:
            _supports_atomic_mset.cache_clear()
            raise

    def get_values(self, keys: list[str]) -> list[str | bytes | None]:
        values: list[str | bytes | None] = []
        for offset in range(0, len(keys), 32):
            batch = keys[offset : offset + 32]
            with self.connection() as client:
                block = client.mget(*batch)
            if len(block) != len(batch):
                raise WorktreeError("STORAGE_ERROR", "KacheDB returned an incomplete batch.")
            values.extend(block)
        return values

    def _candidate_paths(
        self,
        before: State,
        observed: Observation,
        files: dict[str, tuple[str | None, str | None]],
        force: bool,
    ) -> set[str]:
        if force or not before.indexed or bool(before.head) != bool(observed.head):
            return set(files) | observed.eligible
        diff: set[str] | None = set()
        if before.head and observed.head and before.head != observed.head:
            diff = self.worktree.changes_since(before.head, observed.head)
        if diff is None:
            return set(files) | observed.eligible
        dirty_changes = diff | observed.dirty | before.dirty
        return (dirty_changes & (files.keys() | observed.eligible)) | (
            files.keys() ^ observed.eligible
        )

    def _read_code(self, path: str) -> bytes | None:
        try:
            return self.worktree.read(path)
        except FileNotFoundError:
            return None
        except WorktreeError as exc:
            if exc.status == "SOURCE_PATH_BLOCKED":
                return None
            raise

    def _definitions(
        self, parser: CodeParser, payload: bytes, path: str, digest: str, owner: str
    ) -> tuple[list[tuple[str, str, int]], dict[str, str]]:
        language = parser.detect_language(path)
        assert language is not None
        try:
            symbols = parser.parse_source(payload, language, path, strict=True)
        except Exception as exc:
            raise WorktreeError("PARSE_ERROR", "Could not parse the current source.", path) from exc
        definitions = []
        values = {}
        for ordinal, symbol in enumerate(symbols):
            key = self.key("symbol", f"{owner}:{ordinal}")
            definition = symbol.to_definition_dict()
            definition["sha256"] = digest
            value = json.dumps(definition, separators=(",", ":"))
            # Keep the required definition fields even when a docstring alone is oversized.
            if len(value.encode()) > ARG_BYTES and "doc" in definition:
                del definition["doc"]
                value = json.dumps(definition, separators=(",", ":"))
            values[key] = checked_value(value)
            definitions.append((symbol.name, key, ordinal))
        return definitions, values

    def refresh(
        self, *, force: bool = False, allow_update: bool = True, repair: set[str] | None = None
    ) -> dict[str, Any]:
        try:
            return self._refresh(force=force, allow_update=allow_update, repair=repair or set())
        except WorktreeError as exc:
            if exc.status not in ("RETRY", "WORKTREE_CHANGED"):
                raise
        # Retry once from a new Git observation and a new catalog snapshot.
        try:
            self.worktree.remaining()
            return self._refresh(force=force, allow_update=allow_update, repair=repair or set())
        except WorktreeError as exc:
            if exc.status == "RETRY":
                raise WorktreeError("INCOMPLETE", "The catalog changed during refresh.") from exc
            raise

    def _refresh(self, *, force: bool, allow_update: bool, repair: set[str]) -> dict[str, Any]:
        with self.catalog.transaction():
            before, files = self.catalog.state(), self.catalog.files()
        clean_inventory = (
            (before.head, files.keys() | before.ignored)
            if before.indexed and not before.dirty and not force
            else None
        )
        observed = self.worktree.observe(clean_inventory)
        if not before.indexed and not allow_update:
            return {"status": "INDEX_REQUIRED", "current_head": observed.head}
        candidates = self._candidate_paths(before, observed, files, force) | repair
        updates: list[FileUpdate] = []
        parser = CodeParser()
        for path in sorted(candidates):
            eligible = path in observed.eligible
            payload = self._read_code(path) if eligible else None
            digest = hashlib.sha256(payload).hexdigest() if payload is not None else None
            if (
                eligible
                and path in files
                and digest == files[path][0]
                and not force
                and path not in repair
            ):
                continue
            if not allow_update:
                return {"status": "INDEX_REQUIRED", "current_head": observed.head, "path": path}
            owner = uuid4().hex if payload is not None else None
            definitions: list[tuple[str, str, int]] = []
            values: dict[str, str] = {}
            parse_failed = False
            if payload is not None and digest is not None and owner is not None:
                try:
                    definitions, values = self._definitions(parser, payload, path, digest, owner)
                except WorktreeError as exc:
                    if exc.status != "PARSE_ERROR":
                        raise
                    # Retire prior definitions instead of claiming they still match this file.
                    parse_failed = True
                    owner = None
            updates.append(
                FileUpdate(path, eligible, digest, owner, definitions, values, parse_failed)
            )
        changed = (
            not before.indexed
            or before.dirty != observed.dirty
            or before.head != observed.head
            or before.ignored != observed.ignored
        )
        if not (updates or changed):
            if self.catalog.state().generation != before.generation:
                raise WorktreeError("RETRY", "The catalog changed during refresh.")
            return self._index_result("UP_TO_DATE", before, observed)
        if not allow_update:
            return {"status": "INDEX_REQUIRED", "current_head": observed.head}

        attempt = uuid4().hex
        values = {key: value for change in updates for key, value in change.values.items()}
        records = [
            Record(change.owner, key) for change in updates if change.owner for key in change.values
        ]
        self.stage(attempt, records)
        try:
            self.put_values(values)
            if self.worktree.observe(clean_inventory) != observed:
                raise WorktreeError("WORKTREE_CHANGED", "The worktree changed during indexing.")
            for change in updates:
                if change.eligible and self.worktree.digest(change.path) != change.digest:
                    raise WorktreeError(
                        "WORKTREE_CHANGED", "The source changed during indexing.", change.path
                    )
            self.worktree.remaining()
            with self.catalog.publish(before.generation, attempt):
                for change in updates:
                    self.catalog.replace_file(
                        change.path,
                        change.digest,
                        change.owner,
                        change.definitions,
                        eligible=change.eligible,
                        parse_failed=change.parse_failed,
                    )
                self.catalog.publish_worktree(observed.head, observed.dirty, observed.ignored)
        except Exception:
            self.catalog.abandon(attempt)
            raise
        after = State(
            before.generation + 1, True, observed.head, set(observed.dirty), set(observed.ignored)
        )
        return self._index_result("UPDATED", after, observed, refreshed_files=len(updates))

    def _index_result(
        self, status: str, state: State, observed: Observation, *, refreshed_files: int = 0
    ) -> dict[str, Any]:
        return {
            "status": status,
            "generation": state.generation,
            "indexed_head": state.head,
            "current_head": observed.head,
            "refreshed_files": refreshed_files,
            **self.catalog.parse_warnings(),
        }

    def explore(self, symbol: str) -> dict[str, Any]:
        refresh = self.refresh(allow_update=settings.auto_index)
        if refresh["status"] == "INDEX_REQUIRED":
            return refresh
        for _ in range(2):
            with self.catalog.transaction():
                generation = self.catalog.state().generation
                rows = self.catalog.definitions(symbol)
                warnings = self.catalog.parse_warnings()
            values = self.get_values([row["record"] for row in rows])
            verified: dict[str, str | None] = {}
            definitions = []
            repair = set()
            for row, value in zip(rows, values, strict=True):
                path = os.fsdecode(row["path"])
                if path not in verified:
                    verified[path] = self.worktree.digest(path)
                if value is None or verified[path] != row["digest"]:
                    repair.add(path)
                else:
                    definitions.append(json.loads(value))
            if not repair and generation == self.catalog.state().generation:
                return {
                    "status": "OK" if definitions else ("INCOMPLETE" if warnings else "NOT_FOUND"),
                    "symbol": symbol,
                    "generation": generation,
                    "definition_count": len(definitions),
                    "definitions": definitions,
                    **({"freshness": "verified"} if definitions else {}),
                    **({"reason": "unparsed_files"} if warnings and not definitions else {}),
                    **warnings,
                }
            refreshed = self.refresh(allow_update=settings.auto_index, repair=repair)
            if refreshed["status"] == "INDEX_REQUIRED":
                return refreshed
        raise WorktreeError(
            "INCOMPLETE", "Could not verify the current symbol records.", next(iter(repair), None)
        )

    def status(self, auto_index: bool) -> dict[str, Any]:
        allow_update = auto_index and settings.auto_index
        result = self.refresh(allow_update=allow_update)
        state = self.catalog.state()
        status = result["status"]
        return {
            **result,
            "generation": state.generation,
            "indexed_head": state.head,
            "index_state": (
                "INDEX_REQUIRED"
                if status == "INDEX_REQUIRED"
                else "PARTIAL"
                if result.get("unparsed_file_count")
                else "CURRENT"
            ),
        }
