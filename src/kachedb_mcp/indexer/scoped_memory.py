"""Source-backed documents and vector retrieval for one worktree scope."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import settings
from .catalog import Record
from .scoped_index import ScopedIndex, checked_value
from .worktree import SOURCE_LIMIT, WorktreeError

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable

    from kachedb import KacheClient

    from .catalog import Catalog
    from .worktree import Worktree

MAX_SOURCES = 32
MAX_CANDIDATES = 100
VECTOR_SEARCH_RESERVE = 1.0
VECTOR_PUBLISH_RESERVE = 0.25


class SourceInput(BaseModel):
    """A source file and the optional SHA-256 captured when the caller read it."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str | None = None


@dataclass(frozen=True)
class Embeddings:
    identity: str
    query: Callable[[str], list[float]]
    document: Callable[[str], list[float]]
    chunks: Callable[[str, str], list[tuple[str, str]]]


@dataclass(frozen=True)
class Vector:
    record: Record
    embedding: list[float]
    payload: str
    parent: str | None = None


def _alias(text: str) -> str:
    return text.strip().rstrip("?.!").casefold().strip()


def _text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class ScopedMemory(ScopedIndex):
    def __init__(
        self,
        worktree: Worktree,
        catalog: Catalog,
        client: KacheClient,
        embedding_factory: Callable[[], Embeddings],
    ) -> None:
        super().__init__(worktree, catalog, client)
        self._embedding_factory = embedding_factory
        self._digests: dict[str, tuple[str | None, str]] = {}
        self._source_bytes = SOURCE_LIMIT
        self._documents: dict[str, dict[str, Any] | None] = {}

    def _index(self, embeddings: Embeddings, dimension: int) -> str:
        profile = hashlib.sha256(f"{embeddings.identity}:{dimension}".encode()).hexdigest()[:32]
        return self.key("flat", profile)

    def _write_vectors(
        self, vectors: list[Vector], expires: float | None = None, tags: list[str] | None = None
    ) -> None:
        for item in vectors:
            ttl = self._ttl(expires)
            with self.connection() as client:
                if not client.vadd(
                    item.record.index,
                    item.record.key,
                    item.embedding,
                    payload=item.payload,
                    ex=ttl,
                    tags=tags,
                    parent_key=item.parent,
                ):
                    raise WorktreeError("STORAGE_ERROR", "KacheDB rejected a staged vector.")

    @staticmethod
    def _ttl(expires: float | None) -> int | None:
        if expires is None:
            return None
        ttl = math.ceil(expires - time.time())
        if ttl <= 0:
            raise WorktreeError("INCOMPLETE", "The document expired before publication.")
        return ttl

    def _capture(self, inputs: list[SourceInput]) -> list[dict[str, str]]:
        if len(inputs) > MAX_SOURCES:
            raise WorktreeError("INVALID_SOURCES", "At most 32 source paths are allowed.")
        sources: dict[str, str] = {}
        remaining = SOURCE_LIMIT
        for item in inputs:
            try:
                source = SourceInput.model_validate(item)
            except ValidationError as exc:
                raise WorktreeError(
                    "INVALID_SOURCES", "A source requires a path and optional SHA-256."
                ) from exc
            digest = sources.get(source.path)
            if digest is None:
                try:
                    content = self.worktree.read(source.path, remaining, declared=True)
                except FileNotFoundError as exc:
                    raise WorktreeError(
                        "SOURCE_MISSING", "The declared source is missing.", source.path
                    ) from exc
                remaining -= len(content)
                digest = hashlib.sha256(content).hexdigest()
                sources[source.path] = digest
            if source.sha256 is not None and source.sha256.lower() != digest:
                raise WorktreeError(
                    "SOURCE_CHANGED", "The supplied fingerprint does not match.", source.path
                )
        return [{"path": path, "sha256": digest} for path, digest in sources.items()]

    def _fingerprint(self, path: str) -> tuple[str | None, str]:
        if path not in self._digests:
            try:
                contents = self.worktree.read(path, self._source_bytes, declared=True)
            except FileNotFoundError:
                result: tuple[str | None, str] = (None, "source_missing")
            except WorktreeError as exc:
                result = (
                    None,
                    {
                        "SOURCE_PATH_BLOCKED": "source_path_blocked",
                        "SOURCE_TOO_LARGE": "verification_budget",
                        "INCOMPLETE": "verification_budget",
                        "WORKTREE_CHANGED": "source_changed_during_read",
                    }.get(exc.status, "source_unreadable"),
                )
            else:
                self._source_bytes -= len(contents)
                result = (hashlib.sha256(contents).hexdigest(), "")
            self._digests[path] = result
        return self._digests[path]

    def _provenance(self, sources: list[dict[str, str]]) -> dict[str, Any]:
        if not sources:
            return {"freshness": "unverified", "reason": "missing_provenance", "sources": []}
        checks = []
        for source in sources:
            digest, reason = self._fingerprint(source["path"])
            if reason == "source_missing" or (digest is not None and digest != source["sha256"]):
                freshness = "stale"
                reason = reason or "source_changed"
            else:
                freshness = "verified" if digest is not None else "unverified"
            checks.append(
                {**source, "freshness": freshness, **({"reason": reason} if reason else {})}
            )
        for freshness in ("stale", "unverified"):
            issue = next((check for check in checks if check["freshness"] == freshness), None)
            if issue:
                return {"freshness": freshness, "reason": issue["reason"], "sources": checks}
        return {"freshness": "verified", "sources": checks}

    def save(
        self,
        topic: str,
        content: str,
        ttl_seconds: int,
        tags: list[str] | None,
        auto_chunk: bool,
        prompt_alias: str | None,
        sources: list[SourceInput] | None,
    ) -> dict[str, Any]:
        if not topic.strip() or not 0 <= ttl_seconds < 2**31:
            raise WorktreeError("INVALID_INPUT", "Use a nonempty topic and a valid TTL.")
        topic = topic.strip()
        checked_value(content)
        captured = self._capture(sources or [])
        self.worktree.remaining()
        embeddings = self._embedding_factory()
        parts = embeddings.chunks(topic, content) if auto_chunk else [(topic, content)]
        prepared = []
        for title, chunk in parts:
            self.worktree.remaining()
            prepared.append(
                (embeddings.document(f"{title}\n\n{chunk[:6000]}"), checked_value(chunk))
            )
        if prompt_alias and prompt_alias.strip():
            self.worktree.remaining()
            prepared.append((embeddings.query(prompt_alias.strip()), content[:400]))
        dimension = len(prepared[0][0])
        if not dimension or any(len(embedding) != dimension for embedding, _ in prepared):
            raise WorktreeError(
                "EMBEDDING_ERROR", "The embedding provider returned inconsistent dimensions."
            )
        index = self._index(embeddings, dimension)
        aliases = {_alias(alias) for alias in (topic, prompt_alias or "")} - {""}

        for _ in range(2):
            self.worktree.remaining()
            before = self.catalog.state()
            ref = uuid4().hex
            expires = time.time() + ttl_seconds if ttl_seconds else None
            key = self.key("document", ref)
            vectors = [
                Vector(Record(ref, f"d:{ref}:{ordinal}", index), embedding, payload, ref)
                for ordinal, (embedding, payload) in enumerate(prepared)
            ]
            self.stage(ref, [Record(ref, key), *(vector.record for vector in vectors)])
            try:
                with self.connection() as client:
                    if not client.set(key, content, ex=self._ttl(expires)):
                        raise WorktreeError(
                            "STORAGE_ERROR", "KacheDB rejected the document payload."
                        )
                self._write_vectors(vectors, expires, tags)
                checked = self._capture([SourceInput(**source) for source in captured])
                self.worktree.remaining()
                self._ttl(expires)
                with self.catalog.publish(before.generation, ref):
                    self.catalog.publish_document(topic, ref, checked, expires, aliases)
                return {"status": "SAVED", "topic": topic, "revision": ref, "parent_ref": ref}
            except Exception as exc:
                self.catalog.abandon(ref)
                if not isinstance(exc, WorktreeError) or exc.status != "RETRY":
                    raise
        raise WorktreeError("INCOMPLETE", "Another operation changed the catalog during save.")

    def _live_doc(
        self,
        row: sqlite3.Row | None,
        *,
        require_verified: bool,
        payload: str | bytes | None = None,
        parent: bool = True,
    ) -> dict[str, Any] | None:
        if row is None or (row["expires"] is not None and row["expires"] <= time.time()):
            return None
        ref = row["ref"]
        needs_body = parent or payload is None
        cached = self._documents.get(ref)
        if ref not in self._documents or (
            needs_body and cached is not None and "content" not in cached
        ):
            with self.connection() as client:
                content = client.get(self.key("document", ref))
            self._documents[ref] = (
                {
                    "kind": "context",
                    "topic": row["topic"],
                    "revision": ref,
                    "parent_ref": ref,
                    **({"content": _text(content)} if needs_body else {}),
                    **self._provenance(json.loads(row["sources"])),
                }
                if content is not None
                else None
            )
        document = self._documents[ref]
        if row["expires"] is not None and row["expires"] <= time.time():
            return None
        if document is None or (require_verified and document["freshness"] != "verified"):
            return None
        return document if parent or payload is None else {**document, "content": _text(payload)}

    def parent(self, ref: str, require_verified: bool) -> dict[str, Any]:
        generation = self.catalog.state().generation
        match = self._live_doc(
            self.catalog.document(ref=ref) if ref else None, require_verified=require_verified
        )
        if generation != self.catalog.state().generation:
            raise WorktreeError("INCOMPLETE", "The catalog changed during parent retrieval.")
        return {"status": "OK", **match} if match else {"status": "NOT_FOUND"}

    def delete(self, key: str) -> dict[str, Any]:
        with self.catalog.transaction(write=True):
            row = self.catalog.document(ref=key) if key else None
            if row is None:
                row = self.catalog.document(topic=key)
            if row is None:
                return {"status": "NOT_FOUND"}
            self.catalog.remove_document(row)
            self.catalog.db.execute("UPDATE state SET generation = generation + 1 WHERE id = 1")
        return {"status": "DELETED", "revision": row["ref"]}

    def _backfill(self, embeddings: Embeddings, index: str) -> bool:
        """Publish one bounded profile batch, returning whether more remains."""
        with self.catalog.transaction():
            generation = self.catalog.state().generation
            rows = self.catalog.definitions_without_vectors(index)
        if not rows:
            return False

        rows, more = rows[:32], len(rows) > 32
        values = self.get_values([row["record"] for row in rows])
        missing = {
            os.fsdecode(row["path"])
            for row, value in zip(rows, values, strict=True)
            if value is None
        }
        if missing:
            self.refresh(allow_update=settings.auto_index, repair=missing)
            return True

        attempt = uuid4().hex
        vectors = []
        longest_embedding = 0.0
        for ordinal, value in enumerate(values):
            if self.worktree.remaining() <= longest_embedding + VECTOR_PUBLISH_RESERVE:
                break
            assert value is not None
            definition = json.loads(value)
            text = (
                f"{definition['kind']} {definition['name']} ({definition['language']}) "
                f"in {definition['file']}: {definition['signature']}"
            )
            if definition.get("doc"):
                text += f"\n{definition['doc']}"
            started = time.monotonic()
            embedding = embeddings.document(text[:6000])
            longest_embedding = max(longest_embedding, time.monotonic() - started)
            if self.worktree.remaining() <= VECTOR_PUBLISH_RESERVE:
                break
            owner = f"{attempt}:{ordinal}"
            vector = Record(owner, f"c:{owner}", index)
            vectors.append(Vector(vector, embedding, _text(value)))
        if not vectors:
            return True
        self.stage(attempt, [item.record for item in vectors])
        try:
            self._write_vectors(vectors)
            self.worktree.remaining()
            with self.catalog.publish(generation, attempt):
                for row, item in zip(rows[: len(vectors)], vectors, strict=True):
                    self.catalog.publish_vector(
                        row["path"], index, row["record"], item.record.owner
                    )
        except Exception as exc:
            self.catalog.abandon(attempt)
            if isinstance(exc, WorktreeError) and exc.status == "RETRY":
                return True
            raise
        return more or len(vectors) < len(rows)

    def _repair_vector_profile(self, index: str) -> None:
        with self.catalog.transaction():
            generation = self.catalog.state().generation
            expected = self.catalog.code_vector_count(index)
        if expected:
            with self.connection() as client:
                stats = client.vstats(index)
            if not stats or stats["active_vectors"] < expected:
                with self.catalog.publish(generation):
                    self.catalog.retire_code_profile(index)

    def search(
        self,
        query: str,
        top_k: int,
        threshold: float,
        tags: list[str] | None,
        exact_first: bool,
        resolve_parent: bool,
        require_verified: bool,
    ) -> dict[str, Any]:
        if (
            not 1 <= top_k <= MAX_CANDIDATES
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise WorktreeError(
                "INVALID_INPUT", "Use top_k from 1 to 100 and a threshold from 0 to 1."
            )
        # Exact aliases carry no filter metadata. Tagged queries start at the vector index.
        if exact_first and not tags:
            generation = self.catalog.state().generation
            match = self._live_doc(
                self.catalog.exact_document(_alias(query)), require_verified=require_verified
            )
            if match and generation == self.catalog.state().generation:
                return {"status": "OK", "matches": [{**match, "score": 1.0, "match": "exact"}]}

        embeddings = self._embedding_factory()
        self.worktree.remaining()
        vector = embeddings.query(query)
        index = self._index(embeddings, len(vector))

        def retrieve(*, include_code: bool) -> dict[str, Any]:
            return self._search_vectors(
                index,
                vector,
                top_k,
                threshold,
                tags,
                require_verified,
                resolve_parent,
                include_code=include_code,
            )

        result = None
        code_incomplete = True
        warnings: dict[str, Any] = {}
        deadline = self.worktree.deadline
        code_deadline = deadline - VECTOR_SEARCH_RESERVE
        try:
            self.worktree.deadline = code_deadline
            refresh = self.refresh(allow_update=settings.auto_index)
            if refresh["status"] != "INDEX_REQUIRED":
                warnings = {
                    key: refresh[key]
                    for key in ("unparsed_file_count", "unparsed_files")
                    if key in refresh
                }
                self._repair_vector_profile(index)
                code_incomplete = bool(self.catalog.definitions_without_vectors(index))
                if code_incomplete:
                    # A synchronous embedding can overrun its budget, so retrieve saved
                    # context before doing any code embeddings for an incomplete profile.
                    self.worktree.deadline = deadline
                    result = retrieve(include_code=False)
                    self.worktree.deadline = code_deadline
                    for _ in range(4):
                        batch_started = time.monotonic()
                        code_incomplete = self._backfill(embeddings, index)
                        if not code_incomplete or self.worktree.remaining() < max(
                            1.0, time.monotonic() - batch_started
                        ):
                            break
        except WorktreeError as exc:
            if exc.status not in (
                "INCOMPLETE",
                "WORKTREE_CHANGED",
                "PARSE_ERROR",
                "SOURCE_UNREADABLE",
                "RETRY",
            ):
                raise
            code_incomplete = True
        finally:
            self.worktree.deadline = deadline

        if result is None or not code_incomplete:
            try:
                result = retrieve(include_code=not code_incomplete)
            except WorktreeError as exc:
                if result is None or exc.status != "INCOMPLETE":
                    raise
                code_incomplete = True
        if code_incomplete:
            result["status"] = "INCOMPLETE"
            result.setdefault("reason", "code_index_pending")
        elif warnings and result["status"] == "NO_MATCH":
            result.update(status="INCOMPLETE", reason="unparsed_files")
        result.update(warnings)
        return result

    def _search_vectors(
        self,
        index: str,
        vector: list[float],
        top_k: int,
        threshold: float,
        tags: list[str] | None,
        require_verified: bool,
        resolve_parent: bool,
        *,
        include_code: bool,
    ) -> dict[str, Any]:
        generation = self.catalog.state().generation
        passes = [(threshold, tags)]
        if threshold > 0.55:
            passes.append((max(0.50, threshold - 0.20), tags))
        passes.append((min(0.30, threshold), tags))
        if tags:
            passes.append((threshold, None))
        matches: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        budget = MAX_CANDIDATES
        incomplete = False
        tags_relaxed = False
        observed_code: dict[str, str | None] = {}
        for cutoff, filters in dict.fromkeys((x, tuple(y or [])) for x, y in passes):
            if not budget or matches:
                break
            with self.connection() as client:
                candidates = client.vsearch(
                    index, vector, top_k=budget, threshold=cutoff, filter_tags=list(filters) or None
                )
            incomplete |= len(candidates) == budget
            budget -= len(candidates)
            for record in candidates:
                key = _text(record.id)
                if not self.catalog.active_vector(index, key):
                    continue
                if key.startswith("d:"):
                    ref = key.split(":")[1]
                    if resolve_parent:
                        if ref in seen_parents:
                            continue
                        seen_parents.add(ref)
                    match = self._live_doc(
                        self.catalog.document(ref=ref),
                        require_verified=require_verified,
                        payload=record.payload,
                        parent=resolve_parent,
                    )
                else:
                    match = None
                    if include_code and record.payload is not None:
                        definition = json.loads(record.payload)
                        path = definition["file"]
                        if path not in observed_code:
                            observed_code[path] = self.worktree.digest(path)
                        if observed_code[path] == definition["sha256"]:
                            match = {
                                "kind": "symbol",
                                "content": definition,
                                "freshness": "verified",
                                "sources": [{"path": path, "sha256": definition["sha256"]}],
                            }
                        else:
                            incomplete = True
                if match:
                    tags_relaxed |= bool(tags and not filters)
                    matches.append({**match, "score": record.score, "match": "semantic"})
                    if len(matches) == top_k:
                        break

        if generation != self.catalog.state().generation:
            raise WorktreeError("INCOMPLETE", "The catalog changed during semantic retrieval.")
        result: dict[str, Any] = {"matches": matches}
        if tags_relaxed:
            result["tag_filter_relaxed"] = True
        if incomplete and len(matches) < top_k:
            return {**result, "status": "INCOMPLETE", "reason": "candidate_limit_or_changed_source"}
        return {**result, "status": "OK" if matches else "NO_MATCH"}
