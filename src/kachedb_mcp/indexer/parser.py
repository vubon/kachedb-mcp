"""Tree-Sitter AST parsing and symbol extraction engine for KacheDB MCP."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Symbol:
    """Represents an extracted codebase symbol (function, class, struct, etc.)."""

    name: str
    kind: (
        str  # "function" | "method" | "class" | "struct" | "interface" | "trait" | "type" | "enum"
    )
    language: str
    file_path: str
    line_start: int  # 1-indexed
    line_end: int  # 1-indexed
    signature: str
    docstring: str | None = None
    parent_symbol: str | None = None

    def to_definition_dict(self) -> dict[str, Any]:
        """Compact dictionary representation for SwissTable multi-definition arrays."""
        d: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "language": self.language,
            "file": self.file_path,
            "line": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
        }
        if self.docstring:
            d["doc"] = self.docstring
        if self.parent_symbol:
            d["parent"] = self.parent_symbol
        return d

    def to_embedding_text(self) -> str:
        """Formatted text used for SIMD vector embedding generation."""
        header = f"{self.kind} {self.name} ({self.language}) in {self.file_path}: {self.signature}"
        parts = [header]
        if self.docstring:
            parts.append(self.docstring)
        return "\n".join(parts).strip()


class SymbolRegistry:
    """Aggregates extracted symbols and prepares SwissTable KV and vector payloads.

    Handles symbol collisions by grouping identically named symbols under an array:
    `sym:<workspace_id>:<symbol_name>` -> JSON array of all definitions across files.
    Also registers fully qualified keys:
    `sym:<workspace_id>:<file_rel_path>:<symbol_name>` -> exact single definition.
    """

    def __init__(self) -> None:
        self._symbols: list[Symbol] = []
        # Key: symbol_name -> list of definitions
        self._by_name: dict[str, list[dict[str, Any]]] = {}
        # Key: (file_path, symbol_name) -> definition
        self._by_file_name: dict[tuple[str, str], dict[str, Any]] = {}

    def add(self, symbol: Symbol) -> None:
        """Add a single symbol to the registry."""
        self._symbols.append(symbol)
        defn = symbol.to_definition_dict()
        self._by_name.setdefault(symbol.name, []).append(defn)
        self._by_file_name[(symbol.file_path, symbol.name)] = defn

    def add_all(self, symbols: list[Symbol]) -> None:
        """Add multiple symbols to the registry."""
        for sym in symbols:
            self.add(sym)

    @property
    def symbols(self) -> list[Symbol]:
        """All registered Symbol instances."""
        return self._symbols

    def get_definitions(self, symbol_name: str) -> list[dict[str, Any]]:
        """Retrieve all definition locations for a given symbol name."""
        return self._by_name.get(symbol_name, [])

    def get_exact_definition(self, file_path: str, symbol_name: str) -> dict[str, Any] | None:
        """Retrieve exact definition for a given file and symbol name."""
        return self._by_file_name.get((file_path, symbol_name))

    def to_swisstable_payload(self, workspace_id: str) -> dict[str, str]:
        """Generate key-value dictionary for SwissTable insertion.

        Returns:
            Dictionary of { "sym:<workspace>:<symbol>": json_array_string, ... }
        """
        payload: dict[str, str] = {}
        ws = workspace_id.lower().replace(" ", "-").replace("_", "-")

        # 1. Collision-handled symbol arrays
        for name, defs in self._by_name.items():
            key = f"sym:{ws}:{name}"
            payload[key] = json.dumps(defs, ensure_ascii=False)

        # 2. Fully qualified file-specific definitions
        for (file_path, name), defn in self._by_file_name.items():
            norm_path = file_path.replace("\\", "/").strip("/")
            key = f"sym:{ws}:{norm_path}:{name}"
            payload[key] = json.dumps([defn], ensure_ascii=False)

        return payload

    def to_vector_records(self, workspace_id: str) -> list[tuple[str, str, dict[str, Any]]]:
        """Generate records for vector indexing.

        Returns:
            List of (vector_key, embedding_text, metadata_dict)
        """
        records: list[tuple[str, str, dict[str, Any]]] = []
        ws = workspace_id.lower().replace(" ", "-").replace("_", "-")

        for sym in self._symbols:
            norm_path = sym.file_path.replace("\\", "/").strip("/")
            vector_key = f"vec:{ws}:{norm_path}:{sym.name}:{sym.line_start}"
            text = sym.to_embedding_text()
            metadata: dict[str, Any] = {
                "workspace_id": ws,
                "symbol": sym.name,
                "kind": sym.kind,
                "language": sym.language,
                "file": sym.file_path,
                "line": sym.line_start,
                "line_end": sym.line_end,
                "signature": sym.signature,
            }
            records.append((vector_key, text, metadata))

        return records


# =========================================================================
# Language AST Extractors
# =========================================================================


class BaseExtractor(ABC):
    """Abstract base class for language-specific AST symbol extraction."""

    def __init__(self, language_name: str) -> None:
        self.language_name = language_name

    @abstractmethod
    def extract(self, root: Node, code_bytes: bytes, file_path: str) -> list[Symbol]:
        """Traverse AST and return all extracted symbols."""

    @staticmethod
    def _text(node: Node, code_bytes: bytes) -> str:
        """Decode slice of code bytes corresponding to node."""
        return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _extract_preceding_comments(self, node: Node, code_bytes: bytes) -> str | None:
        """Extract continuous comment lines directly preceding a node."""
        comments: list[str] = []
        curr = node.prev_sibling
        while curr is not None and curr.type in (
            "comment",
            "line_comment",
            "block_comment",
            "attribute_item",
            "attribute",
            "decorator",
        ):
            if curr.type in ("comment", "line_comment", "block_comment"):
                text = self._text(curr, code_bytes).strip()
                # Clean comment syntax (//, /*, *, #)
                cleaned = re.sub(r"^(\/{2,3}|\/\*+|\*+|\#)\s?", "", text).rstrip("*/").strip()
                if cleaned:
                    comments.insert(0, cleaned)
            curr = curr.prev_sibling

        return "\n".join(comments).strip() if comments else None


class PythonExtractor(BaseExtractor):
    """Extracts classes, functions, methods, async defs, and docstrings from Python AST."""

    def __init__(self) -> None:
        super().__init__("python")

    def extract(self, root: Node, code_bytes: bytes, file_path: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        self._walk(root, code_bytes, file_path, symbols, parent_class=None)
        return symbols

    def _walk(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None = None,
    ) -> None:
        for child in node.children:
            if child.type == "function_definition":
                self._handle_function(child, code_bytes, file_path, symbols, parent_class)
            elif child.type == "class_definition":
                self._handle_class(child, code_bytes, file_path, symbols)
            elif child.type == "decorated_definition":
                inner = child.child_by_field_name("definition")
                if inner is not None:
                    if inner.type == "function_definition":
                        self._handle_function(inner, code_bytes, file_path, symbols, parent_class)
                    elif inner.type == "class_definition":
                        self._handle_class(inner, code_bytes, file_path, symbols)
            else:
                if child.type not in ("block", "module"):
                    self._walk(child, code_bytes, file_path, symbols, parent_class)

    def _handle_function(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        kind = "method" if parent_class else "function"

        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = (
            code_bytes[node.start_byte : sig_end]
            .decode("utf-8", errors="replace")
            .strip()
            .rstrip(":")
        )
        signature = re.sub(r"\s+", " ", raw_sig)
        docstring = self._extract_docstring(body_node, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind=kind,
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=parent_class,
            )
        )

    def _handle_class(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = (
            code_bytes[node.start_byte : sig_end]
            .decode("utf-8", errors="replace")
            .strip()
            .rstrip(":")
        )
        signature = re.sub(r"\s+", " ", raw_sig)
        docstring = self._extract_docstring(body_node, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind="class",
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=None,
            )
        )

        if body_node:
            for child in body_node.children:
                if child.type == "function_definition":
                    self._handle_function(child, code_bytes, file_path, symbols, parent_class=name)
                elif child.type == "decorated_definition":
                    inner = child.child_by_field_name("definition")
                    if inner and inner.type == "function_definition":
                        self._handle_function(
                            inner,
                            code_bytes,
                            file_path,
                            symbols,
                            parent_class=name,
                        )

    def _extract_docstring(self, body_node: Node | None, code_bytes: bytes) -> str | None:
        """Extract first expression statement string literal in Python block."""
        if not body_node or not body_node.children:
            return None

        first = body_node.children[0]
        if first.type == "expression_statement" and first.children:
            expr = first.children[0]
            if expr.type == "string":
                text = self._text(expr, code_bytes)
                stripped = text.strip("\"' \n\t")
                return stripped if stripped else None
        return None


class RustExtractor(BaseExtractor):
    """Extracts functions, structs, enums, traits, and impl blocks from Rust AST."""

    def __init__(self) -> None:
        super().__init__("rust")

    def extract(self, root: Node, code_bytes: bytes, file_path: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        self._walk(root, code_bytes, file_path, symbols, parent_impl=None)
        return symbols

    def _walk(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        parent_impl: str | None = None,
    ) -> None:
        for child in node.children:
            if child.type in ("function_item", "function_signature_item"):
                self._handle_function(child, code_bytes, file_path, symbols, parent_impl)
            elif child.type == "struct_item":
                self._handle_struct_or_enum(child, code_bytes, file_path, symbols, kind="struct")
            elif child.type == "enum_item":
                self._handle_struct_or_enum(child, code_bytes, file_path, symbols, kind="enum")
            elif child.type == "trait_item":
                self._handle_trait(child, code_bytes, file_path, symbols)
            elif child.type == "impl_item":
                self._handle_impl(child, code_bytes, file_path, symbols)
            elif child.type == "macro_definition":
                self._handle_macro(child, code_bytes, file_path, symbols)
            elif child.type in ("source_file", "declaration_list"):
                self._walk(child, code_bytes, file_path, symbols, parent_impl)

    def _handle_function(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        parent_impl: str | None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        kind = "method" if parent_impl else "function"

        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = (
            code_bytes[node.start_byte : sig_end]
            .decode("utf-8", errors="replace")
            .strip()
            .rstrip(";")
        )
        signature = re.sub(r"\s+", " ", raw_sig)
        docstring = self._extract_preceding_comments(node, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind=kind,
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=parent_impl,
            )
        )

    def _handle_struct_or_enum(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        kind: str,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = code_bytes[node.start_byte : sig_end].decode("utf-8", errors="replace").strip()
        signature = re.sub(r"\s+", " ", raw_sig)
        docstring = self._extract_preceding_comments(node, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind=kind,
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=None,
            )
        )

    def _handle_trait(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = code_bytes[node.start_byte : sig_end].decode("utf-8", errors="replace").strip()
        signature = re.sub(r"\s+", " ", raw_sig)
        docstring = self._extract_preceding_comments(node, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind="trait",
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=None,
            )
        )

        if body_node:
            for child in body_node.children:
                if child.type in ("function_item", "function_signature_item"):
                    self._handle_function(child, code_bytes, file_path, symbols, parent_impl=name)

    def _handle_impl(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        type_node = node.child_by_field_name("type")
        impl_name = self._text(type_node, code_bytes) if type_node else None

        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type in ("function_item", "function_signature_item"):
                    self._handle_function(
                        child,
                        code_bytes,
                        file_path,
                        symbols,
                        parent_impl=impl_name,
                    )

    def _handle_macro(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = code_bytes[node.start_byte : sig_end].decode("utf-8", errors="replace").strip()
        signature = re.sub(r"\s+", " ", raw_sig)
        docstring = self._extract_preceding_comments(node, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind="macro",
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=None,
            )
        )


class GoExtractor(BaseExtractor):
    """Extracts structs, interfaces, functions, and methods from Go AST."""

    def __init__(self) -> None:
        super().__init__("go")

    def extract(self, root: Node, code_bytes: bytes, file_path: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        for child in root.children:
            if child.type == "function_declaration":
                self._handle_function(child, code_bytes, file_path, symbols)
            elif child.type == "method_declaration":
                self._handle_method(child, code_bytes, file_path, symbols)
            elif child.type == "type_declaration":
                self._handle_type(child, code_bytes, file_path, symbols)
        return symbols

    def _handle_function(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = code_bytes[node.start_byte : sig_end].decode("utf-8", errors="replace").strip()
        signature = re.sub(r"\s+", " ", raw_sig)
        docstring = self._extract_preceding_comments(node, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind="function",
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=None,
            )
        )

    def _handle_method(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        receiver_node = node.child_by_field_name("receiver")
        parent_name: str | None = None
        if receiver_node:
            rcv_text = self._text(receiver_node, code_bytes)
            m = re.search(r"\*?([A-Za-z0-9_]+)\s*\)?$", rcv_text.strip())
            if m:
                parent_name = m.group(1)

        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = code_bytes[node.start_byte : sig_end].decode("utf-8", errors="replace").strip()
        signature = re.sub(r"\s+", " ", raw_sig)
        docstring = self._extract_preceding_comments(node, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind="method",
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=parent_name,
            )
        )

    def _handle_type(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        for child in node.children:
            if child.type == "type_spec":
                name_node = child.child_by_field_name("name")
                type_node = child.child_by_field_name("type")
                if not name_node:
                    continue

                name = self._text(name_node, code_bytes)
                kind = "type"
                if type_node:
                    if type_node.type == "struct_type":
                        kind = "struct"
                    elif type_node.type == "interface_type":
                        kind = "interface"

                raw_sig = f"type {name} {kind}"
                docstring = self._extract_preceding_comments(node, code_bytes)

                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        language=self.language_name,
                        file_path=file_path,
                        line_start=child.start_point.row + 1,
                        line_end=child.end_point.row + 1,
                        signature=raw_sig,
                        docstring=docstring,
                        parent_symbol=None,
                    )
                )


class TypeScriptExtractor(BaseExtractor):
    """Extracts classes, interfaces, types, functions, and methods from TS/JS/TSX/JSX."""

    def __init__(self, language_name: str = "typescript") -> None:
        super().__init__(language_name)

    def extract(self, root: Node, code_bytes: bytes, file_path: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        self._walk(root, code_bytes, file_path, symbols, parent_class=None)
        return symbols

    def _walk(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None = None,
    ) -> None:
        for child in node.children:
            curr = child
            outer_node: Node | None = None
            if curr.type == "export_statement":
                outer_node = curr
                decl = curr.child_by_field_name("declaration")
                if decl is not None:
                    curr = decl

            if curr.type == "function_declaration":
                self._handle_function(
                    curr, code_bytes, file_path, symbols, parent_class, outer_node
                )
            elif curr.type == "class_declaration":
                self._handle_class(curr, code_bytes, file_path, symbols, outer_node)
            elif curr.type == "interface_declaration":
                self._handle_interface_or_type(
                    curr, code_bytes, file_path, symbols, kind="interface", outer_node=outer_node
                )
            elif curr.type == "type_alias_declaration":
                self._handle_interface_or_type(
                    curr, code_bytes, file_path, symbols, kind="type", outer_node=outer_node
                )
            elif curr.type in ("program", "statement_block", "class_body"):
                self._walk(curr, code_bytes, file_path, symbols, parent_class)

    def _handle_function(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        parent_class: str | None,
        outer_node: Node | None = None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        kind = "method" if parent_class else "function"

        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = code_bytes[node.start_byte : sig_end].decode("utf-8", errors="replace").strip()
        signature = re.sub(r"\s+", " ", raw_sig)
        doc_target = outer_node if outer_node is not None else node
        docstring = self._extract_preceding_comments(doc_target, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind=kind,
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=parent_class,
            )
        )

    def _handle_class(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        outer_node: Node | None = None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = code_bytes[node.start_byte : sig_end].decode("utf-8", errors="replace").strip()
        signature = re.sub(r"\s+", " ", raw_sig)
        doc_target = outer_node if outer_node is not None else node
        docstring = self._extract_preceding_comments(doc_target, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind="class",
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=None,
            )
        )

        if body_node:
            for child in body_node.children:
                if child.type == "method_definition":
                    m_name_node = child.child_by_field_name("name")
                    if m_name_node:
                        m_name = self._text(m_name_node, code_bytes)
                        m_body = child.child_by_field_name("body")
                        m_end = m_body.start_byte if m_body else child.end_byte
                        m_sig = (
                            code_bytes[child.start_byte : m_end]
                            .decode("utf-8", errors="replace")
                            .strip()
                        )
                        symbols.append(
                            Symbol(
                                name=m_name,
                                kind="method",
                                language=self.language_name,
                                file_path=file_path,
                                line_start=child.start_point.row + 1,
                                line_end=child.end_point.row + 1,
                                signature=re.sub(r"\s+", " ", m_sig),
                                docstring=self._extract_preceding_comments(child, code_bytes),
                                parent_symbol=name,
                            )
                        )

    def _handle_interface_or_type(
        self,
        node: Node,
        code_bytes: bytes,
        file_path: str,
        symbols: list[Symbol],
        kind: str,
        outer_node: Node | None = None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node, code_bytes)
        body_node = node.child_by_field_name("body")
        sig_end = body_node.start_byte if body_node else node.end_byte
        raw_sig = code_bytes[node.start_byte : sig_end].decode("utf-8", errors="replace").strip()
        signature = re.sub(r"\s+", " ", raw_sig)
        doc_target = outer_node if outer_node is not None else node
        docstring = self._extract_preceding_comments(doc_target, code_bytes)

        symbols.append(
            Symbol(
                name=name,
                kind=kind,
                language=self.language_name,
                file_path=file_path,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=signature,
                docstring=docstring,
                parent_symbol=None,
            )
        )


# =========================================================================
# Main CodeParser Coordinator
# =========================================================================


class CodeParser:
    """Coordinates Tree-Sitter parsing across programming languages with cached parsers."""

    EXTENSION_MAP: ClassVar[dict[str, str]] = {
        ".rs": "rust",
        ".py": "python",
        ".go": "go",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "javascript",
    }

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self._extractors: dict[str, BaseExtractor] = {
            "python": PythonExtractor(),
            "rust": RustExtractor(),
            "go": GoExtractor(),
            "typescript": TypeScriptExtractor("typescript"),
            "tsx": TypeScriptExtractor("tsx"),
            "javascript": TypeScriptExtractor("javascript"),
        }

    def _get_parser(self, lang_key: str) -> Parser | None:
        """Lazily initialize and cache Tree-Sitter parser for the target language."""
        if lang_key in self._parsers:
            return self._parsers[lang_key]

        try:
            if lang_key == "python":
                lang = Language(tspython.language())
            elif lang_key == "rust":
                lang = Language(tsrust.language())
            elif lang_key == "go":
                lang = Language(tsgo.language())
            elif lang_key == "typescript":
                lang = Language(tstypescript.language_typescript())
            elif lang_key == "tsx":
                lang = Language(tstypescript.language_tsx())
            elif lang_key == "javascript":
                lang = Language(tsjavascript.language())
            else:
                return None

            parser = Parser(lang)
            self._parsers[lang_key] = parser
            return parser
        except Exception as e:
            logger.warning("Failed to initialize tree-sitter parser for '%s': %s", lang_key, e)
            return None

    def detect_language(self, path: Path | str) -> str | None:
        """Infer target language from file extension."""
        suffix = Path(path).suffix.lower()
        return self.EXTENSION_MAP.get(suffix)

    def parse_source(
        self,
        code: str | bytes,
        language: str,
        rel_path: str = "",
    ) -> list[Symbol]:
        """Parse source code in memory and extract symbols."""
        parser = self._get_parser(language)
        extractor = self._extractors.get(language)
        if parser is None or extractor is None:
            return []

        code_bytes = code.encode("utf-8") if isinstance(code, str) else code

        try:
            tree = parser.parse(code_bytes)
            if tree is None or tree.root_node is None:
                return []
            return extractor.extract(tree.root_node, code_bytes, rel_path)
        except Exception as e:
            logger.debug("Tree-sitter parse error in %s: %s", rel_path, e)
            return []

    def parse_file(self, file_path: Path | str, rel_path: str = "") -> list[Symbol]:
        """Read file from disk and extract symbols."""
        path = Path(file_path)
        if not path.is_file():
            return []

        lang = self.detect_language(path)
        if not lang:
            return []

        try:
            code_bytes = path.read_bytes()
            effective_rel = rel_path or str(path)
            return self.parse_source(code_bytes, lang, effective_rel)
        except Exception as e:
            logger.warning("Failed to read file %s: %s", path, e)
            return []
