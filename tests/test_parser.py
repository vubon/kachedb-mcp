"""Unit tests for Tree-Sitter AST parser and symbol extraction engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kachedb_mcp.indexer.parser import (
    CodeParser,
    Symbol,
    SymbolRegistry,
)


@pytest.fixture
def parser() -> CodeParser:
    return CodeParser()


# =========================================================================
# Python Extraction Tests
# =========================================================================


def test_python_extraction(parser: CodeParser) -> None:
    code = '''
"""Module docstring."""

class CacheManager:
    """Manages LRU and S3-FIFO cache entries."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def evict(self) -> bool:
        """Evicts oldest entry."""
        return True

async def fetch_remote(url: str) -> str:
    """Fetch remote URL asynchronously."""
    return "ok"

@property
def simple_prop():
    return 42
'''
    symbols = parser.parse_source(code, "python", "src/cache.py")
    assert len(symbols) >= 4

    names = {s.name: s for s in symbols}
    assert "CacheManager" in names
    assert names["CacheManager"].kind == "class"
    assert names["CacheManager"].docstring == "Manages LRU and S3-FIFO cache entries."

    assert "__init__" in names
    assert names["__init__"].kind == "method"
    assert names["__init__"].parent_symbol == "CacheManager"

    assert "evict" in names
    assert names["evict"].kind == "method"
    assert names["evict"].parent_symbol == "CacheManager"
    assert names["evict"].docstring == "Evicts oldest entry."

    assert "fetch_remote" in names
    assert names["fetch_remote"].kind == "function"
    assert names["fetch_remote"].docstring == "Fetch remote URL asynchronously."


# =========================================================================
# Rust Extraction Tests
# =========================================================================


def test_rust_extraction(parser: CodeParser) -> None:
    code = """
/// Core configuration struct for SwissTable
pub struct Config {
    pub port: u16,
    pub max_memory: usize,
}

pub enum CachePolicy {
    Lru,
    S3Fifo,
}

pub trait Storage {
    fn get(&self, key: &str) -> Option<Vec<u8>>;
}

impl Storage for Config {
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        None
    }
}

/// Initializes global memory pool
pub fn init_pool(size: usize) -> Result<(), ()> {
    Ok(())
}
"""
    symbols = parser.parse_source(code, "rust", "src/storage.rs")
    names = {s.name: s for s in symbols}

    assert "Config" in names
    assert names["Config"].kind == "struct"
    assert names["Config"].docstring == "Core configuration struct for SwissTable"

    assert "CachePolicy" in names
    assert names["CachePolicy"].kind == "enum"

    assert "Storage" in names
    assert names["Storage"].kind == "trait"

    assert "init_pool" in names
    assert names["init_pool"].kind == "function"
    assert names["init_pool"].docstring == "Initializes global memory pool"

    assert "get" in names
    assert names["get"].kind == "method"


# =========================================================================
# Go Extraction Tests
# =========================================================================


def test_go_extraction(parser: CodeParser) -> None:
    code = """
package main

// Server handles TCP traffic
type Server struct {
    port int
}

// Router provides request routing
type Router interface {
    Route(path string) error
}

// Start begins listening on TCP port
func (s *Server) Start() error {
    return nil
}

func NewServer(port int) *Server {
    return &Server{port: port}
}
"""
    symbols = parser.parse_source(code, "go", "server.go")
    names = {s.name: s for s in symbols}

    assert "Server" in names
    assert names["Server"].kind == "struct"
    assert names["Server"].docstring == "Server handles TCP traffic"

    assert "Router" in names
    assert names["Router"].kind == "interface"
    assert names["Router"].docstring == "Router provides request routing"

    assert "Start" in names
    assert names["Start"].kind == "method"
    assert names["Start"].parent_symbol == "Server"
    assert names["Start"].docstring == "Start begins listening on TCP port"

    assert "NewServer" in names
    assert names["NewServer"].kind == "function"


# =========================================================================
# TypeScript / JavaScript Extraction Tests
# =========================================================================


def test_typescript_extraction(parser: CodeParser) -> None:
    code = """
/** User session data */
export interface UserSession {
    id: string;
    token: string;
}

export type TokenHandler = (token: string) => Promise<boolean>;

export class SessionService {
    constructor(private secret: string) {}

    /** Verify user session validity */
    async verify(token: string): Promise<boolean> {
        return true;
    }
}

export function createService(secret: string): SessionService {
    return new SessionService(secret);
}
"""
    symbols = parser.parse_source(code, "typescript", "src/auth.ts")
    names = {s.name: s for s in symbols}

    assert "UserSession" in names
    assert names["UserSession"].kind == "interface"
    assert names["UserSession"].docstring == "User session data"

    assert "TokenHandler" in names
    assert names["TokenHandler"].kind == "type"

    assert "SessionService" in names
    assert names["SessionService"].kind == "class"

    assert "verify" in names
    assert names["verify"].kind == "method"
    assert names["verify"].parent_symbol == "SessionService"
    assert names["verify"].docstring == "Verify user session validity"

    assert "createService" in names
    assert names["createService"].kind == "function"


# =========================================================================
# File & Extension Detection Tests
# =========================================================================


def test_code_parser_file_parsing(parser: CodeParser, tmp_path: Path) -> None:
    test_file = tmp_path / "helper.py"
    test_file.write_text("def helper_fn():\n    return 1\n", encoding="utf-8")

    symbols = parser.parse_file(test_file, "helper.py")
    assert len(symbols) == 1
    assert symbols[0].name == "helper_fn"

    # Non-existent file
    assert parser.parse_file(tmp_path / "non_existent.py") == []

    # Unsupported extension
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("plain text", encoding="utf-8")
    assert parser.parse_file(txt_file) == []


def test_code_parser_syntax_error_resilience(parser: CodeParser) -> None:
    # Tree-sitter is resilient and should not throw on syntax errors
    bad_code = "def incomplete():\n    return @@@\n\ndef valid():\n    pass\n"
    symbols = parser.parse_source(bad_code, "python", "broken.py")
    assert any(s.name == "valid" for s in symbols)


# =========================================================================
# SymbolRegistry & Collision Handling Tests
# =========================================================================


def test_symbol_registry_collision_handling() -> None:
    registry = SymbolRegistry()

    # Generic symbol name "new" appearing in two different files
    sym1 = Symbol(
        name="new",
        kind="function",
        language="rust",
        file_path="crates/kachedb-net/src/connection.rs",
        line_start=42,
        line_end=50,
        signature="pub fn new(stream: TcpStream) -> Self",
        docstring="Creates new client connection handler",
    )
    sym2 = Symbol(
        name="new",
        kind="function",
        language="rust",
        file_path="crates/kachedb-hash/src/table.rs",
        line_start=28,
        line_end=35,
        signature="pub fn new(capacity: usize) -> Self",
        docstring="Initializes empty SwissTable hash arena",
    )
    sym3 = Symbol(
        name="calculate_hash",
        kind="function",
        language="rust",
        file_path="crates/kachedb-hash/src/table.rs",
        line_start=60,
        line_end=65,
        signature="pub fn calculate_hash(key: &[u8]) -> u64",
        docstring="Compute Murmur3 hash",
    )

    registry.add_all([sym1, sym2, sym3])

    # 1. Collision resolution for "new"
    defs = registry.get_definitions("new")
    assert len(defs) == 2
    assert defs[0]["file"] == "crates/kachedb-net/src/connection.rs"
    assert defs[1]["file"] == "crates/kachedb-hash/src/table.rs"

    # 2. SwissTable payload generation
    payload = registry.to_swisstable_payload("My Workspace")
    assert "sym:my-workspace:new" in payload

    parsed_arr = json.loads(payload["sym:my-workspace:new"])
    assert isinstance(parsed_arr, list)
    assert len(parsed_arr) == 2

    # Fully qualified key also present
    fq_key = "sym:my-workspace:crates/kachedb-net/src/connection.rs:new"
    assert fq_key in payload
    fq_parsed = json.loads(payload[fq_key])
    assert len(fq_parsed) == 1
    assert fq_parsed[0]["signature"] == "pub fn new(stream: TcpStream) -> Self"

    # 3. Vector records generation
    vec_records = registry.to_vector_records("My Workspace")
    assert len(vec_records) == 3
    key, text, meta = vec_records[0]
    assert key.startswith("vec:my-workspace:crates/kachedb-net/src/connection.rs:new:42")
    assert "pub fn new" in text
    assert meta["workspace_id"] == "my-workspace"
    assert meta["symbol"] == "new"


def test_python_ast_edge_cases(parser: CodeParser) -> None:
    """Verify decorated classes, decorated methods, and nested classes in Python."""
    code = """
@dataclass
class Account:
    username: str

    @property
    def is_active(self) -> bool:
        \"\"\"Check if account is active.\"\"\"
        return True

    @classmethod
    def create_guest(cls) -> Account:
        return cls(username="guest")

    class Metadata:
        created_at: int
"""
    symbols = parser.parse_source(code, "python", "src/models.py")
    names = {s.name: s for s in symbols}

    assert "Account" in names
    assert names["Account"].kind == "class"

    assert "is_active" in names
    assert names["is_active"].kind == "method"
    assert names["is_active"].parent_symbol == "Account"
    assert names["is_active"].docstring == "Check if account is active."

    assert "create_guest" in names
    assert names["create_guest"].kind == "method"
    assert names["create_guest"].parent_symbol == "Account"

    # Definition dict includes parent
    d = names["is_active"].to_definition_dict()
    assert d.get("parent") == "Account"


def test_rust_ast_edge_cases(parser: CodeParser) -> None:
    """Verify generics with trait bounds, traits with method definitions, and macros."""
    code = """
pub trait Service {
    fn serve(&self, req: &str) -> String;
    fn ping(&self) -> bool {
        true
    }
}

pub struct Container<T: Clone + Send> {
    pub inner: T,
}

impl<T: Clone + Send> Container<T> {
    pub fn get_inner(&self) -> T {
        self.inner.clone()
    }
}

pub fn transform<T: Ord + Clone, R: Default>(items: &[T]) -> R {
    R::default()
}

macro_rules! define_handler {
    ($name:ident) => {
        pub struct $name;
    };
}
"""
    symbols = parser.parse_source(code, "rust", "src/service.rs")
    names = {s.name: s for s in symbols}

    assert "Service" in names
    assert names["Service"].kind == "trait"

    assert "serve" in names
    assert names["serve"].kind == "method"

    assert "ping" in names
    assert names["ping"].kind == "method"

    assert "Container" in names
    assert names["Container"].kind == "struct"

    assert "get_inner" in names
    assert names["get_inner"].kind == "method"

    assert "transform" in names
    assert names["transform"].kind == "function"
    assert "Ord + Clone" in names["transform"].signature

    assert "define_handler" in names
    assert names["define_handler"].kind == "macro"


def test_go_ast_edge_cases(parser: CodeParser) -> None:
    """Verify value vs pointer receivers, and type aliases in Go."""
    code = """
package client

type ClientConfig struct {
    TimeoutMs int
}

type ID uint64

func (c ClientConfig) Timeout() int {
    return c.TimeoutMs
}

func (c *ClientConfig) Reset() {
    c.TimeoutMs = 0
}
"""
    symbols = parser.parse_source(code, "go", "client.go")
    names = {s.name: s for s in symbols}

    assert "ClientConfig" in names
    assert names["ClientConfig"].kind == "struct"

    assert "ID" in names
    assert names["ID"].kind == "type"

    assert "Timeout" in names
    assert names["Timeout"].kind == "method"
    assert names["Timeout"].parent_symbol == "ClientConfig"

    assert "Reset" in names
    assert names["Reset"].kind == "method"
    assert names["Reset"].parent_symbol == "ClientConfig"


def test_tsx_and_javascript_extraction(parser: CodeParser) -> None:
    """Verify TSX React components and plain JavaScript parsing."""
    tsx_code = """
import React from 'react';

export interface ButtonProps {
    label: string;
    onClick: () => void;
}

export function ActionButton(props: ButtonProps): JSX.Element {
    return <button onClick={props.onClick}>{props.label}</button>;
}
"""
    tsx_syms = parser.parse_source(tsx_code, "tsx", "src/Button.tsx")
    tsx_names = {s.name: s for s in tsx_syms}
    assert "ButtonProps" in tsx_names
    assert "ActionButton" in tsx_names

    js_code = """
function calculateTax(amount, rate) {
    return amount * rate;
}

class InvoiceManager {
    generate(orderId) {
        return { orderId, status: "pending" };
    }
}
"""
    js_syms = parser.parse_source(js_code, "javascript", "src/tax.js")
    js_names = {s.name: s for s in js_syms}
    assert "calculateTax" in js_names
    assert "InvoiceManager" in js_names
    assert "generate" in js_names


def test_code_parser_corner_cases(parser: CodeParser, tmp_path: Path) -> None:
    """Verify parser caching, unknown languages, and file read error handling."""
    # 1. Cached parser retrieval
    p1 = parser._get_parser("python")
    p2 = parser._get_parser("python")
    assert p1 is p2

    # 2. Unknown language
    assert parser._get_parser("cobol") is None
    assert parser.parse_source("some code", "cobol") == []

    # 3. Detect language
    assert parser.detect_language("app.tsx") == "tsx"
    assert parser.detect_language("script.js") == "javascript"
    assert parser.detect_language("page.jsx") == "javascript"
    assert parser.detect_language("unknown.xyz") is None

    # 4. File read failure
    test_file = tmp_path / "protected.py"
    test_file.write_text("def test(): pass\n", encoding="utf-8")
    with patch.object(Path, "read_bytes", side_effect=OSError("Access denied")):
        assert parser.parse_file(test_file) == []


def test_symbol_registry_extended() -> None:
    """Verify SymbolRegistry symbol list and exact definition lookup."""
    reg = SymbolRegistry()
    sym = Symbol(
        name="Server",
        kind="class",
        language="python",
        file_path="src/server.py",
        line_start=1,
        line_end=10,
        signature="class Server:",
        parent_symbol=None,
    )
    reg.add(sym)

    assert len(reg.symbols) == 1
    assert reg.symbols[0].name == "Server"

    # Exact definition lookup
    exact = reg.get_exact_definition("src/server.py", "Server")
    assert exact is not None
    assert exact["signature"] == "class Server:"

    assert reg.get_exact_definition("src/server.py", "Missing") is None
    assert reg.get_definitions("UnknownSymbol") == []
