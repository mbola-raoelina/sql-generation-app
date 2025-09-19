#!/usr/bin/env python3
"""
Ingest an Oracle-like schema.sql (CREATE TABLE + COMMENTS + CONSTRAINTS + INDEXES)
into a ChromaDB vector store with chunked long comments, ready for LangChain RAG.

- Parses tables, columns (with types), table/column comments, PK/FK constraints, and indexes.
- Creates "documents" at 3 levels: table summary, column-level docs, and chunked long table comments.
- Saves a JSONL export of all documents for inspection.
- Builds (or updates) a Chroma collection with rich metadata for precise retrieval.

Author: ChatGPT (modified to be idempotent + metadata-safe)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional imports - ensure installed in your venv
try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception as e:
    print("Missing chromadb (install via pip). Exception:", e)
    raise

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    # embeddings are executed by chromadb embedding function; SentenceTransformer is still useful if needed
    pass

# ------------------------------
# Utilities
# ------------------------------

def unescape_sql_single_quoted(s: str) -> str:
    """Convert SQL single-quoted string with doubled quotes to Python string."""
    return s.replace("''", "'")


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 800, overlap: int = 80) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = split_sentences(text)
    chunks = []
    cur = ""

    def push(cur_text: str):
        if cur_text:
            chunks.append(cur_text)

    for sent in sentences:
        if len(sent) > max_chars:
            # hard-cut long sentence
            for i in range(0, len(sent), max_chars):
                piece = sent[i:i + max_chars]
                if cur:
                    push(cur)
                    cur = ""
                push(piece)
            continue

        if len(cur) + 1 + len(sent) <= max_chars:
            cur = (cur + " " + sent).strip() if cur else sent
        else:
            push(cur)
            cur = sent

    push(cur)

    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        with_overlap = []
        for i, c in enumerate(chunks):
            if i == 0:
                with_overlap.append(c)
            else:
                prev_tail = chunks[i-1][-overlap:]
                with_overlap.append((prev_tail + " " + c).strip())
        chunks = with_overlap

    return chunks


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class TableInfo:
    name: str
    columns: List[Tuple[str, str]]
    primary_key: List[str]
    foreign_keys: List[Tuple[str, str, List[str], str, List[str]]]
    indexes: List[str]
    table_comment: str
    column_comments: Dict[str, str]


@dataclass
class Doc:
    id: str
    text: str
    metadata: Dict


# ------------------------------
# Parsing regexes
# ------------------------------

CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?P<name>[A-Z0-9_]+)\s*\((?P<body>.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

COLUMN_DEF_RE = re.compile(
    r"^\s*(?P<col>[A-Z0-9_]+)\s+(?P<type>[A-Z0-9_(),\s]+?)\s*(?:,|$)",
    re.IGNORECASE | re.MULTILINE,
)

CONSTRAINT_PK_RE = re.compile(
    r"CONSTRAINT\s+[A-Z0-9_]+\s+PRIMARY\s+KEY\s*\((?P<cols>[^)]+)\)",
    re.IGNORECASE,
)

CONSTRAINT_FK_RE = re.compile(
    r"CONSTRAINT\s+(?P<cname>[A-Z0-9_]+)\s+FOREIGN\s+KEY\s*\((?P<lcols>[^)]+)\)\s+REFERENCES\s+(?P<rtab>[A-Z0-9_]+)\s*\((?P<rcols>[^)]+)\)",
    re.IGNORECASE,
)

COMMENT_TABLE_RE = re.compile(
    r"COMMENT\s+ON\s+TABLE\s+(?P<tab>[A-Z0-9_]+)\s+IS\s+'(?P<txt>.*?)'\s*;",
    re.IGNORECASE | re.DOTALL,
)

COMMENT_COLUMN_RE = re.compile(
    r"COMMENT\s+ON\s+COLUMN\s+(?P<tab>[A-Z0-9_]+)\.(?P<col>[A-Z0-9_]+)\s+IS\s+'(?P<txt>.*?)'\s*;",
    re.IGNORECASE | re.DOTALL,
)

CREATE_INDEX_RE = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?P<idx>[A-Z0-9_]+)\s+ON\s+(?P<tab>[A-Z0-9_]+)\s*\((?P<cols>[^)]+)\)\s*;",
    re.IGNORECASE,
)


# ------------------------------
# Parsing logic
# ------------------------------

def parse_schema(sql: str) -> Dict[str, TableInfo]:
    table_comments: Dict[str, str] = {}
    for m in COMMENT_TABLE_RE.finditer(sql):
        tab = m.group("tab").upper()
        txt = unescape_sql_single_quoted(m.group("txt"))
        table_comments[tab] = txt.strip()

    column_comments: Dict[Tuple[str, str], str] = {}
    for m in COMMENT_COLUMN_RE.finditer(sql):
        tab = m.group("tab").upper()
        col = m.group("col").upper()
        txt = unescape_sql_single_quoted(m.group("txt"))
        column_comments[(tab, col)] = txt.strip()

    table_indexes: Dict[str, List[str]] = {}
    for m in CREATE_INDEX_RE.finditer(sql):
        tab = m.group("tab").upper()
        idx = m.group("idx").upper()
        table_indexes.setdefault(tab, []).append(idx)

    tables: Dict[str, TableInfo] = {}

    for m in CREATE_TABLE_RE.finditer(sql):
        name = m.group("name").upper()
        body = m.group("body")

        # Collect column lines until first CONSTRAINT line
        col_section = []
        for line in body.splitlines():
            if re.search(r"\bCONSTRAINT\b", line, re.IGNORECASE):
                break
            col_section.append(line)
        col_section_text = "\n".join(col_section)

        columns = []
        for cm in COLUMN_DEF_RE.finditer(col_section_text):
            col = cm.group("col").upper()
            typ = normalize_whitespace(cm.group("type").upper())
            columns.append((col, typ))

        pk_cols = []
        for pk in CONSTRAINT_PK_RE.finditer(body):
            cols = [c.strip().upper() for c in pk.group("cols").split(",")]
            pk_cols.extend(cols)

        fks: List[Tuple[str, str, List[str], str, List[str]]] = []
        for fk in CONSTRAINT_FK_RE.finditer(body):
            cname = fk.group("cname").upper()
            lcols = [c.strip().upper() for c in fk.group("lcols").split(",")]
            rtab = fk.group("rtab").upper()
            rcols = [c.strip().upper() for c in fk.group("rcols").split(",")]
            fks.append((cname, name, lcols, rtab, rcols))

        tinfo = TableInfo(
            name=name,
            columns=columns,
            primary_key=pk_cols,
            foreign_keys=fks,
            indexes=table_indexes.get(name, []),
            table_comment=table_comments.get(name, ""),
            column_comments={col: column_comments.get((name, col), "") for col, _ in columns},
        )
        tables[name] = tinfo

    return tables


# ------------------------------
# Document building
# ------------------------------

def table_summary_text(t: TableInfo, max_cols: int = 30) -> str:
    col_list = ", ".join([c for c, _ in t.columns[:max_cols]])
    first_sentence = ""
    if t.table_comment:
        sents = split_sentences(t.table_comment)
        first_sentence = sents[0] if sents else t.table_comment[:240]

    parts = [
        f"TABLE: {t.name}",
        f"PRIMARY KEY: {', '.join(t.primary_key) if t.primary_key else '(none)'}",
        f"FOREIGN KEYS: " + (
            "; ".join([f"{cn}:{'/'.join(l)} -> {rt}({'/'.join(rc)})" for cn, _, l, rt, rc in t.foreign_keys]) if t.foreign_keys else "(none)"
        ),
        f"INDEXES: {', '.join(t.indexes) if t.indexes else '(none)'}",
        f"COLUMNS: {col_list}" if col_list else "COLUMNS: (none)",
    ]
    if first_sentence:
        parts.append(f"SUMMARY: {first_sentence}")
    return "\n".join(parts).strip()


def build_documents(tables: Dict[str, TableInfo], comment_chunk_chars: int) -> List[Doc]:
    docs: List[Doc] = []

    for t in tables.values():
        # Table summary doc
        tbl_text = table_summary_text(t)
        docs.append(Doc(
            id=f"table::{t.name}",
            text=tbl_text,
            metadata={
                "doc_type": "table",
                "table": t.name,
                "primary_key": t.primary_key,
                "indexes": t.indexes,
            }
        ))

        # Column-level docs
        for col, typ in t.columns:
            col_desc = t.column_comments.get(col, "")
            col_text = "\n".join([
                f"TABLE: {t.name}",
                f"COLUMN: {col}",
                f"TYPE: {typ}",
                f"DESCRIPTION: {col_desc or '(none)'}",
                f"TABLE_SUMMARY: {split_sentences(t.table_comment)[0] if t.table_comment else ''}",
                f"PRIMARY_KEY: {'YES' if col in t.primary_key else 'NO'}",
            ]).strip()

            docs.append(Doc(
                id=f"column::{t.name}.{col}",
                text=col_text,
                metadata={
                    "doc_type": "column",
                    "table": t.name,
                    "column": col,
                    "type": typ,
                    "is_pk": col in t.primary_key,
                }
            ))

        # Long table comments chunked
        if t.table_comment:
            chunks = chunk_text(t.table_comment, max_chars=comment_chunk_chars, overlap=80)
            for i, ch in enumerate(chunks):
                docs.append(Doc(
                    id=f"tcomment::{t.name}::{i+1}",
                    text=f"TABLE: {t.name}\nCOMMENT_CHUNK: {i+1}\n{ch}",
                    metadata={
                        "doc_type": "table_comment",
                        "table": t.name,
                        "chunk_index": i + 1,
                        "total_chunks": len(chunks),
                    }
                ))

    return docs


# ------------------------------
# Embedding function wrapper (Chroma) 
# ------------------------------

def get_embedding_function(model_name: str):
    """
    Return a chromadb embedding function wrapper that uses SentenceTransformers.
    model_name should be a sentence-transformers model id like 'sentence-transformers/all-MiniLM-L6-v2'
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


# ------------------------------
# Metadata sanitization & idempotent ingestion
# ------------------------------

def _sanitize_value(v):
    if v is None:
        return None
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        simple = all(isinstance(x, (str, int, float, bool, type(None))) for x in v)
        if simple:
            return ",".join("" if x is None else str(x) for x in v)
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def sanitize_metadata(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        key = str(k)
        out[key] = _sanitize_value(v)
    return out


def ingest_to_chroma(
    docs: List[Doc],
    persist_dir: str,
    collection_name: str,
    model_name: str,
    force_rebuild: bool = False,
    batch_size: int = 512,
    show_progress: bool = True
) -> dict:
    """
    Idempotent ingestion into Chroma:
     - If force_rebuild True: delete persist_dir before creating client (fresh store).
     - Only upserts documents whose ids are NOT already present (avoid re-embedding existing docs).
     - Sanitizes metadata values so they are primitives supported by Chroma.
     - Retries on errors with smaller granularity to surface problematic documents.
    Returns a summary dict with counts and any failures.
    """
    p = Path(persist_dir)
    if force_rebuild and p.exists():
        if show_progress:
            print(f"[ingest] force_rebuild True -> removing persist_dir: {persist_dir}")
        shutil.rmtree(p)

    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=get_embedding_function(model_name),
        )
    except Exception:
        # fallback: try get_collection then create if missing
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=get_embedding_function(model_name),
            )

    existing_ids = set()
    try:
        info = collection.get(include=["ids"])
        existing_ids = set(info.get("ids", []) or [])
        if show_progress:
            print(f"[ingest] Found {len(existing_ids)} existing ids in collection '{collection_name}'.")
    except Exception as e:
        if show_progress:
            print(f"[ingest] Warning: unable to list existing ids: {e}. Will attempt full upsert.")
        existing_ids = set()

    to_embed = [d for d in docs if d.id not in existing_ids]
    to_skip = [d for d in docs if d.id in existing_ids]

    if show_progress:
        print(f"[ingest] {len(to_embed)} new docs will be embedded/upserted; {len(to_skip)} docs skipped (already present).")

    failures = []
    total_new = len(to_embed)

    for i in range(0, total_new, batch_size):
        batch = to_embed[i:i + batch_size]
        ids = [d.id for d in batch]
        documents = [d.text for d in batch]
        metadatas = [sanitize_metadata(d.metadata) for d in batch]

        try:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            if show_progress:
                print(f"[ingest] Upserted {min(i + batch_size, total_new)}/{total_new} new docs")
        except Exception as e:
            if show_progress:
                print(f"[ingest] upsert batch failed at {i}/{total_new}: {e}. Retrying one-by-one...")
            for d in batch:
                try:
                    collection.upsert(ids=[d.id], documents=[d.text], metadatas=[sanitize_metadata(d.metadata)])
                except Exception as e2:
                    failures.append({"id": d.id, "error": str(e2)})
                    if show_progress:
                        print(f"[ingest] FAILED upsert doc id={d.id} : {e2}")

    try:
        approx_count = collection.count()
    except Exception:
        approx_count = None

    summary = {
        "new_indexed": total_new - len(failures),
        "skipped_existing": len(to_skip),
        "failures": failures,
        "approx_collection_count": approx_count,
        "persist_dir": str(p.resolve()),
        "collection": collection_name,
    }

    if show_progress:
        print(f"[ingest] Done. Indexed {summary['new_indexed']} new docs, skipped {summary['skipped_existing']}.")
        if failures:
            print(f"[ingest] {len(failures)} failures during ingestion. See 'failures' in returned summary.")

    return summary


# ------------------------------
# CLI / Main
# ------------------------------

def resolve_default_paths(schema_arg: Optional[str], persist_arg: Optional[str]):
    # default to script directory if relative/None
    base = Path(__file__).resolve().parent
    schema_path = Path(schema_arg) if schema_arg else base / "schema.sql"
    if not schema_path.is_absolute():
        schema_path = (Path.cwd() / schema_path).resolve()
    persist_dir = Path(persist_arg) if persist_arg else base / "chroma_db"
    if not persist_dir.is_absolute():
        persist_dir = (Path.cwd() / persist_dir).resolve()
    return str(schema_path), str(persist_dir)


def main():
    parser = argparse.ArgumentParser(description="Ingest schema.sql into Chroma with chunked comments.")
    parser.add_argument("--schema", type=str, required=False, help="Path to schema.sql (default: ./schema.sql)")
    parser.add_argument("--persist_dir", type=str, default=None, help="ChromaDB persist directory (default: ./chroma_db)")
    parser.add_argument("--collection", type=str, default="schema_docs", help="Chroma collection name")
    parser.add_argument("--embed_model", type=str, default="thenlper/gte-base", help="SentenceTransformers model")
    parser.add_argument("--comment_chunk", type=int, default=800, help="Max chars per comment chunk")
    parser.add_argument("--export_jsonl", type=str, default="./schema_docs.jsonl", help="Where to save a JSONL dump of docs")
    parser.add_argument("--force_rebuild", action="store_true", help="Delete persist_dir and re-ingest from scratch")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for upsert")
    args = parser.parse_args()

    schema_path_str, persist_dir_str = resolve_default_paths(args.schema, args.persist_dir)
    schema_path = Path(schema_path_str)
    persist_dir = str(persist_dir_str)

    if not schema_path.exists():
        print(f"ERROR: file not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    print("Reading schema file:", schema_path)
    sql = schema_path.read_text(encoding="utf-8", errors="ignore")

    print("Parsing schema...")
    tables = parse_schema(sql)
    print(f"Parsed {len(tables)} tables")

    print("Building documents...")
    docs = build_documents(tables, comment_chunk_chars=args.comment_chunk)
    print(f"Built {len(docs)} documents (table summaries, columns, comment chunks)")

    # Export JSONL for inspection
    export_path = Path(args.export_jsonl)
    with export_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "text": d.text, "metadata": d.metadata}, ensure_ascii=False) + "\n")
    print(f"Exported docs to {export_path}")

    print("Ingesting to Chroma...")
    summary = ingest_to_chroma(
        docs,
        persist_dir=persist_dir,
        collection_name=args.collection,
        model_name=args.embed_model,
        force_rebuild=args.force_rebuild,
        batch_size=args.batch_size,
        show_progress=True
    )

    print("Ingestion summary:", json.dumps(summary, indent=2, ensure_ascii=False))
    print("Done.")


if __name__ == "__main__":
    main()
