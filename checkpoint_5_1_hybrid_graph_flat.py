r"""Capstone Checkpoint 5.1 — Agentic RAG over the 3.1 HYBRID retriever, as a LangGraph.

Puts the pieces together, without dropping anything that worked:

  - RETRIEVAL: the Checkpoint 3.1 HybridEvaluationRetriever (BM25 + Chroma vector,
    normalized-and-blended) is the engine. It is exposed to the agent as the
    `semantic_search` tool, so every "search the corpus" step goes through the same
    hybrid search that scored well in 3.1.
  - AGENT: a LangGraph state graph runs the ReAct loop
        (entry) -> decide -> [act -> decide]* -> answer -> END
    `decide` calls choose_tool() to pick the next action; `act` runs it and appends
    the result; the conditional edge loops until the agent chooses `answer`.
  - TOOLS: semantic_search (hybrid, INTERNAL) + web_paper_search / calculator / clarify
    (EXTERNAL), plus the terminal `answer`.
  - EVALUATION: for each task we run the FIXED single-pass hybrid pipeline (the 3.1
    baseline) AND the agent, print both, the step count, and whether an external tool
    fired — the before/after evidence Step 4 needs.

  *** MODIFIED (recursive): the corpus is read RECURSIVELY — the directory in PDF_DIR
      *and all its subfolders* are scanned. Chunk/source ids use each file's path
      RELATIVE to PDF_DIR, so two files with the same name in different subfolders
      never collide.

  *** MODIFIED (Option 2 — Abnox web): the indexer also reads TEXT files (.txt/.md).

  *** ADDED (ported from 5.7) — CSV tables, drawings and four document-level tools:
      * The indexer now also reads CSV tables (.csv: one overview chunk + one chunk per
        row; tables bigger than VECTOR_MAX_TABLE_ROWS keep their rows BM25-only) and
        drawings (.png/.jpg inside a 'drawings' subfolder, described once by the vision
        model so they can be found by text). Every chunk carries a `kind`
        ("text" | "table" | "image").
      * drawing_search(query): hybrid search restricted to drawings.
      * list_documents(keyword, manufacturer, doc_type): catalog of the collection, one
        line per file (type, manufacturer, title, codes found). Built locally, no API.
      * read_document(doc, part, query): read ONE whole file in order, in parts, or only
        its passages most relevant to `query`.
      * table_lookup(code, columns, table): EXACT search of article / model codes in the
        CSV tables (also the big ones whose rows are BM25-only).
      Same Chroma store and manifest as before: files already indexed are NOT re-read;
      only the new CSVs / drawings are processed on the next run. ***

Setup: pip install langchain-openai langchain-core langchain-chroma langchain-text-splitters
       pypdf rank-bm25 python-dotenv requests langgraph beautifulsoup4 pillow
       OPENROUTER_API_KEY in a .env next to this script.
"""

# # Checkpoint 5.1 — hybrid retrieval + agentic graph + external tools + document tools

from __future__ import annotations

import ast
import base64
import csv
import difflib
import io
import json
import math
import operator as op
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END   # END from langgraph, not tkinter

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
EMBED_MODEL = "text-embedding-3-small"   # kept as in your working 3.1; "openai/..." also works
TEMPERATURE = 0.2
MAX_STEPS = 5                            # safety cap on agent loops
INTERACTIVE = True                      # True => clarify() prompts you at the console

# --- hybrid retriever config (from 3.1) ---
TOP_K = 10
CANDIDATE_POOL = 20
VECTOR_WEIGHT = 0.4
# Own store for the RECURSIVE index, so it never mixes IDs with a "flat" run's DB.
CHROMA_DIR = os.path.abspath("./chroma_db_eval_recursive")
MANIFEST_CACHE_FILE = "./pdf_manifest_eval_recursive_cache.pkl"
PDF_DIR = r"./data"

# Read files from PDF_DIR and every subfolder underneath it. Set False for top-level only.
RECURSIVE = True

# File types the indexer will read. PDFs go through pypdf; .txt/.md are read as plain text.
PDF_EXTS = (".pdf",)
TEXT_EXTS = (".txt", ".md")
# --- ADDED (from 5.7): CSV tables + drawings ---
CSV_EXTS = (".csv",)
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
IMAGE_SUBDIRS = ("drawings",)         # images are indexed only inside these subfolders
CSV_ROWS_PER_CHUNK = 1
VECTOR_MAX_TABLE_ROWS = 2000          # bigger CSV -> rows BM25-only (saves embeddings)
DESCRIBE_IMAGES = True                # vision model describes each drawing ONCE
VISION_MODEL = LLM_MODEL
VISION_WORKERS = 4
IMAGE_MAX_SIDE = 1600

# --- ADDED (from 5.7): document-level tools ---
LIST_DOCS_MAX = 25            # catalog lines returned per list_documents call
READ_DOC_MAX_CHARS = 8000     # characters returned per read_document part
READ_TABLE_MAX_ROWS = 60      # rows shown when read_document opens a CSV table
TABLE_LOOKUP_MAX_ROWS = 15    # rows returned per table_lookup call
MANUFACTURERS = (             # display name, keys searched in file path / first page
    ("ABNOX", ("abnox",)),
    ("Schütze", ("schutze", "alfred schutze")),
    ("Soma", ("soma",)),
    ("Lubtec", ("lubtec",)),
    ("Walther Systemtechnik", ("walther",)),
)

LOG_PATH = Path.cwd() / "checkpoint_5_1_hybrid_graph.log"
SCENARIO = ("data folder with PDFs + crawled Abnox web pages + CSV tables + drawings, "
            "hybrid retrieval, agentic graph, external + document tools")


if sys.version_info >= (3, 13):
    print(
        f"[warning] Running on Python {sys.version_info.major}.{sys.version_info.minor}. "
        "This course targets Python 3.11/3.12. If you hit a pathlib/chromadb crash, "
        "run: pip install -U chromadb langchain-chroma  (or use a 3.12 venv).\n"
    )

# --- system prompts ---
TOOL_SYSTEM = (
    "You are a research-paper navigator agent over a HYBRID (BM25 + vector) search index "
    "of the user's collection (PDFs, web pages, CSV tables and technical drawings). "
    "Each step you call exactly ONE tool:\n"
    "- semantic_search(query): hybrid search over the LOCAL index. Results tagged "
    "[TABLE ROW] are rows of a spreadsheet (Column: value | ...); results tagged [DRAWING] "
    "are exploded views / technical drawings.\n"
    "- drawing_search(query): hybrid search over the DRAWINGS only (exploded views, parts "
    "lists). Use it when the question is about a specific model, its parts, spare parts, "
    "assembly or disassembly.\n"
    "- list_documents(keyword, manufacturer, doc_type): the CATALOG of the collection, one "
    "line per file (manuals, datasheets, CSV tables, drawings, catalogues) with file name, "
    "type, manufacturer, title and the model/article codes found in it. All arguments are "
    "optional: manufacturer (e.g. 'abnox', 'schuetze'), doc_type ('manual', 'datasheet', "
    "'table', 'drawing', 'catalogue', 'spare_parts'), keyword (a topic or a code; ranks "
    "the files). Use it for questions about the collection as a whole ('which manuals / "
    "datasheets do we have for ...') or to find the exact file name before read_document.\n"
    "- read_document(doc, part, query): read ONE whole file in order. doc = file name as "
    "shown in results or in the catalog (partial names work). Long files come in parts: "
    "call again with part=2, 3...; or give query to get only the passages of that file "
    "most relevant to it. Use it when you know which datasheet / manual / table matters "
    "and need its full content instead of scattered chunks.\n"
    "- table_lookup(code, columns, table): EXACT lookup of article / model / part codes in "
    "the CSV tables (stock, prices, parts lists). code may hold several codes separated by "
    "commas; columns (optional) limits the columns shown, e.g. 'stock, price'; table "
    "(optional) restricts the search to one table file.\n"
    "- web_paper_search(query): search EXTERNAL papers (Semantic Scholar) NOT in the local "
    "collection. Use only when the local index cannot answer.\n"
    "- calculator(expression): exact arithmetic, e.g. '128 / 3'.\n"
    "- clarify(question): ask the user when the request is ambiguous.\n"
    "- answer(text): give the FINAL grounded answer (in args.text) and stop.\n\n"
    "Policy: when the question contains a specific article/model code and asks for data "
    "kept in tables (stock, price, description, parts), call table_lookup FIRST. For "
    "questions about which documents exist, call list_documents. When one document clearly "
    "holds the answer, read it with read_document. Otherwise search the LOCAL index FIRST; "
    "refine the query or search again if the results are partial; reach for "
    "web_paper_search only when the collection lacks the answer; use calculator for any "
    "arithmetic. Do not repeat a call that already failed. Call answer as soon as the "
    "gathered notes support a grounded reply, quoting document ids. If a [DRAWING] is "
    "relevant, quote its id and use its position numbers when you name parts.\n\n"
    'Respond with ONLY a JSON object: {"tool": "<name>", "args": {...}, "reasoning": "..."}'
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided notes (local "
    "documents, table rows, drawings and any external tool results), quoting document ids "
    "where you can. If the notes do not contain the answer, say so."
)
FIXED_SYSTEM = (
    "You are a technical assistant. You help our customers find the right spare parts. We are approved distributors of Abnox, Soma, Lubtec and Walther Systemtecnik. Answer the question using ONLY the provided documents, and "
    "quote from them where you can. If the documents do not contain the answer, say so."
)
IMAGE_PROMPT = (
    "This is a technical drawing from a spare-parts catalogue (often an exploded view of a "
    "valve, pump or dispensing component). Write a description that will be used to FIND "
    "this drawing with a text search:\n"
    "1. Product: manufacturer, product name and model/type numbers visible anywhere.\n"
    "2. Title block: transcribe drawing number, title, revision, date if visible.\n"
    "3. Parts list: transcribe EVERY position number with its part name / article number "
    "exactly as written (one per line: 'Pos 12 - O-ring 10x2 - Art. 123456').\n"
    "4. One or two sentences on what the drawing shows, in English AND in Spanish.\n"
    "Only write what you can actually read; write 'illegible' instead of guessing."
)


def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Put it in a .env next to this script.")
    return key


def make_llm(temperature: float = TEMPERATURE, model: str = LLM_MODEL) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)       # JSON surrounded by text
        if m:
            return json.loads(m.group(0))
        raise


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ## ADDED (from 5.7) — CSV reading, codes, image loading

def read_csv_table(path: str | Path) -> tuple[list[str], list[list[str]], list[str]]:
    """Excel-exported CSV -> (headers, rows, preamble). Detects encoding, delimiter and
    skips title rows above the real header."""
    raw = Path(path).read_bytes()
    text = ""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:20000]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        delim = ";" if first.count(";") > first.count(",") else ","

    rows = [[" ".join(c.split()) for c in r]
            for r in csv.reader(io.StringIO(text), delimiter=delim)]
    rows = [r for r in rows if any(r)]
    if not rows:
        return [], [], []

    def filled(r: list[str]) -> int:
        return sum(1 for c in r if c)

    max_filled = max(filled(r) for r in rows[:50])
    hdr_i = next((i for i, r in enumerate(rows[:15]) if filled(r) >= max(2, 0.5 * max_filled)), 0)
    preamble = [" ".join(c for c in r if c) for r in rows[:hdr_i]]
    width = max(len(r) for r in rows)
    header_row = rows[hdr_i] + [""] * (width - len(rows[hdr_i]))
    headers, used = [], set()
    for j, h in enumerate(header_row):
        h = h or f"Column {j + 1}"
        base, n = h, 2
        while h in used:
            h, n = f"{base} ({n})", n + 1
        used.add(h)
        headers.append(h)
    data = [r + [""] * (width - len(r)) for r in rows[hdr_i + 1:]]
    return headers, data, preamble


_CODE_RE = re.compile(r"\b[A-Za-z]{1,6}-?\d{1,5}[A-Za-z0-9\-]*\b|\b\d{5,}\b")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def extract_codes(text: str) -> set[str]:
    return {c for c in (_norm(m) for m in _CODE_RE.findall(text or "")) if len(c) >= 3}


def image_data_url(path: str, max_side: int = IMAGE_MAX_SIDE) -> str:
    """Drawing -> data URL for the vision model (resized with Pillow when available)."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            im.draft("RGB", (max_side, max_side))
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im.copy()
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            data, mime = buf.getvalue(), "image/png"
    except ImportError:
        data = Path(path).read_bytes()
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


# ## ADDED (from 5.7) — document catalog helpers (list_documents / read_document)

DOC_TYPE_LABEL = {"manual": "manual", "datasheet": "datasheet", "table": "CSV table",
                  "drawing": "drawing", "catalogue": "catalogue",
                  "spare_parts": "spare-parts list", "web": "web page",
                  "document": "document"}
_TYPE_NAME_WORDS = (
    ("datasheet", ("datasheet", "data sheet", "datenblatt", "ficha tecnica", "hoja de datos",
                   "hoja tecnica", "product data", "tds")),
    ("manual", ("manual", "instructions", "anleitung", "instructivo", "handbuch", "bda")),
    ("spare_parts", ("spare parts", "spare part", "ersatzteil", "refacciones", "repuestos",
                     "parts list")),
    ("catalogue", ("catalog", "catalogue", "katalog", "catalogo", "brochure", "prospekt",
                   "folleto", "price list", "preisliste", "lista de precios")),
)
_TYPE_TEXT_WORDS = (
    ("manual", ("operating instructions", "instruction manual", "instructions for use",
                "user manual", "operating manual", "original instructions",
                "betriebsanleitung", "bedienungsanleitung", "gebrauchsanweisung",
                "manual de usuario", "manual de operacion", "manual de instrucciones")),
    ("datasheet", ("datasheet", "data sheet", "datenblatt", "ficha tecnica",
                   "hoja de datos")),
    ("catalogue", ("catalogue", "catalog", "katalog", "catalogo", "price list",
                   "preisliste")),
    ("spare_parts", ("spare parts list", "ersatzteilliste", "lista de refacciones",
                     "parts list")),
)
_TYPE_ALIASES = {
    "manual": "manual", "manuals": "manual", "instructions": "manual", "instructivo": "manual",
    "datasheet": "datasheet", "datasheets": "datasheet", "data sheet": "datasheet",
    "ficha": "datasheet", "ficha tecnica": "datasheet", "fichas tecnicas": "datasheet",
    "hoja de datos": "datasheet", "tds": "datasheet",
    "table": "table", "tables": "table", "csv": "table", "tabla": "table",
    "spreadsheet": "table", "stock": "table",
    "drawing": "drawing", "drawings": "drawing", "dibujo": "drawing", "image": "drawing",
    "exploded view": "drawing", "despiece": "drawing", "plano": "drawing",
    "catalogue": "catalogue", "catalog": "catalogue", "catalogo": "catalogue",
    "brochure": "catalogue", "price list": "catalogue",
    "spare parts": "spare_parts", "spare_parts": "spare_parts", "parts list": "spare_parts",
    "refacciones": "spare_parts", "repuestos": "spare_parts",
    "web": "web", "web page": "web", "document": "document", "pdf": "document",
}


def _fold(s: str) -> str:
    """lower-case, German umlauts -> ae/oe/ue, accents removed, punctuation -> spaces."""
    s = (s or "").lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue") \
        .replace("ß", "ss")
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _mkey(s: str) -> str:
    """Loose key for names: 'Schütze', 'Schuetze' and 'schutze' all give 'schutze'."""
    return _fold(s).replace("ue", "u").replace(" ", "")


def _raw_codes(text: str) -> dict[str, str]:
    """normalized code -> first spelling seen (e.g. 'vdv25' -> 'VDV-25')."""
    out: dict[str, str] = {}
    for m in _CODE_RE.findall(text or ""):
        n = _norm(m)
        if len(n) >= 3 and n not in out:
            out[n] = m
    return out


def _canon_type(s: str) -> str:
    f = _fold(s).replace("_", " ")
    if not f:
        return ""
    if f in _TYPE_ALIASES:
        return _TYPE_ALIASES[f]
    for k, v in _TYPE_ALIASES.items():
        if k in f:
            return v
    return f


def _guess_manufacturer(rel: str, head: str) -> str:
    name = " " + _fold(rel).replace("ue", "u") + " "
    text = " " + _fold(head[:2500]).replace("ue", "u") + " "
    for where in (name, text):
        for disp, keys in MANUFACTURERS:
            for k in keys:
                if (f" {k} " in where) if len(k) < 5 else (k in where):
                    return disp
    parts = rel.split("/")
    return parts[0] if len(parts) > 1 else ""


def _guess_doc_type(ext: str, rel: str, head: str) -> str:
    if ext in IMAGE_EXTS:
        return "drawing"
    if ext in CSV_EXTS:
        return "table"
    name = " " + _fold(rel) + " "
    for t, words in _TYPE_NAME_WORDS:
        if any((f" {w} " in name) if len(w) <= 3 else (w in name) for w in words):
            return t
    if ext in TEXT_EXTS and re.search(r"https?://", head[:400]):
        return "web"
    text = _fold(head[:3000])
    for t, words in _TYPE_TEXT_WORDS:
        if any(w in text for w in words):
            return t
    return "document"


def _guess_title(kind: str, rel: str, chunks: list[dict]) -> str:
    stem = Path(rel).stem
    if not chunks:
        return stem
    first = chunks[0]["text"]
    if kind == "drawing":
        m = re.search(r"(?im)^\W*(?:1\.\s*)?product[^:\n]*:\s*(.+)$", first)
        prod = m.group(1).strip(" *") if m else ""
        return f"{stem} — {prod[:90]}" if prod else stem
    if kind == "table":
        m = re.search(r"(?m)^Title: (.+)$", first)
        return m.group(1).strip()[:110] if m else stem
    for line in first.splitlines():
        t = " ".join(line.split())
        if (len(t) >= 8 and sum(ch.isalpha() for ch in t) >= 5
                and not re.match(r"(?i)^(page|seite|pagina|página)\s*\d", t)):
            return t[:110]
    return stem


def build_catalog_entry(path: str, rel: str, chunks: list[dict]) -> dict:
    """One catalog line per file, built from its chunks (no API calls)."""
    ext = Path(path).suffix.lower()
    head = "\n".join(c["text"] for c in chunks[:3])
    kind = _guess_doc_type(ext, rel, head)
    codes = _raw_codes("\n".join(c["text"] for c in chunks) + " " + Path(rel).stem)
    entry = {"rel": rel, "path": path, "ext": ext, "type": kind,
             "manufacturer": _guess_manufacturer(rel, head),
             "title": _guess_title(kind, rel, chunks), "codes": set(codes),
             "codes_show": list(codes.values())[:10], "n_codes": len(codes),
             "n_chunks": len(chunks), "snippet": "", "columns": "", "n_rows": 0}
    if kind == "table" and chunks:
        m = re.search(r"(?m)^Columns: (.+)$", chunks[0]["text"])
        entry["columns"] = m.group(1).strip() if m else ""
        m = re.search(r"(?m)^Number of rows: (\d+)", chunks[0]["text"])
        entry["n_rows"] = int(m.group(1)) if m else max(0, len(chunks) - 1)
    elif kind != "drawing" and chunks:
        entry["snippet"] = " ".join(chunks[0]["text"].split())[:180]
    return entry


def format_catalog_entry(e: dict) -> str:
    line = (f"- {e['rel']} | {DOC_TYPE_LABEL.get(e['type'], e['type'])} | "
            f"{e['manufacturer'] or '?'} | {e['title'][:100]}")
    if e["type"] == "table":
        line += f" | {e['n_rows']} rows | columns: {e['columns'][:160]}"
    if e["codes_show"]:
        more = f" (+{e['n_codes'] - len(e['codes_show'])})" if e["n_codes"] > len(e["codes_show"]) else ""
        line += f"\n    codes: {', '.join(e['codes_show'])}{more}"
    if e["snippet"]:
        line += f"\n    {e['snippet'][:160]}"
    return line


def merge_chunks(texts: list[str]) -> str:
    """Join consecutive chunks of one file, removing the splitter's overlap."""
    out = ""
    for t in texts:
        t = t or ""
        if not out:
            out = t
            continue
        probe = t[:30]
        cut = -1
        if len(probe) >= 10:
            i = out.find(probe, max(0, len(out) - 450))
            while i != -1:
                if t.startswith(out[i:]):
                    cut = len(out) - i
                    break
                i = out.find(probe, i + 1)
        out = out + t[cut:] if cut > 0 else out + "\n" + t
    return out


# ## The 3.1 hybrid retriever (engine unchanged — recursive discovery + text/CSV/drawings)

class HybridEvaluationRetriever:
    """Retriever híbrido que combina BM25 y ChromaDB con caché automático (de 3.1).

    MODIFIED: descubre los archivos de forma recursiva e indexa PDFs, texto (.txt/.md),
    tablas CSV y dibujos (.png/.jpg en subcarpetas 'drawings'). Además expone el catálogo
    de documentos, la lectura de un archivo completo y la búsqueda exacta en tablas.
    """

    def __init__(self, pdf_dir: str, top_k: int = 3, recursive: bool = RECURSIVE,
                 vision_llm=None):
        self._pdf_dir = Path(pdf_dir)
        self._top_k = top_k
        self._recursive = recursive
        self._vision_llm = vision_llm
        self._vision_ok = True
        self._retry_later: set[str] = set()
        self._captions: dict[str, str] = {}
        self._chunks: dict[str, dict] = {}
        self._chunk_ids: list[str] = []
        self._bm25 = None
        self._kind_idx: dict[str, list[int]] = {}      # kind -> chunk indices
        self._catalog: dict[str, dict] = {}            # rel -> catalog entry
        self._file_idx: dict[str, list[int]] = {}      # rel -> chunk indices (in order)
        self._chunk_rel: list[str] = []                # chunk index -> rel
        self._table_cache: dict[str, tuple] = {}       # csv path -> parsed rows
        self.embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL, api_key=check_api_key(), base_url=OPENROUTER_BASE_URL,
        )
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=self.embeddings)
        self._sync_and_index()

    # ---------- discovery ----------
    def _image_allowed(self, f: Path) -> bool:
        if not IMAGE_SUBDIRS:
            return True
        try:
            parents = f.relative_to(self._pdf_dir).parts[:-1]
        except ValueError:
            parents = f.parts[:-1]
        wanted = {s.lower() for s in IMAGE_SUBDIRS}
        return any(p.lower() in wanted for p in parents)

    @staticmethod
    def _has_sibling_image(f: Path) -> bool:
        return any(f.with_suffix(e).exists() or f.with_suffix(e.upper()).exists()
                   for e in IMAGE_EXTS)

    def _iter_pdfs(self):
        """Yield every indexable file (PDF, .txt/.md, .csv, drawings) under the corpus dir.
        Recursive when self._recursive."""
        if not self._pdf_dir.exists():
            print(f"Warning: Directory '{self._pdf_dir}' does not exist.")
            return
        globber = self._pdf_dir.rglob if self._recursive else self._pdf_dir.glob
        docs = PDF_EXTS + TEXT_EXTS + CSV_EXTS
        seen = set()
        for f in sorted(globber("*")):
            try:
                if not f.is_file() or f.name.startswith("~$"):   # skip Office lock files
                    continue
            except OSError:
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                ok = self._image_allowed(f)
            elif ext in docs:
                # a .txt next to a drawing with the same name is its "notes" sidecar
                ok = not (ext == ".txt" and self._has_sibling_image(f))
            else:
                ok = False
            key = str(f.resolve())
            if ok and key not in seen:
                seen.add(key)
                yield f

    def _rel_name(self, path: str) -> str:
        """Path relative to the corpus root (posix style) so files in different subfolders
        with the same basename don't collide as ids/sources. Falls back to the basename."""
        try:
            return Path(path).relative_to(self._pdf_dir).as_posix()
        except ValueError:
            return Path(path).name

    # ---------- per-type chunking ----------
    def _extract_text(self, path: str) -> str:
        """PDF -> pypdf; everything else (.txt/.md) -> read as UTF-8 text."""
        if path.lower().endswith(".pdf"):
            reader = PdfReader(path)
            return " ".join(pg.extract_text() for pg in reader.pages if pg.extract_text())
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    def _text_chunks(self, path: str, rel: str, splitter) -> list[dict]:
        return [{"id": f"{rel}_chunk{i}", "source": rel, "text": t, "kind": "text", "path": path}
                for i, t in enumerate(splitter.split_text(self._extract_text(path)))]

    def _csv_chunks(self, path: str, rel: str) -> list[dict]:
        headers, rows, preamble = read_csv_table(path)
        title = Path(path).stem
        overview = (f"[TABLE] {title}\nFile: {rel}\n"
                    + (f"Title: {' / '.join(preamble)}\n" if preamble else "")
                    + f"Columns: {', '.join(headers)}\nNumber of rows: {len(rows)}")
        chunks = [{"id": f"{rel}_table", "source": rel, "text": overview, "kind": "table",
                   "path": path}]
        for start in range(0, len(rows), CSV_ROWS_PER_CHUNK):
            lines = []
            for n, r in enumerate(rows[start:start + CSV_ROWS_PER_CHUNK], start=start + 1):
                pairs = [f"{h}: {v}" for h, v in zip(headers, r) if v]
                lines.append(f"Row {n}: " + " | ".join(pairs))
            chunks.append({"id": f"{rel}_row{start + 1}", "source": rel,
                           "text": f"[TABLE ROW] {title}\n" + "\n".join(lines),
                           "kind": "table", "path": path})
        return chunks

    def _describe_image(self, path: str) -> str:
        """One vision call. A failure -> indexed by name now, described on the next run."""
        if not DESCRIBE_IMAGES or self._vision_llm is None or not self._vision_ok:
            if DESCRIBE_IMAGES and self._vision_llm is not None:
                self._retry_later.add(path)
            return ""
        try:
            msg = HumanMessage(content=[
                {"type": "text", "text": IMAGE_PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_url(path)}},
            ])
            return str(self._vision_llm.invoke([msg]).content).strip()
        except Exception as e:
            self._retry_later.add(path)
            if any(code in str(e) for code in ("401", "402", "403")):
                self._vision_ok = False
                print(f"  [vision] STOPPED ({str(e)[:160]}). Drawings are indexed by name "
                      "and will be described on the next run.")
            else:
                print(f"  [vision] could not describe {Path(path).name}: {str(e)[:160]}")
            return ""

    def _describe_images_parallel(self, paths: list[str]) -> None:
        if not paths or not DESCRIBE_IMAGES or self._vision_llm is None:
            return
        print(f"[vision] describing {len(paths)} drawing(s) with {VISION_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=VISION_WORKERS) as ex:
            futs = {ex.submit(self._describe_image, p): p for p in paths}
            for n, fut in enumerate(as_completed(futs), 1):
                p = futs[fut]
                self._captions[p] = fut.result()
                print(f"  [vision {n}/{len(paths)}] {Path(p).name}: "
                      f"{'ok' if self._captions[p] else 'no description'}")

    def _image_chunks(self, path: str, rel: str) -> list[dict]:
        p = Path(path)
        name_words = re.sub(r"[_\-.]+", " ", p.stem)
        sidecar = ""
        for cand in (p.with_suffix(".txt"), p.with_suffix(".TXT")):
            if cand.exists():
                sidecar = cand.read_text(encoding="utf-8", errors="ignore").strip()
                break
        caption = self._captions[path] if path in self._captions else self._describe_image(path)
        text = ("[DRAWING] Technical drawing / exploded view (dibujo técnico, vista explosionada, "
                f"despiece)\nFile: {rel}\nName: {p.stem} ({name_words})\nFolder: {p.parent.name}\n"
                + (f"Notes: {sidecar}\n" if sidecar else "")
                + (f"Description:\n{caption}\n" if caption else ""))
        return [{"id": f"{rel}_drawing", "source": rel, "text": text, "kind": "image",
                 "path": path}]

    def _build_chunks(self, path: str, splitter) -> list[dict]:
        rel, ext = self._rel_name(path), Path(path).suffix.lower()
        if ext in CSV_EXTS:
            return self._csv_chunks(path, rel)
        if ext in IMAGE_EXTS:
            return self._image_chunks(path, rel)
        return self._text_chunks(path, rel, splitter)

    # ---------- sync ----------
    def _sync_and_index(self) -> None:
        scope = "recursively (incl. subfolders)" if self._recursive else "(top level only)"
        print(f"Syncing files in {self._pdf_dir} {scope}...")
        cached_manifest = {}
        if os.path.exists(MANIFEST_CACHE_FILE):
            try:
                with open(MANIFEST_CACHE_FILE, "rb") as f:
                    cached_manifest = pickle.load(f)
            except Exception:
                cached_manifest = {}

        current_files = {str(p): os.path.getmtime(p) for p in self._iter_pdfs()}
        kinds = Counter("csv" if Path(p).suffix.lower() in CSV_EXTS
                        else "drawing" if Path(p).suffix.lower() in IMAGE_EXTS else "pdf/text"
                        for p in current_files)
        print(f"Found {len(current_files)} indexable file(s): "
              + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))

        # mtime == -1 marks a drawing whose description failed -> retried now
        new_or_modified = [p for p, m in current_files.items()
                           if p not in cached_manifest or cached_manifest[p]["mtime"] != m]
        deleted_files = [p for p in cached_manifest if p not in current_files]
        has_changes = bool(new_or_modified or deleted_files)

        if has_changes:
            print("[Auto-Sync] Detected changes. Updating indices...")
            for path in deleted_files:
                del cached_manifest[path]
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
            self._describe_images_parallel(
                [p for p in new_or_modified if Path(p).suffix.lower() in IMAGE_EXTS])
            new_docs_to_embed = []
            for path in new_or_modified:
                try:
                    chunk_dicts = self._build_chunks(path, splitter)
                except Exception as e:
                    print(f"Failed to process {path}: {e}")
                    continue
                to_embed = chunk_dicts
                if (Path(path).suffix.lower() in CSV_EXTS
                        and len(chunk_dicts) - 1 > VECTOR_MAX_TABLE_ROWS):
                    to_embed = chunk_dicts[:1]    # big table: overview embedded, rows BM25-only
                    print(f"  {self._rel_name(path)}: {len(chunk_dicts) - 1} rows "
                          "(big table: rows BM25-only)")
                new_docs_to_embed.extend(Document(
                    page_content=c["text"],
                    metadata={"chunk_id": c["id"], "source": c["source"], "kind": c["kind"]})
                    for c in to_embed if c["text"].strip())
                mtime = -1 if path in self._retry_later else current_files[path]
                cached_manifest[path] = {"mtime": mtime, "rel": self._rel_name(path),
                                         "chunks": chunk_dicts}

            if new_docs_to_embed:
                print(f"[Auto-Sync] Embedding {len(new_docs_to_embed)} new chunks...")
                batch_size = 5000
                for i in range(0, len(new_docs_to_embed), batch_size):
                    batch = new_docs_to_embed[i:i + batch_size]
                    # ids = chunk ids -> a re-indexed file overwrites its old vectors
                    self.vector_db.add_documents(
                        batch, ids=[d.metadata["chunk_id"] for d in batch])
            with open(MANIFEST_CACHE_FILE, "wb") as f:
                pickle.dump(cached_manifest, f)
            if self._retry_later:
                print(f"  ({len(self._retry_later)} drawing(s) without description; "
                      "retried on the next run)")
            print("[Auto-Sync] Success.")
        else:
            print("Fast Start: No changes detected. Loaded from cache.")

        self._build_search_index(cached_manifest)

    def _build_search_index(self, manifest: dict) -> None:
        """BM25 + chunk table + document catalog, from the manifest (no API calls)."""
        tokenized_corpus = []
        self._chunks, self._chunk_ids, self._kind_idx = {}, [], {}
        self._catalog, self._file_idx, self._chunk_rel = {}, {}, []
        self._table_cache = {}
        for path, item in manifest.items():
            rel = item.get("rel") or (item["chunks"][0]["source"] if item.get("chunks")
                                      else self._rel_name(path))
            for chunk in item["chunks"]:
                chunk.setdefault("kind", "text")     # chunks indexed by the old 5.1
                self._kind_idx.setdefault(chunk["kind"], []).append(len(self._chunk_ids))
                self._file_idx.setdefault(rel, []).append(len(self._chunk_ids))
                self._chunk_rel.append(rel)
                self._chunks[chunk["id"]] = chunk
                self._chunk_ids.append(chunk["id"])
                tokenized_corpus.append(simple_tokenize(chunk["text"]))
            try:
                self._catalog[rel] = build_catalog_entry(path, rel, item["chunks"])
            except Exception as e:           # a catalog problem must never block the index
                print(f"  (catalog: could not describe {rel}: {e})")
        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus)
            print(f"Total active indexed chunks: {len(self._chunks)} "
                  f"({len(self._kind_idx.get('table', []))} table, "
                  f"{len(self._kind_idx.get('image', []))} drawings)")
            print(f"Catalog: {self.catalog_summary()}")
        else:
            self._bm25 = None

    # ---------- search ----------
    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if lo == hi:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def retrieve(self, query: str, top_k: int | None = None,
                 kinds: set[str] | None = None) -> list[dict]:
        """Hybrid search; `kinds` restricts to chunk types ({"image"} = drawings only)."""
        if not self._chunks or self._bm25 is None or not (query or "").strip():
            return []
        k = top_k or self._top_k
        try:
            if kinds:
                vec = self.vector_db.similarity_search_with_score(
                    query, k=CANDIDATE_POOL, filter={"kind": {"$in": sorted(kinds)}})
            else:
                vec = self.vector_db.similarity_search_with_score(query, k=CANDIDATE_POOL)
        except Exception:
            vec = self.vector_db.similarity_search_with_score(query, k=CANDIDATE_POOL)
        vec_scores = {d.metadata["chunk_id"]: (1.0 - dist) for d, dist in vec if "chunk_id" in d.metadata}

        raw = self._bm25.get_scores(simple_tokenize(query))
        idx = ([i for kd in kinds for i in self._kind_idx.get(kd, [])] if kinds
               else range(len(raw)))
        top = sorted(idx, key=lambda i: raw[i], reverse=True)[:CANDIDATE_POOL]
        bm25_scores = {self._chunk_ids[i]: raw[i] for i in top if raw[i] > 0}

        vn, bn = self._normalize(vec_scores), self._normalize(bm25_scores)
        # Keep only chunk_ids that still exist in the CURRENT corpus (and of the wanted kind).
        combined = {c: VECTOR_WEIGHT * vn.get(c, 0.0) + (1 - VECTOR_WEIGHT) * bn.get(c, 0.0)
                    for c in (set(vn) | set(bn))
                    if c in self._chunks and (not kinds or self._chunks[c]["kind"] in kinds)}
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self._chunks[c] for c, _ in ranked]

    # ---------- ADDED (from 5.7): document-level access ----------
    def resolve_doc(self, raw: str) -> tuple[str | None, list[str]]:
        """File name, partial name, title or chunk id -> (rel of one file, other candidates)."""
        r = str(raw or "").strip().strip("[]\"'` ").strip()
        if not r or not self._catalog:
            return None, []
        if r in self._catalog:
            return r, []
        if r in self._chunks:                                  # a chunk id
            return self._chunks[r].get("source"), []
        m = re.match(r"(.+?)_(?:chunk\d+|row\d+|table|drawing)$", r)
        if m and m.group(1) in self._catalog:
            return m.group(1), []
        rl = r.lower().replace("\\", "/")
        low = {k.lower(): k for k in self._catalog}
        if rl in low:
            return low[rl], []
        exact = [k for k in self._catalog
                 if Path(k).name.lower() == rl or Path(k).stem.lower() == rl]
        if exact:
            return exact[0], exact[1:5]
        key = _mkey(r)
        if len(key) >= 3:
            part = [k for k, e in self._catalog.items()
                    if key in _mkey(k) or key in _mkey(e["title"])]
            if part:
                part.sort(key=len)
                return part[0], part[1:5]
        names = {Path(k).name.lower(): k for k in self._catalog}
        titles = {e["title"].lower(): k for k, e in self._catalog.items()}
        close = difflib.get_close_matches(rl, list(names) + list(titles), n=5, cutoff=0.6)
        hits = list(dict.fromkeys(names.get(c) or titles[c] for c in close))
        return (hits[0], hits[1:]) if hits else (None, [])

    def _file_bm25(self, query: str) -> dict[str, float]:
        """Best BM25 chunk score per file for `query`."""
        best: dict[str, float] = {}
        if self._bm25 is None or not query.strip():
            return best
        raw = self._bm25.get_scores(simple_tokenize(query))
        for i, sc in enumerate(raw):
            if sc > 0:
                rel = self._chunk_rel[i]
                if sc > best.get(rel, 0.0):
                    best[rel] = float(sc)
        return best

    def list_documents(self, keyword: str = "", manufacturer: str = "", doc_type: str = "",
                       limit: int = LIST_DOCS_MAX) -> tuple[list[dict], int]:
        """Catalog lines: manufacturer / doc_type filter; keyword ranks (codes, title, text)."""
        items = list(self._catalog.values())
        if manufacturer:
            mk = _mkey(manufacturer)
            items = [e for e in items if mk in _mkey(e["manufacturer"]) or mk in _mkey(e["rel"])]
        if doc_type:
            t = _canon_type(doc_type)
            items = [e for e in items if e["type"] == t or (t == "document" and e["type"] in
                                                            ("manual", "datasheet", "document"))]
        keyword = str(keyword or "").strip()
        if keyword:
            q_codes = extract_codes(keyword)
            q_words = {w for w in simple_tokenize(keyword) if len(w) >= 3}
            text_score = self._file_bm25(keyword)
            top = max(text_score.values(), default=0.0) or 1.0
            scored = []
            for e in items:
                sc = 10.0 * len(q_codes & e["codes"])
                name_words = set(simple_tokenize(f"{e['rel']} {e['title']} {e['manufacturer']}"))
                sc += 2.0 * len(q_words & name_words)
                sc += 5.0 * text_score.get(e["rel"], 0.0) / top
                if sc > 0:
                    scored.append((sc, e))
            scored.sort(key=lambda x: x[0], reverse=True)
            items = [e for _, e in scored]
        else:
            items.sort(key=lambda e: (e["manufacturer"].lower(), e["type"], e["rel"].lower()))
        return items[:max(1, int(limit))], len(items)

    def catalog_summary(self) -> str:
        by_type = Counter(DOC_TYPE_LABEL.get(e["type"], e["type"]) for e in self._catalog.values())
        return f"{len(self._catalog)} files: " + ", ".join(f"{v} {k}" for k, v in by_type.most_common())

    def read_document(self, doc: str, part: int = 1, query: str = "",
                      max_chars: int = READ_DOC_MAX_CHARS) -> str:
        """The content of ONE file, in order, in parts of max_chars (or its passages most
        relevant to `query`)."""
        rel, alts = self.resolve_doc(doc)
        if not rel or rel not in self._catalog:
            sugg = ", ".join(alts) if alts else ""
            return (f"No document matches '{doc}'. Call list_documents to see the file "
                    "names" + (f" (similar: {sugg})" if sugg else "") + ".")
        e = self._catalog[rel]
        chunks = [self._chunks[self._chunk_ids[i]] for i in self._file_idx.get(rel, [])]
        header = (f"[DOCUMENT {rel}] {DOC_TYPE_LABEL.get(e['type'], e['type'])} | "
                  f"{e['manufacturer'] or '?'} | {e['title'][:100]}")
        if alts:
            header += f"\n(other files with a similar name: {', '.join(alts)})"
        note = ""
        if e["type"] == "table":
            rows = [c["text"].split("\n", 1)[-1] for c in chunks[1:]]
            shown = rows[:READ_TABLE_MAX_ROWS]
            full = (chunks[0]["text"] if chunks else "") + "\n" + "\n".join(shown)
            if len(rows) > len(shown):
                note = (f"\n... {len(rows) - len(shown)} more rows. Use table_lookup(code=...) "
                        "to find exact rows.")
        else:
            full = merge_chunks([c["text"] for c in chunks])
        if not full.strip():
            return header + "\n(this file has no readable text — scanned PDF?)"

        query = str(query or "").strip()
        if query and len(full) > max_chars and e["type"] not in ("table", "drawing"):
            idx = self._file_idx.get(rel, [])
            raw = self._bm25.get_scores(simple_tokenize(query)) if self._bm25 is not None else []
            order = sorted(range(len(idx)), key=lambda j: raw[idx[j]] if len(raw) else 0,
                           reverse=True)
            picked, used = [], 0
            for j in order:
                t = chunks[j]["text"]
                if used + len(t) > max_chars and picked:
                    break
                picked.append(j)
                used += len(t)
            body = merge_chunks([chunks[j]["text"] for j in sorted(picked)])[:max_chars]
            n_parts = math.ceil(len(full) / max_chars)
            return (f"{header}\n(passages of this file most relevant to '{query}'; the whole "
                    f"file has {n_parts} parts — use part=N without query to read it in "
                    f"order)\n{body}")
        n_parts = max(1, math.ceil(len(full) / max_chars))
        p = min(max(1, int(part or 1)), n_parts)
        body = full[(p - 1) * max_chars: p * max_chars]
        text = f"{header} — part {p} of {n_parts}\n{body}{note}"
        if p < n_parts:
            text += (f"\n... (continues: read_document(doc='{rel}', part={p + 1}), or give "
                     "query to jump to the relevant passages)")
        return text

    def _table_rows(self, path: str) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """(headers, rows, normalized cells) of one CSV, cached until the file changes."""
        m = os.path.getmtime(path)
        got = self._table_cache.get(path)
        if got and got[0] == m:
            return got[1], got[2], got[3]
        headers, rows, _ = read_csv_table(path)
        normed = [[_norm(c) for c in r] for r in rows]
        self._table_cache[path] = (m, headers, rows, normed)
        return headers, rows, normed

    def table_lookup(self, code: str, columns: str = "", table: str = "",
                     limit: int = TABLE_LOOKUP_MAX_ROWS) -> str:
        """Exact search of article / model codes in every CSV table."""
        codes = [c.strip() for c in re.split(r"[,;]+", str(code or "")) if c.strip()]
        codes = [c for c in codes if len(_norm(c)) >= 2]
        if not codes:
            return "table_lookup needs a code, e.g. code='123456' or code='VDV-25'."
        tables = [e for e in self._catalog.values() if e["type"] == "table"]
        if table:
            tk = _mkey(table)
            tables = [e for e in tables if tk in _mkey(e["rel"]) or tk in _mkey(e["title"])]
        if not tables:
            return ("No CSV table" + (f" matches '{table}'" if table else " in the index")
                    + ". Use semantic_search instead.")
        wanted_cols = [c.strip() for c in re.split(r"[,;]+", str(columns or "")) if c.strip()]
        found: list[tuple[int, str, int, str]] = []    # (score, rel, row index, code)
        cache: dict[str, tuple[list[str], list[list[str]]]] = {}
        problems = []
        for e in tables:
            try:
                headers, rows, normed = self._table_rows(e["path"])
            except Exception as ex:
                problems.append(f"{e['rel']}: {ex}")
                continue
            cache[e["rel"]] = (headers, rows)
            for c in codes:
                nc = _norm(c)
                for i, cells in enumerate(normed):
                    sc = 0
                    for cell, raw in zip(cells, rows[i]):
                        if not cell:
                            continue
                        if cell == nc:
                            sc = 3
                            break
                        if nc in cell:
                            if nc in {_norm(x) for x in re.split(r"[\s/,;|()]+", raw)}:
                                sc = max(sc, 2)
                            elif len(nc) >= 4:
                                sc = max(sc, 1)
                    if sc:
                        found.append((sc, e["rel"], i, c))
        if not found:
            msg = (f"No row contains {', '.join(repr(c) for c in codes)} in the "
                   f"{len(tables)} CSV table(s). Check the spelling, or use semantic_search "
                   "/ drawing_search.")
            return msg + (f"\n(unreadable tables: {'; '.join(problems)})" if problems else "")
        found.sort(key=lambda x: (-x[0], x[1], x[2]))
        seen, lines = set(), []
        label = {3: "exact", 2: "code in cell", 1: "partial"}
        n_exact = sum(1 for f in found if f[0] == 3)
        col_note = ""
        for sc, rel, i, c in found:
            if (rel, i) in seen:
                continue
            seen.add((rel, i))
            if len(lines) >= limit:
                continue
            headers, rows = cache[rel]
            pairs = [(h, v) for h, v in zip(headers, rows[i]) if v]
            if wanted_cols:
                keys = [_mkey(w) for w in wanted_cols]
                sel = [(h, v) for h, v in pairs
                       if any(k and (k in _mkey(h) or _mkey(h) in k) for k in keys)
                       or _norm(c) in _norm(v)]
                if len(sel) > sum(1 for h, v in pairs if _norm(c) in _norm(v)):
                    pairs = sel
                else:
                    col_note = (f"\n(columns '{columns}' not found; available in {rel}: "
                                f"{', '.join(headers)[:300]})")
            start = (i // CSV_ROWS_PER_CHUNK) * CSV_ROWS_PER_CHUNK
            cid = f"{rel}_row{start + 1}"
            lines.append(f"[{cid}] ({label[sc]} match for {c}) Row {i + 1}: "
                         + " | ".join(f"{h}: {v}" for h, v in pairs))
        head = (f"table_lookup({', '.join(codes)}) -> {len(seen)} matching row(s) "
                f"({n_exact} exact) in {len({f[1] for f in found})} table(s); showing "
                f"{len(lines)}")
        return head + "\n" + "\n".join(lines) + col_note


# ## Tools — semantic_search is the hybrid retriever; the rest are external / document tools

def _format_docs(docs: list[dict], n: int = 6, width: int = 400) -> str:
    if not docs:
        return "No matching documents in the local corpus."
    out = []
    for d in docs[:n]:
        w = max(width, 900) if d.get("kind") == "image" else width   # keep the parts list
        out.append(f"[{d['id']}] {d['text'][:w]}")
    return "\n".join(out)


def make_tools(retriever: HybridEvaluationRetriever) -> dict:
    """Build the tool registry. semantic_search closes over the hybrid retriever."""

    def tool_semantic_search(query: str = "", **_) -> str:
        """INTERNAL: hybrid (BM25 + vector) search over the indexed collection."""
        return _format_docs(retriever.retrieve(str(query)))

    def tool_drawing_search(query: str = "", **_) -> str:
        """INTERNAL: hybrid search over the drawings (exploded views) only."""
        if not retriever._kind_idx.get("image"):
            return ("No drawings are indexed (put .png/.jpg files in a 'drawings' subfolder "
                    "of the data folder). Use semantic_search instead.")
        return _format_docs(retriever.retrieve(str(query), top_k=3, kinds={"image"}), n=3)

    def _txt(v) -> str:
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        return str(v or "")

    def tool_list_documents(keyword: str = "", manufacturer: str = "", doc_type: str = "",
                            limit: int = LIST_DOCS_MAX, **kw) -> str:
        """INTERNAL: catalog of the collection (one line per file)."""
        keyword = _txt(keyword or kw.get("query") or kw.get("topic") or kw.get("code"))
        doc_type = _txt(doc_type or kw.get("type") or kw.get("kind"))
        manufacturer = _txt(manufacturer or kw.get("brand") or kw.get("maker"))
        try:
            limit = max(1, min(int(limit), 60))
        except (TypeError, ValueError):
            limit = LIST_DOCS_MAX
        items, total = retriever.list_documents(keyword, manufacturer, doc_type, limit)
        head = (f"list_documents(keyword='{keyword}', manufacturer='{manufacturer}', "
                f"doc_type='{doc_type}') -> showing {len(items)} of {total} matching "
                f"(collection: {retriever.catalog_summary()})"
                + (" [ranked by relevance to keyword]" if keyword else ""))
        return head + "\n" + ("\n".join(format_catalog_entry(e) for e in items)
                               or "(no document matched — try fewer filters)")

    def tool_read_document(doc: str = "", part: int = 1, query: str = "", **kw) -> str:
        """INTERNAL: read one whole file (in parts, or its most relevant passages)."""
        doc = doc or kw.get("document") or kw.get("file") or kw.get("doc_id") \
            or kw.get("doc_ids") or kw.get("name") or kw.get("id") or ""
        docs = [str(d) for d in doc] if isinstance(doc, (list, tuple)) else [str(doc)]
        docs = [d for d in docs if d.strip()][:2]
        if not docs:
            return "read_document needs doc = a file name (see list_documents)."
        try:
            part = int(part or kw.get("page") or 1)
        except (TypeError, ValueError):
            part = 1
        budget = READ_DOC_MAX_CHARS // len(docs)
        return "\n\n".join(retriever.read_document(d, part=part, query=_txt(query),
                                                   max_chars=budget) for d in docs)

    def tool_table_lookup(code: str = "", columns: str = "", table: str = "", **kw) -> str:
        """INTERNAL: exact article / model code lookup in the CSV tables."""
        code = _txt(code or kw.get("codes") or kw.get("article") or kw.get("part_number")
                    or kw.get("model") or kw.get("query"))
        columns = _txt(columns or kw.get("column") or kw.get("fields"))
        table = _txt(table or kw.get("file") or kw.get("doc"))
        return retriever.table_lookup(code, columns=columns, table=table)

    def tool_web_paper_search(query: str = "", k: int = 3, **_) -> str:
        """EXTERNAL: Semantic Scholar — papers NOT in the local collection."""
        try:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": k, "fields": "title,year,authors,abstract"},
                timeout=20,
            )
            r.raise_for_status()
            hits = r.json().get("data", []) or []
        except Exception as e:
            return f"web_paper_search failed: {e}"
        if not hits:
            return "No external papers found."
        return "\n".join(
            f"{h.get('title')} ({h.get('year')}) — "
            f"{', '.join(a.get('name', '') for a in (h.get('authors') or [])[:3])}. "
            f"{(h.get('abstract') or '')[:200]}"
            for h in hits
        )

    def tool_calculator(expression: str = "", **_) -> str:
        """EXTERNAL: safe arithmetic (no eval)."""
        ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
               ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

        def ev(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                return ops[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](ev(node.operand))
            raise ValueError("unsupported expression")

        try:
            return str(ev(ast.parse(expression, mode="eval").body))
        except Exception as e:
            return f"calc error: {e}"

    def tool_clarify(question: str = "", **_) -> str:
        """HUMAN-IN-THE-LOOP: ask the user (skipped when non-interactive)."""
        if not INTERACTIVE:
            return "(clarify skipped — non-interactive; assume the general interpretation)"
        return input(f"\n[agent asks] {question}\n> ")

    return {
        "semantic_search": tool_semantic_search,
        "drawing_search": tool_drawing_search,
        "list_documents": tool_list_documents,
        "read_document": tool_read_document,
        "table_lookup": tool_table_lookup,
        "web_paper_search": tool_web_paper_search,
        "calculator": tool_calculator,
        "clarify": tool_clarify,
    }


# main argument of each tool, used when the model sends args as a plain string
_TOOL_MAIN_ARG = {"semantic_search": "query", "drawing_search": "query",
                  "list_documents": "keyword", "read_document": "doc", "table_lookup": "code",
                  "web_paper_search": "query", "calculator": "expression",
                  "clarify": "question", "answer": "text"}


# ## choose_tool — the decision function (kept from checkpoint_5_1.py)

def choose_tool(llm: ChatOpenAI, question: str, transcript: str) -> dict:
    user = (f"Question: {question}\n\nSteps so far:\n{transcript or '(none)'}\n\n"
            "What is the next tool call?")
    raw = llm.invoke([SystemMessage(content=TOOL_SYSTEM), HumanMessage(content=user)]).content
    try:
        d = _parse_json(raw)
        if not isinstance(d, dict):
            raise ValueError("not an object")
    except (json.JSONDecodeError, ValueError):
        return {"tool": "answer", "args": {"text": "(couldn't parse a tool call)"},
                "reasoning": "parse-fail"}
    tool = str(d.get("tool") or "answer")
    args = d.get("args") or {}
    if not isinstance(args, dict):                 # e.g. "args": "VDV-25"
        args = {_TOOL_MAIN_ARG.get(tool, "query"): str(args)}
    return {"tool": tool, "args": args, "reasoning": str(d.get("reasoning", ""))}


# ## The agent as a LangGraph
#     (entry) -> decide -> [act -> decide]* -> answer -> END
# decide picks the tool; the conditional edge routes to act (loop) or answer (stop).

class AgentState(TypedDict):
    question: str
    transcript: list[str]
    iterations: int
    decision: dict
    answer: str


class HybridGraphAgent:
    def __init__(self, llm: ChatOpenAI, retriever: HybridEvaluationRetriever,
                 max_steps: int = MAX_STEPS):
        self.llm = llm
        self.tools = make_tools(retriever)
        self.max_steps = max_steps
        self.app = self._build_graph()

    # NODE: decide — call choose_tool, record the decision
    def _decide_node(self, state: AgentState) -> dict:
        d = choose_tool(self.llm, state["question"], "\n".join(state["transcript"]))
        line = f"[decide] {d['tool']}({d['args']}) :: {d['reasoning'][:70]}"
        print(f"  step {state['iterations'] + 1}: {line}")
        return {"decision": d, "transcript": state["transcript"] + [line]}

    # NODE: act — run the chosen tool, append the observation
    def _act_node(self, state: AgentState) -> dict:
        d = state["decision"]
        tool, args = d["tool"], d["args"]
        fn = self.tools.get(tool)
        try:
            obs = fn(**args) if fn else f"ERROR: unknown tool '{tool}'"
        except Exception as e:
            obs = f"tool error ({tool} {args}): {e}"
        return {"transcript": state["transcript"] + [f"[{tool} {args}] -> {obs}"],
                "iterations": state["iterations"] + 1}

    # NODE: answer — use the model's answer text, or synthesize from the notes
    def _answer_node(self, state: AgentState) -> dict:
        d = state["decision"]
        text = str((d.get("args") or {}).get("text", ""))
        if d.get("tool") == "answer" and text.strip():
            return {"answer": text}
        ctx = "\n".join(state["transcript"])
        ans = self.llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                               HumanMessage(content=f"Notes:\n{ctx}\n\nQuestion: {state['question']}")]).content
        return {"answer": ans}

    # CONDITIONAL EDGE: the decision -> a path
    def _route(self, state: AgentState) -> str:
        d = state["decision"]
        if d["tool"] == "answer":
            return "answer"                              # enough info -> stop
        if state["iterations"] >= self.max_steps:
            return "answer"                              # safety cap -> stop
        return "act"                                     # not enough -> run tool, loop

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("decide", self._decide_node)
        g.add_node("act", self._act_node)
        g.add_node("answer", self._answer_node)
        g.set_entry_point("decide")
        g.add_conditional_edges("decide", self._route, {"act": "act", "answer": "answer"})
        g.add_edge("act", "decide")                      # loop back to decide
        g.add_edge("answer", END)
        return g.compile()

    def run(self, question: str) -> dict:
        init: AgentState = {"question": question, "transcript": [], "iterations": 0,
                            "decision": {}, "answer": ""}
        return self.app.invoke(init)


# ## Baseline (the 3.1 fixed single-pass hybrid pipeline)

def fixed_pipeline_answer(llm: ChatOpenAI, retriever: HybridEvaluationRetriever, question: str) -> str:
    docs = retriever.retrieve(question)
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs) or "(no documents retrieved)"
    return llm.invoke([SystemMessage(content=FIXED_SYSTEM),
                       HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content


# ## Step 2 — agent design (for the worksheet)

EXAMPLE_PROMPTS = [
    "What is the VDV-25 dispensing valve and what is it used for?",          # direct fact
    "What are the differences between the VDV-25 and another valve model?",  # multi-step search
    "What is the operating pressure range of the VDV-25, and its midpoint?",  # search + calculator
    "Show me the exploded view of the VDV-25 and list its spare parts.",      # drawing_search
    "Which Schütze manuals and datasheets do we have in the collection?",     # list_documents
    "Read the VDV-25 datasheet and give me all its technical data.",          # read_document
    "Look up VDV-25 in the tables and give me its stock and price.",          # table_lookup
    "How do I fix the leak?",                                               # vague -> clarify
]


def my_agent_plan() -> dict:
    return {
        "tools": ["semantic_search (hybrid BM25+vector: PDFs, web pages, CSV rows, drawings)",
                  "drawing_search (hybrid, drawings only)",
                  "list_documents (catalog: manuals, datasheets, tables, drawings)",
                  "read_document (one whole file, in parts or by relevant passages)",
                  "table_lookup (exact article/model code in the CSV tables)",
                  "web_paper_search (external)", "calculator (external)", "clarify (human)",
                  "answer"],
        "stop_condition": "the agent emits `answer`, or MAX_STEPS is reached",
        "system_prompt_idea": ("table_lookup first for codes, list_documents for 'which "
                               "documents', read_document when one file holds the answer; "
                               "otherwise search the LOCAL hybrid index first; go external "
                               "only when the collection lacks the answer; calculator for "
                               "arithmetic."),
        "tasks_mode": "interactive chat loop (type your own question at the 'You:' prompt)",
        "example_prompts": EXAMPLE_PROMPTS,
    }


# ## Step 3 — evaluation: answer ONE user question with BOTH systems (fixed vs agent)

def answer_query(llm: ChatOpenAI, retriever: HybridEvaluationRetriever,
                 agent: HybridGraphAgent, question: str) -> None:
    """Run the fixed 3.1 pipeline AND the agent on one question, print + log both."""
    print("=" * 72)

    print("FIXED hybrid pipeline (Checkpoint 3.1 baseline):")
    base = fixed_pipeline_answer(llm, retriever, question)
    print(f"  {base}\n")

    print("AGENT (graph loop):")
    final = agent.run(question)
    agent_ans, steps = final["answer"], final["iterations"]
    used_external = any(t.startswith(("[web_paper_search", "[calculator"))
                        for t in final["transcript"])
    print(f"\n  answer: {agent_ans}")
    print(f"  steps={steps}  used_external_tool={used_external}")
    print("=" * 72)

    log("QUERY", f"Q: {question}\n\nFIXED:\n{base}\n\n"
                 f"AGENT ({steps} steps):\n"
                 + "\n".join(final["transcript"]) + f"\n\nANSWER:\n{agent_ans}")


def chat_loop(handler) -> None:
    """Minimal command-line chat loop: reads input, calls handler(input)."""
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        try:
            handler(user_input)
        except Exception as e:
            print(f"Error: {e}")


def run() -> None:
    llm = make_llm()
    vision_llm = make_llm(temperature=0.0, model=VISION_MODEL) if DESCRIBE_IMAGES else None
    try:
        retriever = HybridEvaluationRetriever(pdf_dir=PDF_DIR, top_k=TOP_K, vision_llm=vision_llm)
    except RecursionError:
        print(
            "\nFATAL: chromadb tripped a pathlib recursion. This happens on Python 3.13.\n"
            "Fix ONE of these, then delete the ./chroma_db_eval_recursive folder and rerun:\n"
            "  1) pip install -U chromadb langchain-chroma\n"
            "  2) use a Python 3.11/3.12 virtual environment (see the course setup)\n"
        )
        return
    agent = HybridGraphAgent(llm, retriever)

    print(f"\nCheckpoint 5.1 — hybrid + agentic graph  |  scenario: {SCENARIO}\n")
    print("Agent plan:")
    print(json.dumps(my_agent_plan(), indent=2, ensure_ascii=False))
    print("=" * 72)
    print("Ask anything about your indexed collection. Each question is answered by BOTH")
    print("the fixed 3.1 pipeline and the agent, so you can compare them for the report.")
    print("Example prompts:")
    for p in EXAMPLE_PROMPTS:
        print(f"  - {p}")
    print()

    chat_loop(lambda q: answer_query(llm, retriever, agent, q))

    print(f"Done. Full transcripts saved to {LOG_PATH.name} for your report's evaluation.")


if __name__ == "__main__":
    run()
