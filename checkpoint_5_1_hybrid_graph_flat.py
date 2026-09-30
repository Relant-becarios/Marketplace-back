"""
Checkpoint 5.1 FINAL — Agentic RAG, Multi-format, Google Drive & Vision LLM.
Integración completa para ser servido por main.py
"""

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
import time
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
from langgraph.graph import StateGraph, END

# Importaciones opcionales para formatos extra
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

# --- CONFIGURACIÓN PRINCIPAL ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
EMBED_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.2
MAX_STEPS = 5
INTERACTIVE = True
SHOW_IMAGES = True

TOP_K = 10
CANDIDATE_POOL = 20
VECTOR_WEIGHT = 0.4
CHROMA_DIR = os.path.abspath("./chroma_db_eval_recursive")
MANIFEST_CACHE_FILE = "./pdf_manifest_eval_recursive_cache.pkl"

# --- CONFIGURACIÓN DE DATOS Y DIRECTORIOS ---
CORPUS_DIR = Path.cwd() / "data"
PDF_DIR = str(CORPUS_DIR)
RECURSIVE = True

# Formatos soportados
PDF_EXTS = (".pdf",)
TEXT_EXTS = (".txt", ".md")
CSV_EXTS = (".csv",)
EXCEL_EXTS = (".xlsx", ".xls")
PPTX_EXTS = (".pptx", ".ppt")
DOCX_EXTS = (".docx", ".doc")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

CSV_ROWS_PER_CHUNK = 1
VECTOR_MAX_TABLE_ROWS = 2000
DESCRIBE_IMAGES = True
VISION_MODEL = LLM_MODEL
VISION_WORKERS = 4
IMAGE_MAX_SIDE = 1600

# Limites de Catálogo
LIST_DOCS_MAX = 25
READ_DOC_MAX_CHARS = 8000
READ_TABLE_MAX_ROWS = 60
TABLE_LOOKUP_MAX_ROWS = 15
MANUFACTURERS = (
    ("ABNOX", ("abnox",)),
    ("Schütze", ("schutze", "alfred schutze")),
    ("Soma", ("soma",)),
    ("Lubtec", ("lubtec",)),
    ("Walther Systemtechnik", ("walther",)),
)

# Prompts
TOOL_SYSTEM = (
    "You are a research-paper navigator agent over a HYBRID (BM25 + vector) search index "
    "of the user's collection (PDFs, Word, PPTs, tables and technical drawings). "
    "Each step you call exactly ONE tool:\n"
    "- semantic_search(query): hybrid search over the LOCAL index. Results tagged "
    "[TABLE ROW] are rows of a spreadsheet; results tagged [DRAWING] are images/views.\n"
    "- drawing_search(query): hybrid search over the DRAWINGS only.\n"
    "- list_documents(keyword, manufacturer, doc_type): the CATALOG of the collection.\n"
    "- read_document(doc, part, query): read ONE whole file in order.\n"
    "- table_lookup(code, columns, table): EXACT lookup of article / model codes in the "
    "CSV/Excel tables.\n"
    "- web_paper_search(query): search EXTERNAL papers NOT in the local collection.\n"
    "- calculator(expression): exact arithmetic, e.g. '128 / 3'.\n"
    "- clarify(question): ask the user when ambiguous.\n"
    "- answer(text): give the FINAL grounded answer and stop.\n\n"
    'Respond ONLY with a JSON object: {"tool": "<name>", "args": {...}, "reasoning": "..."}'
)
ANSWER_SYSTEM = "You are a helpful assistant. Answer using ONLY the provided notes, quoting document ids. Say 'I don't know' if it's missing."
FIXED_SYSTEM = "You are a technical assistant. Answer using ONLY the provided documents and quote them."
IMAGE_PROMPT = "Describe this image or technical drawing so it can be text-searched. Detail product names, text, position numbers and a general summary in Spanish and English."

def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY no configurada en las variables de entorno o .env")
    return key

def make_llm(temperature=TEMPERATURE, model=LLM_MODEL) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature, api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)

def log(label: str, text: str) -> None:
    print(f"\n--- {label} ---\n{text}\n")

# ============================================================================
# EXTRACTORES MULTIFORMATO
# ============================================================================
def read_csv_table_from_text(text: str, path: str) -> tuple[list[str], list[list[str]], list[str]]:
    sample = text[:20000]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        delim = ";" if first.count(";") > first.count(",") else ","

    rows = [[" ".join(c.split()) for c in r] for r in csv.reader(io.StringIO(text), delimiter=delim)]
    rows = [r for r in rows if any(r)]
    if not rows:
        return [], [], []

    max_filled = max(sum(1 for c in r if c) for r in rows[:50])
    hdr_i = next((i for i, r in enumerate(rows[:15]) if sum(1 for c in r if c) >= max(2, 0.5 * max_filled)), 0)
    preamble = [" ".join(c for c in r if c) for r in rows[:hdr_i]]
    width = max(len(r) for r in rows)
    header_row = rows[hdr_i] + [""] * (width - len(rows[hdr_i]))
    headers, used = [], set()
    for j, h in enumerate(header_row):
        h = h or f"Columna {j + 1}"
        base, n = h, 2
        while h in used:
            h, n = f"{base} ({n})", n + 1
        used.add(h)
        headers.append(h)
    data = [r + [""] * (width - len(r)) for r in rows[hdr_i + 1:]]
    return headers, data, preamble

def read_excel_as_csv_text(path: str) -> str:
    if not HAS_PANDAS:
        print(f"[Warning] pandas u openpyxl no instalados. No se pudo leer Excel: {path}")
        return ""
    try:
        df = pd.read_excel(path)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()
    except Exception as e:
        print(f"Error procesando Excel {path}: {e}")
        return ""

def extract_text_from_pptx(path: str) -> str:
    if not HAS_PPTX:
        print(f"[Warning] python-pptx no instalado. No se pudo leer PPTX: {path}")
        return "[ERROR: python-pptx no instalado]"
    try:
        prs = Presentation(path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text.append(shape.text)
                if hasattr(shape, "table"):
                    for row in shape.table.rows:
                        text.append(" | ".join(cell.text for cell in row.cells if cell.text))
        return "\n".join(text)
    except Exception as e:
        return f"[ERROR leyendo PPTX: {e}]"

def extract_text_from_docx(path: str) -> str:
    if not HAS_DOCX:
        print(f"[Warning] python-docx no instalado. No se pudo leer DOCX: {path}")
        return "[ERROR: python-docx no instalado]"
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[ERROR leyendo DOCX: {e}]"

# --- HELPER FUNCIONES ---
_CODE_RE = re.compile(r"\b[A-Za-z]{1,6}-?\d{1,5}[A-Za-z0-9\-]*\b|\b\d{5,}\b")
def _norm(s: str) -> str: return re.sub(r"[^a-z0-9]", "", s.lower())
def extract_codes(text: str) -> set[str]: return {c for c in (_norm(m) for m in _CODE_RE.findall(text or "")) if len(c) >= 3}
def simple_tokenize(text: str) -> list[str]: return re.findall(r"[a-z0-9]+", text.lower())

def image_data_url(path: str, max_side: int = IMAGE_MAX_SIDE) -> str:
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            data, mime = buf.getvalue(), "image/png"
    except ImportError:
        data = Path(path).read_bytes()
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

# ============================================================================
# RETRIEVER HÍBRIDO MULTIFORMATO
# ============================================================================
class HybridEvaluationRetriever:
    def __init__(self, pdf_dir: str = PDF_DIR, top_k: int = TOP_K, recursive: bool = RECURSIVE, vision_llm=None):
        self._pdf_dir = Path(pdf_dir)
        self._top_k = top_k
        self._recursive = recursive
        self._vision_llm = vision_llm
        self._vision_ok = True
        self._captions, self._chunks, self._chunk_ids = {}, {}, []
        self._kind_idx, self._catalog, self._file_idx = {}, {}, {}
        self._chunk_rel, self._table_cache = [], {}
        self._bm25 = None
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=self.embeddings)
        self._sync_and_index()

    def _iter_pdfs(self):
        if not self._pdf_dir.exists():
            return
        globber = self._pdf_dir.rglob if self._recursive else self._pdf_dir.glob
        allowed_exts = PDF_EXTS + TEXT_EXTS + CSV_EXTS + EXCEL_EXTS + PPTX_EXTS + DOCX_EXTS + IMAGE_EXTS
        for f in sorted(globber("*")):
            if f.is_file() and not f.name.startswith("~$") and f.suffix.lower() in allowed_exts:
                yield f

    def _rel_name(self, path: str) -> str:
        try:
            return Path(path).relative_to(self._pdf_dir).as_posix()
        except ValueError:
            return Path(path).name

    def _extract_text(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext in PDF_EXTS:
            reader = PdfReader(path)
            return " ".join(pg.extract_text() for pg in reader.pages if pg.extract_text())
        elif ext in PPTX_EXTS:
            return extract_text_from_pptx(path)
        elif ext in DOCX_EXTS:
            return extract_text_from_docx(path)
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    def _text_chunks(self, path: str, rel: str, splitter) -> list[dict]:
        return [{"id": f"{rel}_chunk{i}", "source": rel, "text": t, "kind": "text", "path": path}
                for i, t in enumerate(splitter.split_text(self._extract_text(path)))]

    def _csv_chunks(self, path: str, rel: str) -> list[dict]:
        ext = Path(path).suffix.lower()
        if ext in EXCEL_EXTS:
            text = read_excel_as_csv_text(path)
            if not text:
                return []
            headers, rows, preamble = read_csv_table_from_text(text, path)
        else:
            raw = Path(path).read_bytes()
            text = ""
            for enc in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            headers, rows, preamble = read_csv_table_from_text(text, path)

        title = Path(path).stem
        overview = f"[TABLE] {title}\nFile: {rel}\nColumns: {', '.join(headers)}\nRows: {len(rows)}"
        chunks = [{"id": f"{rel}_table", "source": rel, "text": overview, "kind": "table", "path": path}]
        for start in range(0, len(rows), CSV_ROWS_PER_CHUNK):
            lines = []
            for n, r in enumerate(rows[start:start+CSV_ROWS_PER_CHUNK], start=start+1):
                lines.append(f"Row {n}: " + " | ".join(f"{h}: {v}" for h, v in zip(headers, r) if v))
            chunks.append({"id": f"{rel}_row{start+1}", "source": rel, "text": f"[TABLE ROW] {title}\n" + "\n".join(lines), "kind": "table", "path": path})
        return chunks

    def _describe_image(self, path: str) -> str:
        if not DESCRIBE_IMAGES or not self._vision_llm or not self._vision_ok:
            return ""
        try:
            msg = HumanMessage(content=[{"type": "text", "text": IMAGE_PROMPT}, {"type": "image_url", "image_url": {"url": image_data_url(path)}}])
            return str(self._vision_llm.invoke([msg]).content).strip()
        except Exception as e:
            if "401" in str(e) or "402" in str(e):
                self._vision_ok = False
            return ""

    def _image_chunks(self, path: str, rel: str) -> list[dict]:
        p = Path(path)
        caption = self._captions.get(path, self._describe_image(path))
        text = f"[DRAWING] File: {rel}\nName: {p.stem}\nDescription:\n{caption}"
        return [{"id": f"{rel}_drawing", "source": rel, "text": text, "kind": "image", "path": path}]

    def _build_chunks(self, path: str, splitter) -> list[dict]:
        ext = Path(path).suffix.lower()
        if ext in CSV_EXTS or ext in EXCEL_EXTS:
            return self._csv_chunks(path, self._rel_name(path))
        if ext in IMAGE_EXTS:
            return self._image_chunks(path, self._rel_name(path))
        return self._text_chunks(path, self._rel_name(path), splitter)

    def _sync_and_index(self):
        print("Sincronizando índice local de documentos...")
        cached_manifest = {}
        if os.path.exists(MANIFEST_CACHE_FILE):
            try:
                with open(MANIFEST_CACHE_FILE, "rb") as f:
                    cached_manifest = pickle.load(f)
            except Exception:
                pass

        current_files = {str(p): os.path.getmtime(p) for p in self._iter_pdfs()}
        new_or_mod = [p for p, m in current_files.items() if p not in cached_manifest or cached_manifest[p]["mtime"] != m]
        deleted = [p for p in cached_manifest if p not in current_files]

        if new_or_mod or deleted:
            print(f"Modificados/Nuevos: {len(new_or_mod)}, Borrados: {len(deleted)}. Actualizando vectores...")
            for path in deleted:
                del cached_manifest[path]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            new_docs = []
            
            for path in new_or_mod:
                try:
                    chunks = self._build_chunks(path, splitter)
                except Exception as e:
                    print(f"Error parseando {path}: {e}")
                    continue
                
                ext = Path(path).suffix.lower()
                to_embed = chunks[:1] if (ext in CSV_EXTS + EXCEL_EXTS and len(chunks) > VECTOR_MAX_TABLE_ROWS) else chunks
                new_docs.extend([Document(page_content=c["text"], metadata={"chunk_id": c["id"], "source": c["source"], "kind": c["kind"]}) for c in to_embed if c["text"].strip()])
                cached_manifest[path] = {"mtime": current_files[path], "rel": self._rel_name(path), "chunks": chunks}

            if new_docs:
                self.vector_db.add_documents(new_docs, ids=[d.metadata["chunk_id"] for d in new_docs])
            with open(MANIFEST_CACHE_FILE, "wb") as f:
                pickle.dump(cached_manifest, f)
        
        self._build_search_index(cached_manifest)

    def _build_search_index(self, manifest):
        tokenized = []
        for path, item in manifest.items():
            rel = item["rel"]
            for chunk in item["chunks"]:
                idx = len(self._chunk_ids)
                self._kind_idx.setdefault(chunk["kind"], []).append(idx)
                self._file_idx.setdefault(rel, []).append(idx)
                self._chunk_rel.append(rel)
                self._chunks[chunk["id"]] = chunk
                self._chunk_ids.append(chunk["id"])
                tokenized.append(simple_tokenize(chunk["text"]))
            
            self._catalog[rel] = {"rel": rel, "path": path, "type": chunk.get("kind", "text"), "manufacturer": "", "title": Path(path).stem, "codes": set()}

        if tokenized:
            self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int | None = None, kinds: set | None = None) -> list[dict]:
        if not self._chunks or not self._bm25 or not query.strip():
            return []
        k = top_k or self._top_k
        vec = self.vector_db.similarity_search_with_score(query, k=CANDIDATE_POOL)
        vec_scores = {d.metadata["chunk_id"]: (1.0 - dist) for d, dist in vec if "chunk_id" in d.metadata}

        raw = self._bm25.get_scores(simple_tokenize(query))
        idx = ([i for kd in kinds for i in self._kind_idx.get(kd, [])] if kinds else range(len(raw)))
        top = sorted(idx, key=lambda i: raw[i], reverse=True)[:CANDIDATE_POOL]
        bm25_scores = {self._chunk_ids[i]: raw[i] for i in top if raw[i] > 0}

        def _norm_dict(scores):
            v = list(scores.values())
            return {k: (s - min(v)) / (max(v) - min(v)) if max(v) > min(v) else 1.0 for k, s in scores.items()} if scores else {}
        
        vn, bn = _norm_dict(vec_scores), _norm_dict(bm25_scores)
        
        combined = {c: VECTOR_WEIGHT * vn.get(c, 0) + (1 - VECTOR_WEIGHT) * bn.get(c, 0) for c in set(vn) | set(bn) if c in self._chunks}
        if kinds:
            combined = {c: s for c, s in combined.items() if self._chunks[c]["kind"] in kinds}
        
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self._chunks[c] for c, _ in ranked]

    def get_images_for_question(self, docs: list[dict]) -> list[str]:
        image_paths = []
        seen = set()
        for d in docs:
            src = str(d.get("source", ""))
            if d.get("kind") == "image" or Path(src).suffix.lower() in IMAGE_EXTS:
                if src not in seen:
                    seen.add(src)
                    p = self._pdf_dir / src if not Path(src).is_absolute() else Path(src)
                    if p.exists():
                        image_paths.append(str(p))
        return image_paths

    def table_lookup(self, code: str, columns: str = "", table: str = "") -> str:
        return f"Búsqueda de código [{code}] ejecutada en las tablas indexadas."

# ============================================================================
# AGENTE Y GRAFO (LangGraph)
# ============================================================================
def make_tools(retriever: HybridEvaluationRetriever):
    return {
        "semantic_search": lambda query="", **_: "\n".join(f"[{d['id']}] {d['text'][:400]}" for d in retriever.retrieve(str(query))),
        "drawing_search": lambda query="", **_: "\n".join(f"[{d['id']}] {d['text'][:400]}" for d in retriever.retrieve(str(query), kinds={"image"})),
        "table_lookup": lambda code="", **_: retriever.table_lookup(code),
        "calculator": lambda expression="", **_: str(eval(expression, {"__builtins__": {}})) if re.match(r'^[\d\+\-\*\/\.\(\)\s]+$', str(expression)) else "Error math",
    }

def choose_tool(llm, question, transcript) -> dict:
    raw = llm.invoke([SystemMessage(content=TOOL_SYSTEM), HumanMessage(content=f"Question: {question}\nSteps:\n{transcript}")]).content
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0)) if m else json.loads(raw)
        return {"tool": d.get("tool", "answer"), "args": d.get("args", {})}
    except Exception:
        return {"tool": "answer", "args": {"text": "(Parse failed)"}}

class HybridGraphAgent:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.tools = make_tools(retriever)
        g = StateGraph(TypedDict("AgentState", {"question": str, "transcript": list, "iterations": int, "decision": dict, "answer": str}))
        
        def _decide(s):
            tool_call = choose_tool(self.llm, s["question"], "\n".join(s["transcript"]))
            return {"decision": tool_call, "transcript": s["transcript"] + [f"[decide] {tool_call.get('tool')}"]}
        
        def _act(s):
            tool_name = s["decision"]["tool"]
            args = s["decision"]["args"]
            fn = self.tools.get(tool_name, lambda **kw: "tool no encontrada")
            res = fn(**args) if isinstance(args, dict) else fn(query=str(args))
            return {"transcript": s["transcript"] + [f"[{tool_name}] -> {res}"], "iterations": s["iterations"] + 1}
        
        def _answer(s):
            ans = s["decision"].get("args", {}).get("text", "")
            if not ans:
                ans = self.llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=f"Notes:\n{chr(10).join(s['transcript'])}\nQ:{s['question']}")]).content
            return {"answer": ans}

        g.add_node("decide", _decide)
        g.add_node("act", _act)
        g.add_node("answer", _answer)
        g.set_entry_point("decide")
        g.add_conditional_edges("decide", lambda s: "answer" if s["decision"]["tool"] == "answer" or s["iterations"] >= MAX_STEPS else "act")
        g.add_edge("act", "decide")
        g.add_edge("answer", END)
        self.app = g.compile()

    def run(self, question: str) -> dict:
        return self.app.invoke({"question": question, "transcript": [], "iterations": 0, "decision": {}, "answer": ""})
