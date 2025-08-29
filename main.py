"""
main.py
PDF QA Agent with Chroma + embeddings + Flan-T5
CPU-friendly version with chunk logging and SQL storage
"""

import os
import json
import uuid
import shutil
import re
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import time

# =========================
# CONFIG (defaults)
# =========================
DEFAULT_MODEL_ALIAS = "small"  # options: small, base, large
MODEL_MAP = {
    "small": "google/flan-t5-small",
    "base":  "google/flan-t5-base",
    "large": "google/flan-t5-large"
}

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_pdf_store"
CHROMA_COLLECTION = "pdf_chunks"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.2

# SQL (optional)
SQL_SERVER = "localhost,1433"
SQL_DATABASE = "PdfQA"
SQL_USERNAME = "SA"
SQL_PASSWORD = "Admin@1234"
SQL_DRIVER = "{ODBC Driver 18 for SQL Server}"

# Short system prompt (better for small models)
SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the answer is not in the context, reply: 'I don't know.' "
    "Cite sources inline like this: [cite: FILENAME p.PAGE]."
)

# =========================
# Optional SQL helpers (best-effort)
# =========================
try:
    import pyodbc
    _SQL_AVAILABLE = True
except Exception:
    _SQL_AVAILABLE = False

def sql_conn():
    return pyodbc.connect(
        f"DRIVER={SQL_DRIVER};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};"
        f"UID={SQL_USERNAME};PWD={SQL_PASSWORD};Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=5;"
    )

def init_sql():
    if not _SQL_AVAILABLE:
        return
    ddl = """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'QAResults')
    CREATE TABLE QAResults (
        qa_id INT IDENTITY(1,1) PRIMARY KEY,
        question NVARCHAR(MAX) NOT NULL,
        answer NVARCHAR(MAX) NOT NULL,
        citations NVARCHAR(MAX) NOT NULL,
        created_at DATETIME2 NOT NULL DEFAULT SYSDATETIME()
    );
    """
    try:
        with sql_conn() as conn:
            c = conn.cursor()
            c.execute(ddl)
            conn.commit()
    except Exception as e:
        print(f"⚠️ SQL init error: {e}")

def save_qa(question: str, answer: str, citations: List[Dict]):
    if not _SQL_AVAILABLE:
        return
    try:
        with sql_conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO QAResults (question, answer, citations) VALUES (?, ?, ?)",
                (question, answer, json.dumps(citations)),
            )
            conn.commit()
    except Exception as e:
        print(f"⚠️ Could not save QA to SQL: {e}")

# =========================
# Chroma helpers
# =========================
def get_chroma_collection():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    coll = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return coll

def reset_chroma():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    print("🗑️  Chroma store reset.")

# =========================
# Embedding helpers
# =========================
def load_embedder():
    print("🔹 Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    print("✅ Embedding model loaded")
    return embedder

def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).tolist()

# =========================
# LLM helpers
# =========================
def load_llm(model_alias: str = None):
    if model_alias is None:
        model_alias = DEFAULT_MODEL_ALIAS
    model_name = MODEL_MAP.get(model_alias, MODEL_MAP[DEFAULT_MODEL_ALIAS])
    print(f"🔹 Loading LLM model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ LLM loaded on {device}")
    return tokenizer, model, device

# =========================
# PDF chunking
# =========================
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        pages.append((i + 1, text))
    return pages

# =========================
# Ingestion
# =========================
def ingest_pdfs(folder_or_file: str):
    coll = get_chroma_collection()
    embedder = load_embedder()

    pdf_files = []
    if os.path.isdir(folder_or_file):
        for root, _, files in os.walk(folder_or_file):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))
    else:
        if folder_or_file.lower().endswith(".pdf"):
            pdf_files.append(folder_or_file)

    if not pdf_files:
        print("⚠️ No PDFs found to ingest.")
        return

    total_chunks = 0
    for pdf in pdf_files:
        filename = os.path.basename(pdf)
        pages = extract_pdf_pages(pdf)
        print(f"📥 Ingesting {filename} ({len(pages)} pages)...")
        for page_num, page_text in tqdm(pages, desc=f"{filename} pages"):
            if not page_text or not page_text.strip():
                continue
            page_chunks = chunk_text(page_text)
            if not page_chunks:
                continue
            embeddings = embed_texts(embedder, page_chunks)

            ids, metas = [], []
            for idx in range(len(page_chunks)):
                ids.append(str(uuid.uuid4()))
                metas.append({"filename": filename, "page_number": page_num, "chunk_index": idx})

            coll.add(ids=ids, documents=page_chunks, metadatas=metas, embeddings=embeddings)
            total_chunks += len(page_chunks)

    print(f"✅ Ingestion complete. Chroma count: {get_chroma_collection().count()}")
    print(f"Total chunks added: {total_chunks}")

# =========================
# Context building + postprocess fallback
# =========================
def build_context(snippets: List[Dict]) -> str:
    lines = []
    for i, s in enumerate(snippets, 1):
        fn = s["metadata"].get("filename", "unknown")
        pg = s["metadata"].get("page_number", "?")
        txt = s["document"].replace("\n", " ").strip()
        txt_short = txt[:1000]
        lines.append(f"[{i}] Source: {fn} (p.{pg})\n{txt_short}")
    return "\n\n".join(lines)

def _is_bad_short_answer(ans: str) -> bool:
    if not ans:
        return True
    s = ans.strip()
    # tokens like "[1]" or "[3]" or single digit, or extremely short output
    if re.fullmatch(r"\[\d+\]", s):
        return True
    if len(s) < 20:
        return True
    return False

def _fallback_summarize_from_chunks(retrieved: List[Dict], max_sentences=3) -> str:
    # Take first meaningful sentences from retrieved chunks and assemble a fallback answer with citations.
    sentences = []
    seen = set()
    for r in retrieved:
        text = r["document"].replace("\n", " ").strip()
        # split into sentences (naive)
        parts = re.split(r'(?<=[.!?])\s+', text)
        for p in parts:
            p = p.strip()
            if len(p) > 40:
                key = p[:80]
                if key not in seen:
                    seen.add(key)
                    fn = r["metadata"].get("filename", "unknown")
                    pg = r["metadata"].get("page_number", "?")
                    sentences.append(f"{p} [cite: {fn} p.{pg}]")
                    break
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        return "I don't know."
    return " ".join(sentences)

# =========================
# Pipeline (load + ask)
# =========================
def load_pipeline(model_alias: str = None):
    embedder = load_embedder()
    tokenizer, llm, device = load_llm(model_alias)
    coll = get_chroma_collection()
    return {"embedder": embedder, "tokenizer": tokenizer, "llm": llm, "device": device, "coll": coll}

def ask_with_pipeline(pipeline: dict, question: str, top_k: int = TOP_K, return_retrieved: bool = False):
    start_time = time.time()
    embedder = pipeline["embedder"]
    tokenizer = pipeline["tokenizer"]
    llm = pipeline["llm"]
    device = pipeline["device"]
    coll = pipeline["coll"]

    # Embed question
    q_vec = embed_texts(embedder, [question])[0]

    # Retrieve top K chunks
    results = coll.query(query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas"])
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    retrieved = [{"document": d, "metadata": m} for d, m in zip(docs, metas)]

    # duplicate
    seen_chunks = set()
    unique_retrieved = []
    for r in retrieved:
        key = (r['metadata'].get('filename'), r['metadata'].get('page_number'), r['document'][:200])
        if key not in seen_chunks:
            seen_chunks.add(key)
            unique_retrieved.append(r)
    retrieved = unique_retrieved

    if not retrieved:
        answer_text = "I don't know."
        citations = []
    else:
        # build prompt
        context_block = build_context(retrieved)
        prompt_text = (
            f"{SYSTEM_PROMPT}\n\nContext:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        )

        # tokenize & run model
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            output = llm.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE>0,
                temperature=TEMPERATURE
            )
        answer_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # if LLM output is clearly bad/short like "[1]", use fallback
        if _is_bad_short_answer(answer_text):
            answer_text = _fallback_summarize_from_chunks(retrieved, max_sentences=3)

        # collect unique citations
        citations, seen = [], set()
        for r in retrieved:
            fn = r["metadata"].get("filename", "unknown")
            pg = r["metadata"].get("page_number", "?")
            if (fn, pg) not in seen:
                seen.add((fn, pg))
                citations.append({"filename": fn, "page": pg})

    # Save to SQL (best-effort)
    try:
        init_sql()
        save_qa(question, answer_text, citations)
    except Exception:
        pass

    elapsed = time.time() - start_time
    result = {"answer": answer_text, "citations": citations, "elapsed": elapsed}
    if return_retrieved:
        return result, retrieved
    return result

# =========================
# CLI (simple)
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PDF QA Agent (main)")
    parser.add_argument("--ingest", type=str, help="Folder or single PDF to ingest")
    parser.add_argument("--ask", type=str, help="Ask a single question and exit")
    parser.add_argument("--reset", action="store_true", help="Reset Chroma store")
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()), default=DEFAULT_MODEL_ALIAS, help="Model alias to use (small/base/large)")
    parser.add_argument("--topk", type=int, default=TOP_K, help="Top-K chunks to retrieve")
    args = parser.parse_args()

    if args.reset:
        reset_chroma()

    if args.ingest:
        ingest_pdfs(args.ingest)

    if args.ask:
        print("⏳ Loading pipeline...")
        pipeline = load_pipeline(model_alias=args.model)
        print("✅ Pipeline ready")
        result, retrieved = ask_with_pipeline(pipeline, args.ask, top_k=args.topk, return_retrieved=True)
        print("\n=== Answer ===")
        print(result["answer"])
        if result["citations"]:
            print("\n=== Citations ===")
            for c in result["citations"]:
                print(f"- {c['filename']} p.{c['page']}")
        if retrieved:
            print("\n=== Retrieved Chunks ===")
            for r in retrieved:
                fn = r['metadata'].get('filename'); pg = r['metadata'].get('page_number')
                txt = r['document'][:300].replace("\n"," ").strip()
                print(f"- {fn} p.{pg}: {txt}...")
        print(f"\n⏱ Elapsed: {result['elapsed']:.2f}s")
