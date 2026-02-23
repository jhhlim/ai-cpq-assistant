import os
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from openai import OpenAI
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Put it in backend/.env")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Put it in backend/.env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="AI-Powered CPQ Config Assistant", version="0.1.0")


# ----------------------------
# DB helpers
# ----------------------------
def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """
            )
            # 1536 dims is common for embedding models; we'll detect at insert time if needed.
            # We'll store as vector without fixed dims to keep it simple.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """
            )
            # index for similarity search (works best if embeddings have consistent dim)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            conn.commit()


@app.on_event("startup")
def _startup():
    init_db()


# ----------------------------
# Simple chunking + embedding
# ----------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + max_chars)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = end - overlap
        if i < 0:
            i = 0
        if i >= len(text):
            break
    return chunks


def embed(texts: List[str]) -> List[List[float]]:
    # OpenAI embeddings API
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# ----------------------------
# API models
# ----------------------------
class IngestRequest(BaseModel):
    title: str = Field(..., description="Doc title, e.g., 'CPQ Rules v1'")
    content: str = Field(..., description="Paste CPQ rules/config logic/doc text here")


class IngestResponse(BaseModel):
    document_id: int
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    system_hint: Optional[str] = Field(
        default="You are a CPQ configuration expert. Be precise and practical.",
        description="Optional system message to steer the assistant."
    )


class Source(BaseModel):
    document_id: int
    chunk_id: int
    title: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


# ----------------------------
# Routes
# ----------------------------
@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    chunks = chunk_text(req.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Content is empty after preprocessing.")

    embeddings = embed(chunks)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (title, content) VALUES (%s, %s) RETURNING id;",
                (req.title, req.content),
            )
            doc_id = cur.fetchone()[0]

            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (doc_id, idx, chunk, emb),
                )
            conn.commit()

    return IngestResponse(document_id=doc_id, chunks_indexed=len(chunks))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q_emb = embed([req.question])[0]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, d.title, c.chunk_text
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s
                LIMIT %s;
                """,
                (q_emb, req.top_k),
            )
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No indexed data found. Ingest documents first.")

    sources = []
    context_blocks = []
    for chunk_id, doc_id, title, chunk_text in rows:
        snippet = chunk_text[:400].replace("\n", " ")
        sources.append(Source(document_id=doc_id, chunk_id=chunk_id, title=title, snippet=snippet))
        context_blocks.append(f"[{title} | chunk {chunk_id}]\n{chunk_text}")

    context = "\n\n---\n\n".join(context_blocks)

    messages = [
        {"role": "system", "content": req.system_hint},
        {"role": "system", "content": "Use the provided context. If context is insufficient, say what is missing and ask for the rule/doc needed."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{req.question}\n\nAnswer with (1) direct answer, (2) rule references from context."}
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    answer = resp.choices[0].message.content

    return AskResponse(answer=answer, sources=sources)