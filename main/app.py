from __future__ import annotations
import os, json, re, traceback
from typing import Dict, List, Tuple
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ollama import Client

# ------------------- Config -------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9000"))

DATA_DIR = os.getenv("DATA_DIR", "data")
DEFAULT_MODEL = os.getenv("MODEL", "gemma3:4b")               # LOCKED: generation always uses this
DEFAULT_EMBED = os.getenv("EMBED_MODEL", "nomic-embed-text")  # used unless meta overrides

K = int(os.getenv("K", "3"))
THRESH = float(os.getenv("THRESH", "0.30"))

# Point to your existing Ollama daemon (or leave unset for default)
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
oll = Client(host=OLLAMA_HOST) if OLLAMA_HOST else Client()

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------- FastAPI -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------- Schemas -------------------
class ChatRequest(BaseModel):
    slug: str
    message: str

class ChatReply(BaseModel):
    reply: str

# ------------------- Cache -------------------
# slug -> (qa_list, vecs ndarray, meta dict)
_CACHE: Dict[str, Tuple[List[Dict[str,str]], np.ndarray, dict]] = {}

# ------------------- Utilities -------------------
def _slug_dir(slug: str) -> str:
    return os.path.join(DATA_DIR, slug)

def _load_slug(slug: str) -> Tuple[List[Dict[str,str]], np.ndarray, dict]:
    if slug in _CACHE:
        return _CACHE[slug]
    root = _slug_dir(slug)
    qa_path = os.path.join(root, "qa.json")
    vec_path = os.path.join(root, "vecs.npy")
    meta_path = os.path.join(root, "meta.json")
    if not (os.path.exists(qa_path) and os.path.exists(vec_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Bot not prepared. Import JSON first.")
    qa = json.load(open(qa_path))
    vecs = np.load(vec_path)
    meta = json.load(open(meta_path))
    if len(qa) != vecs.shape[0]:
        raise ValueError(f"Mismatch: qa={len(qa)} vs vecs={vecs.shape[0]}")
    _CACHE[slug] = (qa, vecs, meta)
    return _CACHE[slug]

def _get_embed_model(meta: dict) -> str:
    # Allow per-bot override; to hard-lock, return DEFAULT_EMBED unconditionally.
    return meta.get("embed_model") or DEFAULT_EMBED

def _sanitize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _retrieve(qa: List[Dict[str,str]], vecs: np.ndarray, query: str, embed_model: str):
    qv = oll.embeddings(model=embed_model, prompt=_sanitize_text(query))["embedding"]
    qv = np.asarray(qv, dtype=np.float32)
    denom = (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(qv) + 1e-9) + 1e-9)
    sims = (vecs @ qv) / denom
    top = sims.argsort()[-K:][::-1]
    items = [qa[i] for i in top]
    scores = sims[top]
    # Optional threshold filter
    filtered = [(it, sc) for it, sc in zip(items, scores) if sc >= THRESH]
    if filtered:
        items = [it for it, _ in filtered]
    return items

# ------------------- Pages -------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("static/index.html")

@app.get("/b/{slug}", response_class=HTMLResponse)
def bot_page(slug: str):
    return FileResponse("static/index.html")

# ------------------- Health -------------------
@app.get("/health")
def health():
    try:
        models = oll.list()
        return {"status": "healthy", "models_available": len(models.get("models", []))}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# ------------------- Chat API -------------------
SYS_MSG = (
    "You are the website chatbot. Use the shown Q-A pairs when possible.\n"
    "Answer ONLY with the information provided below.\n"
    "If the answer is not present, reply exactly: \"I don't know.\""
)

def _build_messages(ctx_qa: List[Dict[str,str]], user_msg: str):
    if ctx_qa:
        kb = "\n".join(f"Q: {c['q']}\nA: {c['a']}" for c in ctx_qa)
        sys = SYS_MSG + "\n\n" + kb
    else:
        sys = SYS_MSG + "\n\nNo relevant context found."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_msg},
    ]

@app.post("/api/chat", response_model=ChatReply)
def chat(req: ChatRequest):
    try:
        qa, vecs, meta = _load_slug(req.slug)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        items = _retrieve(qa, vecs, req.message, embed_model=_get_embed_model(meta))
        messages = _build_messages(items, req.message)
        gen_opts = {
            "temperature": meta.get("temperature", 0.0),
            "top_p": meta.get("top_p", 0.1),
            "repeat_penalty": 1.1,
            "stop": ["\nQ:", "\nA:"],
        }
        # LOCKED generator
        res = oll.chat(model=DEFAULT_MODEL, messages=messages, stream=False, options=gen_opts)
        raw = (res.get("message", {}) or {}).get("content", "") or ""
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        raw = raw.strip() or "I don't know."
        return ChatReply(reply=raw)
    except Exception as e:
        print("Chat error:", e)
        print(traceback.format_exc())
        raise HTTPException(500, "Model error")

