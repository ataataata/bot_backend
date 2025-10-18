import os, sys, json, re
from typing import List, Dict
import numpy as np
from ollama import Client

DATA_DIR = os.getenv("DATA_DIR", "data")
DEFAULT_EMBED = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
oll = Client(host=OLLAMA_HOST) if OLLAMA_HOST else Client()

def sanitize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def embed_batch(texts: List[str], model: str) -> np.ndarray:
    vecs = []
    for t in texts:
        t = sanitize(t)
        emb = oll.embeddings(model=model, prompt=t)["embedding"]
        vecs.append(np.asarray(emb, dtype=np.float32))
    return np.vstack(vecs).astype(np.float32)

def import_one(json_path: str) -> str:
    with open(json_path, "r") as f:
        payload = json.load(f)

    bot = payload["bot"]
    pairs = payload["pairs"]
    slug = sanitize(bot["slug"])
    if not slug:
        raise ValueError("bot.slug is required")

    root = os.path.join(DATA_DIR, slug)
    ensure_dir(root)

    # Normalize pairs -> qa.json
    qa: List[Dict[str,str]] = []
    for p in pairs:
        q = sanitize(p["q"])
        a = sanitize(p["a"])
        if q and a:
            qa.append({"q": q, "a": a})
    if not qa:
        raise ValueError("No valid Q/A pairs after normalization")

    with open(os.path.join(root, "qa.json"), "w") as f:
        json.dump(qa, f, indent=2, ensure_ascii=False)

    # Build embeddings over questions
    embed_model = bot.get("embed_model") or DEFAULT_EMBED
    V = embed_batch([x["q"] for x in qa], model=embed_model)
    np.save(os.path.join(root, "vecs.npy"), V)

    # Save meta (generation model is locked by app.py, we only store knobs/info)
    meta = {
        "name": bot.get("name", ""),
        "lab": bot.get("lab", ""),
        "owner_email": bot.get("owner_email", ""),
        "description": bot.get("description", ""),
        "slug": slug,
        "embed_model": embed_model,
        "generator_model_received": bot.get("model"),
        "temperature": bot.get("temperature", 0.0),
        "top_p": bot.get("top_p", 0.1),
        "created_at": payload.get("created_at"),
        "version": payload.get("version"),
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Imported {json_path} -> /b/{slug}  (pairs={len(qa)})")
    return slug

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_from_file.py <path.json> OR comma-separated list")
        sys.exit(1)
    ensure_dir(DATA_DIR)
    paths = []
    for arg in sys.argv[1:]:
        paths.extend([p.strip() for p in arg.split(",") if p.strip()])
    slugs = []
    for p in paths:
        slugs.append(import_one(p))
    # Print a simple machine-friendly summary line for run.sh
    print("IMPORT_DONE:" + ",".join(slugs))

