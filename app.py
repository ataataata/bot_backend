import os
import re
import json
import uuid
import socket
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------- Config (env-driven) ----------
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1419470822095065200/HROgUioJDnqViCJoAfaCw7x3Uc5Jy-GhI62ZxrPqO9NMvfYoB7lhp2KNRd517LjNGzzc"
SAVE_DIR        = os.getenv("SAVE_DIR", "submissions")
ALLOWED_ORIGINS = "*"

# ---------- App ----------
app = FastAPI(title="Chatbot Saver", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ---------- Utils ----------
SAFE_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = SAFE_CHARS.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "chatbot"

def find_free_port(start: int = 8081, end: int = 8099) -> int:
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return 0  # OS will pick

async def send_to_discord_as_file(filename: str, payload: Any, note: Optional[str] = None) -> None:
    if not DISCORD_WEBHOOK:
        return
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    files = {"file": (filename, text, "application/json")}
    data = {"payload_json": json.dumps({"content": note or ""})}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(DISCORD_WEBHOOK, data=data, files=files)
        if r.status_code >= 300:
            raise RuntimeError(f"Discord webhook failed: {r.status_code} {r.text[:200]}")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/chatbots")
async def save_and_notify(req: Request):
    # 1) Read JSON body
    try:
        payload: Dict[str, Any] = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 2) Build filename
    bot = payload.get("bot", {}) if isinstance(payload, dict) else {}
    base = slugify(bot.get("slug") or bot.get("name") or "chatbot")
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{base}-{ts}.json"

    # 3) Save to disk
    try:
        ensure_dir(SAVE_DIR)
        fullpath = os.path.join(SAVE_DIR, filename)
        with open(fullpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 4) Send to Discord (as file attachment with a short note)
    note = f"📥 New chatbot JSON: **{filename}**"
    discord_status = "skipped"
    try:
        await send_to_discord_as_file(filename, payload, note=note)
        discord_status = "sent"
    except Exception as e:
        # We don't fail the whole request if Discord errors; just report it.
        discord_status = f"error: {e}"

    # 5) Respond
    return {
        "ok": True,
        "file": filename,
        "saved_to": os.path.abspath(SAVE_DIR),
        "discord": discord_status,
        "ticket": str(uuid.uuid4()),
    }

# ---------- Entrypoint ----------
if __name__ == "__main__":
    port_env = os.getenv("PORT")
    port = int(port_env) if (port_env and port_env.isdigit()) else find_free_port()
    print(f"[chatbot-saver] Listening on 0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

