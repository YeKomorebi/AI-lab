"""
server.py — AI Lab 工作台 API

端点：
  GET  /                          前端页面
  GET  /api/agents                所有可用角色
  POST /api/agents                创建自定义角色
  GET  /api/rooms                 所有房间配置
  POST /api/room/{room_id}/chat   发送消息，SSE 流式响应
  POST /api/room/{room_id}/transfer  生成移交摘要
  GET  /api/room/{room_id}/stream/{job_id}  SSE 流

启动：
  uvicorn app.server:app --reload --port 8000
"""

import asyncio
import base64
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.orchestrator import (
    AGENTS as AGENTS_DICT,
    ROOMS,
    room_chat,
    generate_transfer_summary,
    transfer_greeting,
)

app = FastAPI(title="AI Lab 工作台")

STATIC_DIR = Path(__file__).parent / "static"
SKILLS_DIR = Path(__file__).parent.parent / "skills" / "colleague"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── 内存存储 ─────────────────────────────────────────────────────────────────
# 每个 session 对应一个房间的完整对话历史
_sessions: dict[str, list[dict]] = {room_id: [] for room_id in ROOMS}
# context_summary：从上个房间带来的移交摘要
_context: dict[str, str | None] = {room_id: None for room_id in ROOMS}
# SSE job 队列
_jobs: dict[str, dict] = {}


# ── 数据模型 ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    model:   str = "claude-sonnet-4-6"
    at:      str | None = None
    attachments: list[dict] = []   # [{type, filename, content/data, mime?}]

class TransferRequest(BaseModel):
    to_room_id: str
    model:      str = "claude-sonnet-4-6"

class AgentCreateRequest(BaseModel):
    name:    str
    role:    str
    focus:   str = ""
    persona: str
    work:    str = ""

class AgentUpdateRequest(BaseModel):
    name:    str
    role:    str
    focus:   str = ""
    persona: str
    work:    str = ""
    rooms:   list[str] = []


# ── 文件解析 ─────────────────────────────────────────────────────────────────

def _extract_text(filename: str, data: bytes) -> str:
    """从上传文件提取文本内容。图片返回空字符串（走 Vision 通道）。"""
    ext = Path(filename).suffix.lower()
    if ext in (".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml",
               ".html", ".htm", ".js", ".ts", ".py", ".java", ".go",
               ".c", ".cpp", ".h", ".rs", ".sh", ".sql"):
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode("latin-1", errors="replace")
    if ext == ".pdf":
        try:
            import io
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return f"[PDF 解析失败: {e}]"
    if ext in (".docx",):
        try:
            import io
            import docx as _docx
            doc = _docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f"[DOCX 解析失败: {e}]"
    return ""   # 图片等二进制文件，文本提取不处理


# ── 页面路由 ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = STATIC_DIR / "index.html"
    if not html.exists():
        raise HTTPException(404, "index.html not found")
    return HTMLResponse(html.read_text(encoding="utf-8"))


# ── 文件上传 ─────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(400, "文件超过 10 MB 限制")

    ext = Path(file.filename or "").suffix.lower()
    if ext in IMAGE_EXTS:
        b64 = base64.b64encode(data).decode()
        mime = {
            ".png":"image/png", ".jpg":"image/jpeg", ".jpeg":"image/jpeg",
            ".gif":"image/gif", ".webp":"image/webp",
        }.get(ext, "image/jpeg")
        return {
            "type":     "image",
            "filename": file.filename,
            "mime":     mime,
            "data":     b64,
            "size":     len(data),
        }
    else:
        text = _extract_text(file.filename or "", data)
        if not text.strip():
            raise HTTPException(400, f"无法提取 {file.filename} 的文本内容，请上传 txt/md/pdf/docx/代码文件或图片")
        return {
            "type":     "text",
            "filename": file.filename,
            "content":  text,
            "size":     len(data),
        }


# ── 角色管理 ─────────────────────────────────────────────────────────────────

@app.get("/api/agents")
async def list_agents():
    built_in_slugs = {v["slug"] for v in AGENTS_DICT.values()}
    result = [
        {"key": k, "slug": v["slug"], "name": v["name"],
         "role": v["role"], "color": v["color"], "focus": v["focus"]}
        for k, v in AGENTS_DICT.items()
    ]
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name in built_in_slugs:
            continue
        meta_file = skill_dir / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            result.append({
                "key":   skill_dir.name,
                "slug":  skill_dir.name,
                "name":  meta.get("name", skill_dir.name),
                "role":  meta.get("role", "自定义"),
                "color": "gray",
                "focus": meta.get("focus", ""),
            })
    return result


@app.post("/api/agents")
async def create_agent(req: AgentCreateRequest):
    safe = re.sub(r"[^\w]", "_", req.name)
    ts   = datetime.now(timezone.utc).strftime("%f")[:6]
    slug = f"custom_{safe}_{ts}"
    skill_dir = SKILLS_DIR / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "persona.md").write_text(
        f"# {req.name} — Persona\n\n{req.persona}\n", encoding="utf-8")
    if req.work:
        (skill_dir / "work.md").write_text(req.work, encoding="utf-8")
    meta = {"name": req.name, "slug": slug, "role": req.role,
            "focus": req.focus, "created_at": datetime.now(timezone.utc).isoformat(), "custom": True}
    (skill_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"key": slug, "slug": slug, "name": req.name, "role": req.role,
            "color": "gray", "focus": req.focus}


def _agent_skill_dir(key: str) -> Path:
    """找到 agent 对应的 skill 目录（内置或自定义）。"""
    if key in AGENTS_DICT:
        return SKILLS_DIR / AGENTS_DICT[key]["slug"]
    candidate = SKILLS_DIR / key
    if candidate.is_dir():
        return candidate
    raise HTTPException(404, f"未知成员: {key}")


@app.get("/api/agents/{key}")
async def get_agent_detail(key: str):
    skill_dir = _agent_skill_dir(key)
    meta_file = skill_dir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(404, "meta.json 不存在")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    persona_f = skill_dir / "persona.md"
    work_f    = skill_dir / "work.md"
    persona   = persona_f.read_text(encoding="utf-8") if persona_f.exists() else ""
    work      = work_f.read_text(encoding="utf-8")    if work_f.exists()    else ""

    current_rooms = [r for r, cfg in ROOMS.items() if key in cfg["members"]]

    agent_info = AGENTS_DICT.get(key)
    return {
        "key":          key,
        "slug":         meta.get("slug", key),
        "name":         agent_info["name"] if agent_info else meta.get("name", key),
        "role":         agent_info["role"] if agent_info else meta.get("role", ""),
        "color":        agent_info.get("color", "gray") if agent_info else "gray",
        "focus":        agent_info["focus"] if agent_info else meta.get("focus", ""),
        "persona":      persona,
        "work":         work,
        "current_rooms": current_rooms,
    }


@app.put("/api/agents/{key}")
async def update_agent(key: str, req: AgentUpdateRequest):
    skill_dir = _agent_skill_dir(key)

    # 写文件
    (skill_dir / "persona.md").write_text(req.persona, encoding="utf-8")
    (skill_dir / "work.md").write_text(req.work, encoding="utf-8")

    meta_file = skill_dir / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    meta.update({"name": req.name, "role": req.role, "focus": req.focus,
                 "updated_at": datetime.now(timezone.utc).isoformat()})
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 更新内存 AGENTS_DICT
    if key in AGENTS_DICT:
        AGENTS_DICT[key]["name"]  = req.name
        AGENTS_DICT[key]["role"]  = req.role
        AGENTS_DICT[key]["focus"] = req.focus

    # 更新内存 ROOMS members
    for room_cfg in ROOMS.values():
        if key in room_cfg["members"]:
            room_cfg["members"].remove(key)
    for room_id in req.rooms:
        if room_id in ROOMS and key not in ROOMS[room_id]["members"]:
            ROOMS[room_id]["members"].append(key)

    return {"key": key, "name": req.name, "role": req.role,
            "focus": req.focus, "rooms": req.rooms}


# ── 房间信息 ─────────────────────────────────────────────────────────────────

@app.get("/api/rooms")
async def list_rooms():
    result = []
    for room_id, room in ROOMS.items():
        result.append({
            "id":               room_id,
            "name":             room["name"],
            "members":          room["members"],
            "can_transfer_to":  room["can_transfer_to"],
            "has_context":      _context[room_id] is not None,
        })
    return result


@app.get("/api/room/{room_id}/history")
async def get_history(room_id: str):
    if room_id not in ROOMS:
        raise HTTPException(404, "未知房间")
    return {
        "messages": _sessions[room_id],
        "context":  _context[room_id],
    }


@app.delete("/api/room/{room_id}/history")
async def clear_history(room_id: str):
    if room_id not in ROOMS:
        raise HTTPException(404, "未知房间")
    _sessions[room_id] = []
    _context[room_id]  = None
    return {"ok": True}


# ── 发送消息（创建 Job → SSE 流） ────────────────────────────────────────────

@app.post("/api/room/{room_id}/chat")
async def start_chat(room_id: str, req: ChatRequest):
    if room_id not in ROOMS:
        raise HTTPException(404, "未知房间")
    if not req.message.strip():
        raise HTTPException(400, "消息不能为空")

    # 记录用户消息
    _sessions[room_id].append({
        "role": "user", "agent": None,
        "content": req.message,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "room_id": room_id, "message": req.message,
        "model": req.model, "at": req.at,
        "attachments": req.attachments,
    }
    return {"job_id": job_id}


@app.get("/api/room/{room_id}/stream/{job_id}")
async def stream_chat(room_id: str, job_id: str):
    job = _jobs.get(job_id)
    if not job or job["room_id"] != room_id:
        raise HTTPException(404, "job not found")

    return StreamingResponse(
        _chat_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _chat_generator(job_id: str) -> AsyncIterator[str]:
    job     = _jobs.get(job_id)
    if not job:
        yield _sse({"type": "error", "content": "job not found"})
        return

    room_id = job["room_id"]
    try:
        async for ev in room_chat(
            room_id    = room_id,
            user_message = job["message"],
            history    = _sessions[room_id],
            model      = job["model"],
            context_summary = _context[room_id],
            at_agent   = job.get("at"),
            attachments = job.get("attachments") or [],
        ):
            # 把 agent_done 的内容存入历史
            if ev["type"] == "agent_done":
                _sessions[room_id].append({
                    "role":    "assistant",
                    "agent":   ev["agent"],
                    "content": ev["content"],
                    "ts":      datetime.now(timezone.utc).isoformat(),
                })
            yield _sse(ev)
            await asyncio.sleep(0)

        yield _sse({"type": "stream_end"})
    except Exception as e:
        yield _sse({"type": "error", "content": str(e)})
    finally:
        _jobs.pop(job_id, None)


# ── 移交 ─────────────────────────────────────────────────────────────────────

@app.post("/api/room/{room_id}/transfer")
async def start_transfer(room_id: str, req: TransferRequest):
    if room_id not in ROOMS:
        raise HTTPException(404, "未知来源房间")
    if req.to_room_id not in ROOMS:
        raise HTTPException(404, "未知目标房间")
    if not _sessions[room_id]:
        raise HTTPException(400, "当前房间没有对话记录")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "type":        "transfer",
        "from_room":   room_id,
        "to_room":     req.to_room_id,
        "model":       req.model,
    }
    return {"job_id": job_id}


@app.get("/api/transfer/stream/{job_id}")
async def stream_transfer(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.get("type") != "transfer":
        raise HTTPException(404, "job not found")

    return StreamingResponse(
        _transfer_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _transfer_generator(job_id: str) -> AsyncIterator[str]:
    job = _jobs.get(job_id)
    if not job:
        yield _sse({"type": "error", "content": "job not found"})
        return

    from_room = job["from_room"]
    to_room   = job["to_room"]
    model     = job["model"]

    try:
        # Step 1：生成摘要
        yield _sse({"type": "transfer_progress", "content": "正在生成移交摘要…"})
        summary = await generate_transfer_summary(
            from_room, to_room, _sessions[from_room], model)

        # Step 2：写入目标房间的 context
        _context[to_room] = summary
        yield _sse({"type": "transfer_summary", "summary": summary, "to_room": to_room})

        # Step 3：目标房间成员开场发言
        yield _sse({"type": "transfer_greeting_start", "to_room": to_room})
        async for ev in transfer_greeting(to_room, summary, model):
            if ev["type"] == "agent_done":
                _sessions[to_room].append({
                    "role":    "assistant",
                    "agent":   ev["agent"],
                    "content": ev["content"],
                    "ts":      datetime.now(timezone.utc).isoformat(),
                })
            yield _sse(ev)
            await asyncio.sleep(0)

        yield _sse({"type": "transfer_done", "to_room": to_room})

    except Exception as e:
        yield _sse({"type": "error", "content": str(e)})
    finally:
        _jobs.pop(job_id, None)


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
