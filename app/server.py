"""
server.py — AI Lab 工作台 API

端点：
  GET  /                              前端页面
  GET  /api/agents                    所有可用角色
  POST /api/agents                    创建自定义角色
  PUT  /api/agents/{key}              更新角色
  GET  /api/agents/{key}              角色详情
  GET  /api/rooms                     所有房间
  POST /api/rooms                     创建房间
  PUT  /api/rooms/{room_id}           更新房间
  DELETE /api/rooms/{room_id}         删除房间
  GET  /api/room/{room_id}/history    历史记录
  DELETE /api/room/{room_id}/history  清空历史
  POST /api/room/{room_id}/chat       发送消息，返回 job_id
  GET  /api/room/{room_id}/stream/{job_id}  SSE 流
  POST /api/room/{room_id}/transfer   创建移交 job
  GET  /api/transfer/stream/{job_id}  移交 SSE 流

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
    _get_agent_info,
    create_room,
    update_room,
    delete_room,
    load_room_memory,
    save_room_memory,
    extract_memory,
)
from app.skill_engine import library as skill_library
from app.skill_engine.decomposer import decompose_task
from app.skill_engine.executor import execute_subtask
from app.skill_engine.evaluator import evaluate_and_update
from app.skill_engine.extractor import extract_skills

app = FastAPI(title="AI Lab 工作台")

STATIC_DIR = Path(__file__).parent / "static"
SKILLS_DIR = Path(__file__).parent.parent / "skills" / "colleague"
CACHE_DIR  = Path(__file__).parent.parent / "data" / "sessions"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── 持久化辅助 ────────────────────────────────────────────────────────────────

def _cache_file(room_id: str) -> Path:
    return CACHE_DIR / f"{room_id}.json"

def _load_cache(room_id: str) -> dict:
    f = _cache_file(room_id)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"messages": [], "context": None}

def _save_cache(room_id: str) -> None:
    _cache_file(room_id).write_text(
        json.dumps({"messages": _sessions[room_id], "context": _context[room_id]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _ensure_room_session(room_id: str) -> None:
    if room_id not in _sessions:
        cache = _load_cache(room_id)
        _sessions[room_id] = cache["messages"]
        _context[room_id]  = cache["context"]
        _memory[room_id]   = load_room_memory(room_id)

# ── 内存存储（启动时从磁盘恢复所有已有 room 的 session） ──────────────────────
_sessions: dict[str, list[dict]] = {}
_context:  dict[str, str | None] = {}
_memory:   dict[str, list[str]]  = {}

for _rid in list(ROOMS.keys()):
    _ensure_room_session(_rid)

# SSE job 队列
_jobs: dict[str, dict] = {}


# ── 数据模型 ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    model:   str = "claude-sonnet-4-6"
    at:      str | None = None
    attachments: list[dict] = []

class TransferRequest(BaseModel):
    to_room_id: str
    model:      str = "claude-sonnet-4-6"

class AgentSkillPatchRequest(BaseModel):
    section:  str        # "persona" | "work"
    content:  str        # 要追加的文本
    mode:     str = "append"   # "append" | "replace"


    name:    str
    role:    str
    focus:   str = ""
    persona: str
    work:    str = ""
    api_key: str = ""
    avatar:  str = ""

class AgentUpdateRequest(BaseModel):
    name:    str
    role:    str
    focus:   str = ""
    persona: str
    work:    str = ""
    rooms:   list[str] = []
    api_key: str = ""
    avatar:  str = ""

class RoomCreateRequest(BaseModel):
    name:        str
    members:     list[str] = []
    description: str = ""

class RoomUpdateRequest(BaseModel):
    name:        str
    members:     list[str] = []
    description: str = ""


# ── 文件解析 ─────────────────────────────────────────────────────────────────

def _extract_text(filename: str, data: bytes) -> str:
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
    return ""


# ── 页面路由 ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = STATIC_DIR / "index.html"
    if not html.exists():
        raise HTTPException(404, "index.html not found")
    return HTMLResponse(html.read_text(encoding="utf-8"))


# ── 文件上传 ─────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024

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
        return {"type": "image", "filename": file.filename, "mime": mime, "data": b64, "size": len(data)}
    else:
        text = _extract_text(file.filename or "", data)
        if not text.strip():
            raise HTTPException(400, f"无法提取 {file.filename} 的文本内容")
        return {"type": "text", "filename": file.filename, "content": text, "size": len(data)}


# ── 角色管理 ─────────────────────────────────────────────────────────────────

@app.get("/api/agents")
async def list_agents():
    built_in_slugs = {v["slug"] for v in AGENTS_DICT.values()}
    result = []
    for k, v in AGENTS_DICT.items():
        slug = v["slug"]
        meta_file = SKILLS_DIR / slug / "meta.json"
        avatar = ""
        if meta_file.exists():
            try:
                avatar = json.loads(meta_file.read_text(encoding="utf-8")).get("avatar", "")
            except Exception:
                pass
        result.append({"key": k, "slug": slug, "name": v["name"],
                       "role": v["role"], "color": v["color"], "focus": v["focus"], "avatar": avatar})
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name in built_in_slugs:
            continue
        meta_file = skill_dir / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            result.append({
                "key":    skill_dir.name,
                "slug":   skill_dir.name,
                "name":   meta.get("name", skill_dir.name),
                "role":   meta.get("role", "自定义"),
                "color":  "gray",
                "focus":  meta.get("focus", ""),
                "avatar": meta.get("avatar", ""),
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
            "focus": req.focus, "api_key": req.api_key, "avatar": req.avatar,
            "created_at": datetime.now(timezone.utc).isoformat(), "custom": True}
    (skill_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"key": slug, "slug": slug, "name": req.name, "role": req.role,
            "color": "gray", "focus": req.focus}


def _agent_skill_dir(key: str) -> Path:
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
    meta  = json.loads(meta_file.read_text(encoding="utf-8"))
    persona_f = skill_dir / "persona.md"
    work_f    = skill_dir / "work.md"
    persona   = persona_f.read_text(encoding="utf-8") if persona_f.exists() else ""
    work      = work_f.read_text(encoding="utf-8")    if work_f.exists()    else ""
    current_rooms = [r for r, cfg in ROOMS.items() if key in cfg["members"]]
    agent_info = _get_agent_info(key)
    return {
        "key":           key,
        "slug":          agent_info["slug"],
        "name":          agent_info["name"],
        "role":          agent_info["role"],
        "color":         agent_info["color"],
        "focus":         agent_info["focus"],
        "persona":       persona,
        "work":          work,
        "current_rooms": current_rooms,
        "api_key":       meta.get("api_key", ""),
        "avatar":        meta.get("avatar", ""),
    }


@app.put("/api/agents/{key}")
async def update_agent(key: str, req: AgentUpdateRequest):
    skill_dir = _agent_skill_dir(key)
    meta_file = skill_dir / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    old_name = meta.get("name", req.name)
    if "original_name" not in meta:
        meta["original_name"] = old_name

    persona = req.persona
    if old_name and old_name != req.name:
        persona = persona.replace(old_name, req.name)

    (skill_dir / "persona.md").write_text(persona, encoding="utf-8")
    (skill_dir / "work.md").write_text(req.work, encoding="utf-8")

    meta.update({"name": req.name, "role": req.role, "focus": req.focus,
                 "api_key": req.api_key, "avatar": req.avatar,
                 "updated_at": datetime.now(timezone.utc).isoformat()})
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if key in AGENTS_DICT:
        AGENTS_DICT[key]["name"]  = req.name
        AGENTS_DICT[key]["role"]  = req.role
        AGENTS_DICT[key]["focus"] = req.focus

    # 更新所属房间
    for room_cfg in ROOMS.values():
        if key in room_cfg["members"]:
            room_cfg["members"].remove(key)
    for room_id in req.rooms:
        if room_id in ROOMS and key not in ROOMS[room_id]["members"]:
            ROOMS[room_id]["members"].append(key)

    from app.orchestrator import save_rooms
    save_rooms(ROOMS)

    return {"key": key, "name": req.name, "role": req.role,
            "focus": req.focus, "rooms": req.rooms}


@app.patch("/api/agents/{key}/skill")
async def patch_agent_skill(key: str, req: AgentSkillPatchRequest):
    skill_dir = _agent_skill_dir(key)
    if not skill_dir.exists():
        raise HTTPException(404, "agent 不存在")
    if req.section not in ("persona", "work"):
        raise HTTPException(400, "section 必须为 persona 或 work")

    target_file = skill_dir / f"{req.section}.md"
    if req.mode == "replace":
        target_file.write_text(req.content, encoding="utf-8")
    else:
        existing = target_file.read_text(encoding="utf-8") if target_file.exists() else ""
        separator = "\n\n---\n\n" if existing.strip() else ""
        target_file.write_text(existing + separator + req.content, encoding="utf-8")

    meta_file = skill_dir / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "key": key, "section": req.section, "mode": req.mode}


@app.delete("/api/agents/{key}")
async def delete_agent(key: str):
    skill_dir = _agent_skill_dir(key)
    if key in AGENTS_DICT:
        del AGENTS_DICT[key]
    # 从所有房间中移除
    for room_cfg in ROOMS.values():
        if key in room_cfg["members"]:
            room_cfg["members"].remove(key)
    from app.orchestrator import save_rooms
    save_rooms(ROOMS)
    # 删除文件
    import shutil
    shutil.rmtree(skill_dir)
    return {"ok": True}


# ── 房间 CRUD ─────────────────────────────────────────────────────────────────

@app.get("/api/rooms")
async def list_rooms():
    result = []
    for room_id, room in ROOMS.items():
        result.append({
            "id":              room_id,
            "name":            room["name"],
            "description":     room.get("description", ""),
            "members":         room["members"],
            "lead":            room.get("lead"),
            "can_transfer_to": room.get("can_transfer_to", []),
            "has_context":     _context.get(room_id) is not None,
            "created_at":      room.get("created_at", ""),
        })
    return result


@app.post("/api/rooms")
async def api_create_room(req: RoomCreateRequest):
    if not req.name.strip():
        raise HTTPException(400, "房间名不能为空")
    room = create_room(req.name, req.members, req.description)
    _ensure_room_session(room["id"])
    return room


@app.put("/api/rooms/{room_id}")
async def api_update_room(room_id: str, req: RoomUpdateRequest):
    if room_id not in ROOMS:
        raise HTTPException(404, "房间不存在")
    room = update_room(room_id, name=req.name, members=req.members, description=req.description)
    return room


@app.delete("/api/rooms/{room_id}")
async def api_delete_room(room_id: str):
    if room_id not in ROOMS:
        raise HTTPException(404, "房间不存在")
    delete_room(room_id)
    _sessions.pop(room_id, None)
    _context.pop(room_id, None)
    f = _cache_file(room_id)
    if f.exists():
        f.unlink()
    return {"ok": True}


# ── 历史记录 ─────────────────────────────────────────────────────────────────

@app.get("/api/room/{room_id}/history")
async def get_history(room_id: str):
    if room_id not in ROOMS:
        raise HTTPException(404, "未知房间")
    _ensure_room_session(room_id)
    return {"messages": _sessions[room_id], "context": _context[room_id]}


@app.delete("/api/room/{room_id}/history")
async def clear_history(room_id: str):
    if room_id not in ROOMS:
        raise HTTPException(404, "未知房间")
    _sessions[room_id] = []
    _context[room_id]  = None
    f = _cache_file(room_id)
    if f.exists():
        f.unlink()
    return {"ok": True}


# ── 发送消息（创建 Job → SSE 流） ────────────────────────────────────────────

@app.post("/api/room/{room_id}/chat")
async def start_chat(room_id: str, req: ChatRequest):
    if room_id not in ROOMS:
        raise HTTPException(404, "未知房间")
    if not req.message.strip() and not req.attachments:
        raise HTTPException(400, "消息不能为空")
    _ensure_room_session(room_id)

    _sessions[room_id].append({
        "role": "user", "agent": None,
        "content": req.message,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    _save_cache(room_id)

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
    job = _jobs.get(job_id)
    if not job:
        yield _sse({"type": "error", "content": "job not found"})
        return
    room_id = job["room_id"]
    try:
        async for ev in room_chat(
            room_id=room_id,
            user_message=job["message"],
            history=_sessions[room_id],
            model=job["model"],
            context_summary=_context[room_id],
            at_agent=job.get("at"),
            attachments=job.get("attachments") or [],
            room_memory=_memory.get(room_id, []),
        ):
            if ev["type"] == "agent_done":
                _sessions[room_id].append({
                    "role":    "assistant",
                    "agent":   ev["agent"],
                    "content": ev["content"],
                    "ts":      datetime.now(timezone.utc).isoformat(),
                })
                _save_cache(room_id)
                if len(_sessions[room_id]) % 10 == 0:
                    asyncio.create_task(_refresh_memory(room_id, job["model"]))
            yield _sse(ev)
            await asyncio.sleep(0)
        yield _sse({"type": "stream_end"})
    except Exception as e:
        yield _sse({"type": "error", "content": str(e)})
    finally:
        _jobs.pop(job_id, None)


async def _refresh_memory(room_id: str, model: str) -> None:
    items = await extract_memory(room_id, _sessions[room_id], model=model)
    _memory[room_id] = items
    save_room_memory(room_id, items)


# ── 移交 ─────────────────────────────────────────────────────────────────────

@app.post("/api/room/{room_id}/transfer")
async def start_transfer(room_id: str, req: TransferRequest):
    if room_id not in ROOMS:
        raise HTTPException(404, "未知来源房间")
    if req.to_room_id not in ROOMS:
        raise HTTPException(404, "未知目标房间")
    _ensure_room_session(room_id)
    if not _sessions[room_id]:
        raise HTTPException(400, "当前房间没有对话记录")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "type":      "transfer",
        "from_room": room_id,
        "to_room":   req.to_room_id,
        "model":     req.model,
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
    _ensure_room_session(to_room)
    try:
        yield _sse({"type": "transfer_progress", "content": "正在生成移交摘要…"})
        summary = await generate_transfer_summary(
            from_room, to_room, _sessions[from_room], model)
        _context[to_room] = summary
        _save_cache(to_room)
        yield _sse({"type": "transfer_summary", "summary": summary, "to_room": to_room})
        yield _sse({"type": "transfer_greeting_start", "to_room": to_room})
        async for ev in transfer_greeting(to_room, summary, model):
            if ev["type"] == "agent_done":
                _sessions[to_room].append({
                    "role":    "assistant",
                    "agent":   ev["agent"],
                    "content": ev["content"],
                    "ts":      datetime.now(timezone.utc).isoformat(),
                })
                _save_cache(to_room)
            yield _sse(ev)
            await asyncio.sleep(0)
        yield _sse({"type": "transfer_done", "to_room": to_room})
    except Exception as e:
        yield _sse({"type": "error", "content": str(e)})
    finally:
        _jobs.pop(job_id, None)


# ── Skill Engine 路由 ─────────────────────────────────────────────────────────

class SkillEngineRunRequest(BaseModel):
    task: str
    model: str = "claude-sonnet-4-6"


@app.post("/api/skill-engine/run")
async def skill_engine_run(req: SkillEngineRunRequest):
    if not req.task.strip():
        raise HTTPException(400, "任务不能为空")
    return StreamingResponse(
        _skill_engine_generator(req.task, req.model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _skill_engine_generator(task: str, model: str):
    try:
        yield _sse({"type": "phase", "phase": 1, "message": "正在分解任务并检索 Skill…"})
        subtasks = await decompose_task(task, model=model)
        yield _sse({"type": "decompose_done", "subtasks": subtasks})
        results = []
        for st in subtasks:
            yield _sse({"type": "phase", "phase": 2,
                        "message": f"执行子任务：{st['name']}", "subtask_id": st["id"]})
            exec_result = await execute_subtask(st, task_context=task, model=model)
            yield _sse({"type": "execute_done", "subtask_id": st["id"],
                        "status": exec_result["status"],
                        "skill_used": exec_result.get("skill_used"),
                        "output": exec_result["output"]})
            yield _sse({"type": "phase", "phase": 3,
                        "message": f"评估子任务：{st['name']}", "subtask_id": st["id"]})
            eval_result = await evaluate_and_update(st, exec_result, model=model)
            yield _sse({"type": "evaluate_done", "subtask_id": st["id"], **eval_result})
            results.append({"subtask": st, "execution": exec_result, "evaluation": eval_result})
            await asyncio.sleep(0)
        yield _sse({"type": "phase", "phase": 4, "message": "整理 Skill Library…"})
        merged = skill_library.merge_similar_skills()
        deprecated = skill_library.deprecate_low_confidence()
        yield _sse({"type": "library_maintenance", "merged": merged, "deprecated": deprecated})
        yield _sse({"type": "run_end", "total_subtasks": len(subtasks),
                    "passed": sum(1 for r in results if r["evaluation"]["pass"])})
    except Exception as e:
        yield _sse({"type": "error", "content": str(e)})


@app.get("/api/skill-engine/skills")
async def list_task_skills(status: str = "active"):
    return skill_library.list_skills(status=status if status != "all" else None)


@app.get("/api/skill-engine/skills/{slug}")
async def get_task_skill(slug: str):
    meta = skill_library.get_skill(slug)
    if meta is None:
        raise HTTPException(404, f"Skill 不存在: {slug}")
    meta["prompt"] = skill_library.get_skill_prompt(slug)
    return meta


@app.delete("/api/skill-engine/skills/{slug}")
async def deprecate_task_skill(slug: str):
    meta = skill_library.get_skill(slug)
    if meta is None:
        raise HTTPException(404, f"Skill 不存在: {slug}")
    skill_library.update_skill_meta(slug, status="deprecated")
    return {"ok": True, "slug": slug}


@app.post("/api/skill-engine/maintenance")
async def run_maintenance():
    merged = skill_library.merge_similar_skills()
    deprecated = skill_library.deprecate_low_confidence()
    return {"merged": merged, "deprecated": deprecated}


class SkillExtractRequest(BaseModel):
    source_type: str                    # "text" | "url" | "feishu"
    model: str = "claude-sonnet-4-6"
    text: str | None = None
    url: str | None = None
    feishu_app_id: str | None = None
    feishu_app_secret: str | None = None
    feishu_doc_token: str | None = None
    feishu_wiki_token: str | None = None


@app.post("/api/skill-engine/extract")
async def skill_extract(req: SkillExtractRequest):
    return StreamingResponse(
        _skill_extract_generator(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _skill_extract_generator(req: SkillExtractRequest):
    async for ev in extract_skills(
        source_type=req.source_type,
        model=req.model,
        text=req.text,
        url=req.url,
        feishu_app_id=req.feishu_app_id,
        feishu_app_secret=req.feishu_app_secret,
        feishu_doc_token=req.feishu_doc_token,
        feishu_wiki_token=req.feishu_wiki_token,
    ):
        yield _sse(ev)
        await asyncio.sleep(0)


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
