"""
orchestrator.py — 房间级 Agent 调度

职责：
  1. 动态房间管理（从磁盘加载 / 保存，支持 CRUD）
  2. 按房间配置派发消息给对应 AI 角色（单人 or 多人并行）
  3. 生成移交摘要（transfer summary）
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
import anthropic

SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

ROOT       = Path(__file__).parent.parent
SKILLS_DIR = ROOT / "skills" / "colleague"
ROOMS_FILE = ROOT / "data" / "rooms.json"
ROOMS_FILE.parent.mkdir(parents=True, exist_ok=True)

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# ── 角色字典 ──────────────────────────────────────────────────────────────────

def _init_agents() -> dict:
    builtin = {
        "zhangsan": {"slug": "example_zhangsan", "color": "blue"},
        "tianyi":   {"slug": "example_tianyi",   "color": "cyan"},
        "jiaxiu":   {"slug": "example_jiaxiu",   "color": "orange"},
        "mingzhi":  {"slug": "example_mingzhi",  "color": "green"},
    }
    result = {}
    for key, base in builtin.items():
        meta_file = SKILLS_DIR / base["slug"] / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        else:
            meta = {}
        result[key] = {
            "slug":  base["slug"],
            "name":  meta.get("name", key),
            "role":  meta.get("role", ""),
            "color": base["color"],
            "focus": meta.get("focus", ""),
        }
    return result

AGENTS: dict = _init_agents()

# ── 动态房间系统 ──────────────────────────────────────────────────────────────

def _default_rooms() -> dict:
    return {}

def load_rooms() -> dict:
    if ROOMS_FILE.exists():
        try:
            return json.loads(ROOMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _default_rooms()

def save_rooms(rooms: dict) -> None:
    ROOMS_FILE.write_text(
        json.dumps(rooms, ensure_ascii=False, indent=2), encoding="utf-8"
    )

# 运行时房间字典（热加载）
ROOMS: dict = load_rooms()


# ── Room Memory ───────────────────────────────────────────────────────────────

def _memory_file(room_id: str) -> Path:
    return SESSIONS_DIR / f"{room_id}_memory.json"


def load_room_memory(room_id: str) -> list[str]:
    f = _memory_file(room_id)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("items", [])
        except Exception:
            pass
    return []


def save_room_memory(room_id: str, items: list[str]) -> None:
    _memory_file(room_id).write_text(
        json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(), "items": items},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def extract_memory(
    room_id: str,
    history: list[dict],
    model: str = "claude-sonnet-4-6",
) -> list[str]:
    history_text = _build_history_text_full(history)
    system = "你是房间的隐形记录员。你的唯一任务是从对话中提炼出结构化记忆条目。"
    user = f"""以下是房间「{ROOMS.get(room_id, {}).get('name', room_id)}」的对话记录：

{history_text}

---
请提取 5-10 条记忆要点，严格按以下格式输出（每行一条，不加序号）：
已确认：<内容>
待解决：<内容>
关键决策：<内容>
背景信息：<内容>

只输出条目，不加任何说明文字。"""
    client = anthropic.AsyncAnthropic(base_url=BASE_URL)
    resp = await client.messages.create(
        model=model, max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    lines = [l.strip() for l in resp.content[0].text.strip().splitlines() if l.strip()]
    return lines


def create_room(name: str, members: list[str], description: str = "") -> dict:
    room_id = str(uuid.uuid4())[:8]
    # 确保不重复
    while room_id in ROOMS:
        room_id = str(uuid.uuid4())[:8]
    room = {
        "id":              room_id,
        "name":            name,
        "description":     description,
        "members":         members,
        "lead":            members[0] if members else None,
        "can_transfer_to": [],
        "system_hint":     f"这是「{name}」，成员各自发挥专长协作完成任务。",
        "created_at":      datetime.now(timezone.utc).isoformat(),
    }
    ROOMS[room_id] = room
    save_rooms(ROOMS)
    return room


def update_room(room_id: str, **fields) -> dict | None:
    room = ROOMS.get(room_id)
    if room is None:
        return None
    room.update(fields)
    if "members" in fields:
        room["lead"] = fields["members"][0] if fields["members"] else None
    save_rooms(ROOMS)
    return room


def delete_room(room_id: str) -> bool:
    if room_id not in ROOMS:
        return False
    del ROOMS[room_id]
    save_rooms(ROOMS)
    return True


# ── Agent 信息查找 ────────────────────────────────────────────────────────────

def _get_agent_info(key: str) -> dict:
    if key in AGENTS:
        return AGENTS[key]
    meta_file = SKILLS_DIR / key / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return {
            "slug":  key,
            "name":  meta.get("name", key),
            "role":  meta.get("role", "自定义"),
            "color": "gray",
            "focus": meta.get("focus", ""),
        }
    return {"slug": key, "name": key, "role": "", "color": "gray", "focus": ""}


def _get_slug(key: str) -> str:
    if key in AGENTS:
        return AGENTS[key]["slug"]
    return key


def _get_api_key(key: str) -> str | None:
    slug = _get_slug(key)
    meta_file = SKILLS_DIR / slug / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        v = meta.get("api_key", "").strip()
        return v if v else None
    return None


def _make_client(key: str) -> anthropic.AsyncAnthropic:
    api_key = _get_api_key(key)
    if api_key:
        return anthropic.AsyncAnthropic(api_key=api_key, base_url=BASE_URL)
    return anthropic.AsyncAnthropic(base_url=BASE_URL)


def _load_system_prompt(slug: str) -> str:
    skill_dir = SKILLS_DIR / slug
    parts = []
    for fname in ("persona.md", "work.md"):
        f = skill_dir / fname
        if f.exists():
            parts.append(f.read_text(encoding="utf-8"))
    if not parts:
        raise FileNotFoundError(f"找不到 skill 文件: {skill_dir}")
    return "\n\n---\n\n".join(parts)


def _build_history_text(messages: list[dict]) -> str:
    lines = []
    for m in messages[-20:]:
        speaker = m.get("agent") or m.get("role", "用户")
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def _build_history_text_full(messages: list[dict]) -> str:
    lines = []
    for m in messages[-40:]:
        speaker = m.get("agent") or m.get("role", "用户")
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def _make_system(agent_key: str, room_id: str, context_summary: str | None, room_memory: list[str] | None = None) -> str:
    agent  = _get_agent_info(agent_key)
    slug   = _get_slug(agent_key)
    room   = ROOMS[room_id]
    base   = _load_system_prompt(slug)

    meta_file = SKILLS_DIR / slug / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        original_name = meta.get("original_name", "")
        current_name  = agent["name"]
        if original_name and original_name != current_name:
            base = base.replace(original_name, current_name)

    hint  = f"\n\n[当前场景] {room['system_hint']}"
    hint += f"\n你现在的名字是【{agent['name']}】，请始终以此名字自称。"
    hint += "\n当你希望用户从几个选项中选择时，在回复末尾用以下格式输出选项（每组只用一次，选项之间用 | 分隔）：\n单选：[选项: 选项A | 选项B | 选项C]\n多选（用户可选多个）：[多选: 选项A | 选项B | 选项C]"
    ctx   = ""
    if context_summary:
        ctx = f"\n\n[来自上一个房间的移交摘要]\n{context_summary}"
    mem = ""
    if room_memory:
        mem = "\n\n[房间记忆]\n" + "\n".join(f"- {item}" for item in room_memory)
    return base + hint + ctx + mem


# ── 构建消息列表（共用） ──────────────────────────────────────────────────────

def _build_msgs(history: list[dict], user_msg: str, attachments: list[dict] | None) -> list[dict]:
    msgs = []
    for m in history[-12:]:
        role    = "user" if (m.get("role") == "user" or m.get("agent") is None) else "assistant"
        msgs.append({"role": role, "content": m["content"]})

    if attachments:
        blocks: list = []
        for att in attachments:
            if att["type"] == "image":
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": att["mime"], "data": att["data"],
                }})
                blocks.append({"type": "text", "text": f"[上图文件名: {att['filename']}]"})
            elif att["type"] == "text":
                blocks.append({"type": "text",
                    "text": f"[附件: {att['filename']}]\n```\n{att['content'][:8000]}\n```"})
        if user_msg:
            blocks.append({"type": "text", "text": user_msg})
        msgs.append({"role": "user", "content": blocks})
    else:
        msgs.append({"role": "user", "content": user_msg})
    return msgs


# ── 单次 LLM 调用（非流式，用于并行 / 移交） ─────────────────────────────────

async def _call(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    history: list[dict],
    user_msg: str,
    attachments: list[dict] | None = None,
) -> str:
    msgs = _build_msgs(history, user_msg, attachments)
    resp = await client.messages.create(
        model=model, max_tokens=2048,
        system=system, messages=msgs,
    )
    return resp.content[0].text


# ── 单次 LLM 调用（流式，逐 token yield） ────────────────────────────────────

async def _call_stream(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    history: list[dict],
    user_msg: str,
    attachments: list[dict] | None = None,
) -> AsyncIterator[str]:
    msgs = _build_msgs(history, user_msg, attachments)
    async with client.messages.stream(
        model=model, max_tokens=2048,
        system=system, messages=msgs,
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ── 房间聊天（SSE 事件流） ────────────────────────────────────────────────────

async def room_chat(
    room_id: str,
    user_message: str,
    history: list[dict],
    model: str = "claude-sonnet-4-6",
    context_summary: str | None = None,
    at_agent: str | None = None,
    attachments: list[dict] | None = None,
    room_memory: list[str] | None = None,
) -> AsyncIterator[dict]:
    room = ROOMS.get(room_id)
    if not room:
        yield {"type": "error", "content": f"未知房间: {room_id}"}
        return

    if at_agent and at_agent in room["members"]:
        responders = [at_agent]
    else:
        responders = list(room["members"])

    # 过滤掉 skill 文件已不存在的成员，避免单个幽灵 agent 崩掉整个请求
    valid_responders = []
    for k in responders:
        slug = _get_slug(k)
        skill_dir = SKILLS_DIR / slug
        if any((skill_dir / f).exists() for f in ("persona.md", "work.md")):
            valid_responders.append(k)
        else:
            yield {"type": "error", "content": f"成员「{k}」的 skill 文件不存在，已跳过"}
    responders = valid_responders

    if not responders:
        yield {"type": "error", "content": "房间中没有可用的成员（skill 文件均缺失）"}
        return

    # 多人并行：非流式批量（避免交错）
    if len(responders) > 1:
        yield {"type": "parallel_start", "agents": [_get_agent_info(k)["name"] for k in responders]}

        async def _one(key: str):
            agent  = _get_agent_info(key)
            system = _make_system(key, room_id, context_summary, room_memory)
            client = _make_client(key)
            yield {"type": "agent_start", "agent": agent["name"], "role": agent["role"]}
            text = await _call(client, model, system, history, user_message, attachments)
            yield {"type": "agent_done",  "agent": agent["name"], "role": agent["role"], "content": text}

        tasks = [_collect(_one(k)) for k in responders]
        results = await asyncio.gather(*tasks)
        for events in results:
            for ev in events:
                yield ev

    # 单人回复：流式 token
    else:
        key    = responders[0]
        agent  = _get_agent_info(key)
        system = _make_system(key, room_id, context_summary, room_memory)
        client = _make_client(key)
        yield {"type": "agent_start", "agent": agent["name"], "role": agent["role"]}
        full_text = ""
        async for token in _call_stream(client, model, system, history, user_message, attachments):
            full_text += token
            yield {"type": "token", "agent": agent["name"], "delta": token}
        yield {"type": "agent_done", "agent": agent["name"], "role": agent["role"], "content": full_text}


async def _collect(gen: AsyncIterator[dict]) -> list[dict]:
    items = []
    async for item in gen:
        items.append(item)
    return items


# ── 移交摘要生成 ──────────────────────────────────────────────────────────────

async def generate_transfer_summary(
    from_room_id: str,
    to_room_id: str,
    history: list[dict],
    model: str = "claude-sonnet-4-6",
) -> str:
    from_room = ROOMS[from_room_id]
    to_room   = ROOMS[to_room_id]
    history_text = _build_history_text(history)

    system = "你是一个中立的项目协作助手，负责整理会议记录，输出结构化移交摘要。语言简洁，每项不超过两句话。"
    user   = f"""以下是「{from_room['name']}」的完整对话记录：

{history_text}

---
请生成一份移交给「{to_room['name']}」的摘要，严格按照以下格式输出：

📋 移交摘要：{from_room['name']} → {to_room['name']}

**背景**：（一句话说清楚这个任务是什么）

**已达成共识**：
- （列出 2-4 条已经确认的结论）

**待解决问题**：
- （列出 1-3 条还没有答案的问题）

**请重点关注**：（告诉目标房间的成员需要重点处理什么）
"""
    client = anthropic.AsyncAnthropic(base_url=BASE_URL)
    resp   = await client.messages.create(
        model=model, max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


# ── 目标房间接收移交后的开场白 ────────────────────────────────────────────────

async def transfer_greeting(
    to_room_id: str,
    summary: str,
    model: str = "claude-sonnet-4-6",
) -> AsyncIterator[dict]:
    room = ROOMS[to_room_id]
    for key in room["members"]:
        agent  = _get_agent_info(key)
        system = _make_system(key, to_room_id, summary)
        client = _make_client(key)
        user   = "你刚刚收到了一份来自上一个房间的移交摘要（已在你的背景信息里）。请结合你的专业视角，简短说一下你对这个任务的初步判断和你准备重点关注什么。不超过 3 句话。"
        yield {"type": "agent_start", "agent": agent["name"], "role": agent["role"]}
        text = await _call(client, model, system, [], user)
        yield {"type": "agent_done",  "agent": agent["name"], "role": agent["role"], "content": text}
