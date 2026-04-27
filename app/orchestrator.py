"""
orchestrator.py — 房间级 Agent 调度

职责：
  1. 按房间配置派发消息给对应 AI 角色（单人 or 多人并行）
  2. 生成移交摘要（transfer summary）
  3. 保留原有三阶段评审（技术评审室的「高级模式」）
"""

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator
import anthropic

ROOT       = Path(__file__).parent.parent
SKILLS_DIR = ROOT / "skills" / "colleague"

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# ── 角色字典 ──────────────────────────────────────────────────────────────────
# 启动时从 meta.json 读取 name/role/focus，保证重启后与磁盘一致

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

# ── 房间配置 ──────────────────────────────────────────────────────────────────

ROOMS = {
    "product": {
        "name":    "需求讨论室",
        "members": ["jiaxiu", "zhangsan", "mingzhi"],
        "lead":    "jiaxiu",
        "can_transfer_to": ["review"],
        "system_hint": "这是需求讨论室，佳秀负责业务拆解和优先级，张三估工期，明志评技术可行性。",
    },
    "review": {
        "name":    "技术评审室",
        "members": ["zhangsan", "tianyi", "mingzhi"],
        "lead":    None,
        "can_transfer_to": ["algo"],
        "system_hint": "这是技术评审室，张三负责后端技术，天意负责安全，明志负责算法可行性。",
    },
    "algo": {
        "name":    "算法实验室",
        "members": ["mingzhi", "zhangsan", "tianyi"],
        "lead":    "mingzhi",
        "can_transfer_to": [],
        "system_hint": "这是算法实验室，明志主导实验设计，张三评工程实现，天意审数据安全。",
    },
}

# ── Agent 信息查找（支持内置 + 自定义） ──────────────────────────────────────

def _get_agent_info(key: str) -> dict:
    """返回 agent 的运行时信息，内置从 AGENTS 取，自定义从 meta.json 读。"""
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
    """返回 agent 对应的 skill 目录名（slug）。"""
    if key in AGENTS:
        return AGENTS[key]["slug"]
    return key   # 自定义成员 key == slug


def _get_api_key(key: str) -> str | None:
    """从 meta.json 读取 agent 专属 API Key，没有则返回 None（用全局环境变量）。"""
    slug = _get_slug(key)
    meta_file = SKILLS_DIR / slug / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        v = meta.get("api_key", "").strip()
        return v if v else None
    return None


def _make_client(key: str) -> anthropic.AsyncAnthropic:
    """为指定 agent 创建 Anthropic client，优先使用其专属 API Key。"""
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


def _make_system(agent_key: str, room_id: str, context_summary: str | None) -> str:
    agent  = _get_agent_info(agent_key)
    slug   = _get_slug(agent_key)
    room   = ROOMS[room_id]
    base   = _load_system_prompt(slug)
    hint   = f"\n\n[当前场景] {room['system_hint']}"
    ctx    = ""
    if context_summary:
        ctx = f"\n\n[来自上一个房间的移交摘要]\n{context_summary}"
    return base + hint + ctx


# ── 单次 LLM 调用 ─────────────────────────────────────────────────────────────

async def _call(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    history: list[dict],
    user_msg: str,
    attachments: list[dict] | None = None,
) -> str:
    msgs = []
    for m in history[-12:]:
        role    = "user" if (m.get("role") == "user" or m.get("agent") is None) else "assistant"
        content = m["content"]
        msgs.append({"role": role, "content": content})

    if attachments:
        content_blocks: list = []
        for att in attachments:
            if att["type"] == "image":
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": att["mime"],
                        "data":       att["data"],
                    },
                })
                content_blocks.append({
                    "type": "text",
                    "text": f"[上图文件名: {att['filename']}]",
                })
            elif att["type"] == "text":
                content_blocks.append({
                    "type": "text",
                    "text": f"[附件: {att['filename']}]\n```\n{att['content'][:8000]}\n```",
                })
        if user_msg:
            content_blocks.append({"type": "text", "text": user_msg})
        msgs.append({"role": "user", "content": content_blocks})
    else:
        msgs.append({"role": "user", "content": user_msg})

    resp = await client.messages.create(
        model=model, max_tokens=1024,
        system=system, messages=msgs,
    )
    return resp.content[0].text


# ── 房间聊天（SSE 事件流） ────────────────────────────────────────────────────

async def room_chat(
    room_id: str,
    user_message: str,
    history: list[dict],
    model: str = "claude-sonnet-4-6",
    context_summary: str | None = None,
    at_agent: str | None = None,
    attachments: list[dict] | None = None,
) -> AsyncIterator[dict]:
    room = ROOMS.get(room_id)
    if not room:
        yield {"type": "error", "content": f"未知房间: {room_id}"}
        return

    if at_agent and at_agent in room["members"]:
        responders = [at_agent]
    elif room["lead"] is None:
        responders = list(room["members"])
    else:
        responders = [room["lead"]]

    if room["lead"] is None:
        yield {"type": "parallel_start", "agents": [_get_agent_info(k)["name"] for k in responders]}

        async def _one(key: str):
            agent  = _get_agent_info(key)
            system = _make_system(key, room_id, context_summary)
            client = _make_client(key)
            yield {"type": "agent_start", "agent": agent["name"], "role": agent["role"]}
            text = await _call(client, model, system, history, user_message, attachments)
            yield {"type": "agent_done",  "agent": agent["name"], "role": agent["role"], "content": text}

        tasks = [_collect(_one(k)) for k in responders]
        results = await asyncio.gather(*tasks)
        for events in results:
            for ev in events:
                yield ev
    else:
        for key in responders:
            agent  = _get_agent_info(key)
            system = _make_system(key, room_id, context_summary)
            client = _make_client(key)
            yield {"type": "agent_start", "agent": agent["name"], "role": agent["role"]}
            text = await _call(client, model, system, history, user_message, attachments)
            yield {"type": "agent_done",  "agent": agent["name"], "role": agent["role"], "content": text}


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


# ── 保留原有三阶段评审（技术评审室高级模式） ──────────────────────────────────

async def run_review(
    material: str,
    model: str = "claude-sonnet-4-6",
    selected_agents: list | None = None,
) -> AsyncIterator[dict]:
    from app.orchestrator_review import run_review_phases
    async for ev in run_review_phases(material, model, selected_agents):
        yield ev
