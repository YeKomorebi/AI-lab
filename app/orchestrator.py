"""
orchestrator.py — 房间级 Agent 调度

职责：
  1. 按房间配置派发消息给对应 AI 角色（单人 or 多人并行）
  2. 生成移交摘要（transfer summary）
  3. 保留原有三阶段评审（技术评审室的「高级模式」）
"""

import asyncio
import os
from pathlib import Path
from typing import AsyncIterator
import anthropic

ROOT       = Path(__file__).parent.parent
SKILLS_DIR = ROOT / "skills" / "colleague"

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# ── 角色字典 ──────────────────────────────────────────────────────────────────

AGENTS = {
    "zhangsan": {
        "slug": "example_zhangsan", "name": "张三",
        "role": "后端技术", "color": "blue",
        "focus": "代码质量、接口设计、N+1 查询、事务边界、幂等性",
    },
    "tianyi": {
        "slug": "example_tianyi", "name": "天意",
        "role": "安全", "color": "cyan",
        "focus": "输入校验、注入风险、权限控制、安全评审",
    },
    "jiaxiu": {
        "slug": "example_jiaxiu", "name": "佳秀",
        "role": "业务/人员", "color": "orange",
        "focus": "业务影响、时间线、用户体验、跨团队协作",
    },
    "mingzhi": {
        "slug": "example_mingzhi", "name": "明志",
        "role": "算法", "color": "green",
        "focus": "实验严谨性、指标定义、data leakage、training-serving skew",
    },
}

# ── 房间配置 ──────────────────────────────────────────────────────────────────
# members: 该房间的驻场成员（key → AGENTS key）
# lead:    默认首先响应的角色（可为 None 表示全员并行）
# can_transfer_to: 可以移交到哪些房间

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
        "lead":    None,   # 全员并行
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

# ── Prompt 加载 ───────────────────────────────────────────────────────────────

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
    """把 session 消息历史拼成可读文本，供注入 system prompt。"""
    lines = []
    for m in messages[-20:]:   # 最近 20 条，避免 context 太长
        speaker = m.get("agent") or m.get("role", "用户")
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def _make_system(agent_key: str, room_id: str, context_summary: str | None) -> str:
    agent  = AGENTS[agent_key]
    room   = ROOMS[room_id]
    base   = _load_system_prompt(agent["slug"])
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
    # 把历史转成 messages 格式
    msgs = []
    for m in history[-12:]:
        role    = "user" if (m.get("role") == "user" or m.get("agent") is None) else "assistant"
        content = m["content"]
        msgs.append({"role": role, "content": content})

    # 构建最后一条用户消息（支持多模态）
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
    """
    根据房间配置派发消息，yield SSE 事件。

    事件类型：
      agent_start  — 某角色开始回复
      agent_done   — 某角色回复完毕，content 为完整文本
      error        — 出错
    """
    room   = ROOMS.get(room_id)
    if not room:
        yield {"type": "error", "content": f"未知房间: {room_id}"}
        return

    client = anthropic.AsyncAnthropic(base_url=BASE_URL)

    # 决定本次谁回复
    if at_agent and at_agent in room["members"]:
        responders = [at_agent]
    elif room["lead"] is None:
        responders = room["members"]   # 全员并行
    else:
        # 默认：lead 先回，其他人视消息内容决定要不要补充
        # 简化版：只让 lead 回复，除非消息里明确 @ 了别人
        responders = [room["lead"]]

    if room["lead"] is None:
        # 全员并行（技术评审室默认模式）
        yield {"type": "parallel_start", "agents": [AGENTS[k]["name"] for k in responders]}

        async def _one(key: str):
            agent  = AGENTS[key]
            system = _make_system(key, room_id, context_summary)
            yield {"type": "agent_start", "agent": agent["name"], "role": agent["role"]}
            text = await _call(client, model, system, history, user_message, attachments)
            yield {"type": "agent_done",  "agent": agent["name"], "role": agent["role"], "content": text}

        tasks = [_collect(_one(k)) for k in responders]
        results = await asyncio.gather(*tasks)
        for events in results:
            for ev in events:
                yield ev
    else:
        # 串行（lead 先，再看是否需要别人补充）
        for key in responders:
            agent  = AGENTS[key]
            system = _make_system(key, room_id, context_summary)
            yield {"type": "agent_start", "agent": agent["name"], "role": agent["role"]}
            text = await _call(client, model, system, history, user_message, attachments)
            yield {"type": "agent_done",  "agent": agent["name"], "role": agent["role"], "content": text}


async def _collect(gen: AsyncIterator[dict]) -> list[dict]:
    """把 async generator 收集成列表，用于并发 gather。"""
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
    """
    用中立视角总结本次对话，生成移交摘要。
    摘要会作为 context 注入目标房间的 system prompt。
    """
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
    """
    目标房间收到移交后，各成员读取摘要并主动发表初步意见。
    """
    room   = ROOMS[to_room_id]
    client = anthropic.AsyncAnthropic(base_url=BASE_URL)

    for key in room["members"]:
        agent  = AGENTS[key]
        system = _make_system(key, to_room_id, summary)
        user   = f"你刚刚收到了一份来自上一个房间的移交摘要（已在你的背景信息里）。请结合你的专业视角，简短说一下你对这个任务的初步判断和你准备重点关注什么。不超过 3 句话。"

        yield {"type": "agent_start", "agent": agent["name"], "role": agent["role"]}
        text = await _call(client, model, system, [], user)
        yield {"type": "agent_done",  "agent": agent["name"], "role": agent["role"], "content": text}


# ── 保留原有三阶段评审（技术评审室高级模式） ──────────────────────────────────

async def run_review(
    material: str,
    model: str = "claude-sonnet-4-6",
    selected_agents: list | None = None,
) -> AsyncIterator[dict]:
    """原有三阶段评审，保持不变。供技术评审室「发起三阶段评审」按钮使用。"""
    from app.orchestrator_review import run_review_phases
    async for ev in run_review_phases(material, model, selected_agents):
        yield ev
