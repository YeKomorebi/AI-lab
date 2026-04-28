"""
decomposer.py — 模块一：任务分解 & Skill 检索

职责：
  1. 用 LLM 把用户任务拆成有序子任务列表
  2. 对每个子任务检索 skill library 中最相关的 skill（Top-3）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic

from app.skill_engine.library import search_skills

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

_DECOMPOSE_SYSTEM = """\
你是一个任务分解专家。接收用户任务后，将其拆解为独立的、可并行或顺序执行的子任务列表。

输出严格 JSON，格式：
{
  "subtasks": [
    {
      "id": "1",
      "name": "子任务名称（10字以内）",
      "description": "具体要做什么（1-2句话）",
      "depends_on": []   // 依赖的子任务 id 列表，无依赖则为空
    }
  ]
}

要求：
- 子任务数量 2-6 个
- 每个子任务应独立可执行
- 不要添加任何 JSON 以外的内容
"""


async def decompose_task(
    task: str,
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """把用户任务拆成子任务列表，每个子任务附带匹配的 skill 候选。"""
    client = anthropic.AsyncAnthropic(base_url=BASE_URL)

    resp = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=_DECOMPOSE_SYSTEM,
        messages=[{"role": "user", "content": task}],
    )
    raw = resp.content[0].text.strip()

    # 容错：提取 JSON 块
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    subtasks = data.get("subtasks", [])

    # 为每个子任务检索匹配的 skill
    for st in subtasks:
        query = f"{st['name']} {st['description']}"
        candidates = search_skills(query, top_k=3)
        st["skill_candidates"] = [
            {
                "slug": c["slug"],
                "name": c["name"],
                "description": c["description"],
                "confidence": c.get("confidence", 0.5),
            }
            for c in candidates
        ]

    return subtasks
