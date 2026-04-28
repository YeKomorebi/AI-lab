"""
executor.py — 模块二：Skill 执行 & 反馈收集

职责：
  - 用 skill 的 prompt_template 构造消息调用 LLM
  - 若子任务没有匹配 skill，用通用 prompt 直接执行
  - 返回 {status: "success"|"failure"|"partial", output, error}
"""

from __future__ import annotations

import os

import anthropic

from app.skill_engine.library import get_skill, get_skill_prompt

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

_FALLBACK_SYSTEM = """\
你是一个通用任务执行助手。请尽力完成用户描述的子任务，输出具体的结果内容。
"""


async def execute_subtask(
    subtask: dict,
    task_context: str,
    model: str = "claude-sonnet-4-6",
    skill_slug: str | None = None,
) -> dict:
    """
    执行单个子任务。

    skill_slug: 指定使用哪个 skill；None 时从 skill_candidates 取第一个，
                仍无 skill 则用通用 fallback。
    返回: {"status": "success"|"failure", "output": str, "error": str, "skill_used": str|None}
    """
    client = anthropic.AsyncAnthropic(base_url=BASE_URL)

    # 确定要用的 skill
    slug = skill_slug
    if not slug:
        candidates = subtask.get("skill_candidates", [])
        if candidates:
            slug = candidates[0]["slug"]

    skill_meta = get_skill(slug) if slug else None
    skill_prompt = get_skill_prompt(slug) if slug else ""

    if skill_prompt:
        system = skill_prompt
    else:
        system = _FALLBACK_SYSTEM

    user_msg = f"""[总任务背景]
{task_context}

[当前子任务]
名称：{subtask['name']}
描述：{subtask['description']}

请完成这个子任务，输出具体结果。"""

    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        output = resp.content[0].text
        return {
            "status": "success",
            "output": output,
            "error": "",
            "skill_used": slug,
        }
    except Exception as e:
        return {
            "status": "failure",
            "output": "",
            "error": str(e),
            "skill_used": slug,
        }
