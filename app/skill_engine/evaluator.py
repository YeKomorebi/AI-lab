"""
evaluator.py — 模块三：Skill 评估 & 更新

职责：
  - Critic LLM 判断执行结果质量（pass/fail + score + reason）
  - 失败 → 修订 skill prompt（调用 library.update_skill_prompt）
  - 成功 → 更新 skill 置信度（library.record_execution）
  - 无 skill 且失败次数 >= 2 → 生成新 skill 写入 library
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import anthropic

from app.skill_engine import library

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# 跟踪每个子任务的连续失败次数（内存，重启清空）
_failure_counts: dict[str, int] = defaultdict(int)
NEW_SKILL_FAILURE_THRESHOLD = 2

_CRITIC_SYSTEM = """\
你是一个严格的质量评审员。评估 AI 执行子任务的输出是否达到要求。

输出严格 JSON，格式：
{
  "pass": true 或 false,
  "score": 0.0 到 1.0 的浮点数,
  "reason": "一句话说明判断依据",
  "improvement_suggestion": "若 pass=false，具体说明 prompt 应如何改进；pass=true 时为空字符串"
}

不要输出 JSON 以外的任何内容。
"""

_REVISE_SYSTEM = """\
你是一个 prompt 工程师。根据失败原因，修订 skill 的 prompt 模板，使其在下次执行时能产生更好的输出。

直接输出修订后的完整 prompt 文本，不加任何说明。
"""

_NEW_SKILL_SYSTEM = """\
你是一个 prompt 工程师。根据子任务描述和改进建议，生成一个新的 skill prompt 模板。

直接输出 prompt 文本，风格：简洁、专业、有明确的角色定位和输出格式要求。不加任何说明。
"""


async def evaluate_and_update(
    subtask: dict,
    execution_result: dict,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    评估执行结果，按情况更新 skill library。

    返回：{
        "pass": bool,
        "score": float,
        "reason": str,
        "action": "updated_confidence" | "revised_skill" | "created_skill" | "no_skill",
        "skill_slug": str | None,
    }
    """
    client = anthropic.AsyncAnthropic(base_url=BASE_URL)
    skill_slug = execution_result.get("skill_used")
    subtask_key = subtask.get("id", subtask.get("name", "unknown"))

    # ── Critic 评估 ───────────────────────────────────────────────────────────
    critic_user = f"""子任务：{subtask['name']}
描述：{subtask['description']}

执行输出：
{execution_result.get('output', '')}

执行状态：{execution_result.get('status', 'unknown')}
{f"错误信息：{execution_result['error']}" if execution_result.get('error') else ""}
"""

    resp = await client.messages.create(
        model=model,
        max_tokens=512,
        system=_CRITIC_SYSTEM,
        messages=[{"role": "user", "content": critic_user}],
    )
    raw = resp.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    critic = json.loads(raw)

    passed = bool(critic.get("pass", False))
    score = float(critic.get("score", 0.5))
    reason = critic.get("reason", "")
    suggestion = critic.get("improvement_suggestion", "")

    action = "no_skill"

    if skill_slug:
        # 更新置信度
        library.record_execution(skill_slug, success=passed)

        if not passed:
            # 失败 → 修订 skill prompt
            old_prompt = library.get_skill_prompt(skill_slug)
            revise_user = f"""原 skill prompt：
{old_prompt}

失败原因：{reason}
改进建议：{suggestion}

子任务描述：{subtask['description']}"""

            rev_resp = await client.messages.create(
                model=model,
                max_tokens=1024,
                system=_REVISE_SYSTEM,
                messages=[{"role": "user", "content": revise_user}],
            )
            new_prompt = rev_resp.content[0].text.strip()
            library.update_skill_prompt(skill_slug, new_prompt)
            library.update_skill_meta(skill_slug, status="active")
            action = "revised_skill"
        else:
            action = "updated_confidence"

    else:
        # 没有匹配的 skill
        if not passed:
            _failure_counts[subtask_key] += 1
        else:
            _failure_counts[subtask_key] = 0

        if _failure_counts[subtask_key] >= NEW_SKILL_FAILURE_THRESHOLD and suggestion:
            # 生成新 skill
            gen_user = f"""子任务描述：{subtask['description']}
改进建议：{suggestion}"""

            gen_resp = await client.messages.create(
                model=model,
                max_tokens=1024,
                system=_NEW_SKILL_SYSTEM,
                messages=[{"role": "user", "content": gen_user}],
            )
            new_prompt = gen_resp.content[0].text.strip()

            new_meta = library.create_skill(
                name=subtask["name"],
                description=subtask["description"],
                prompt_template=new_prompt,
                tags=["auto_generated"],
            )
            skill_slug = new_meta["slug"]
            _failure_counts[subtask_key] = 0
            action = "created_skill"

    return {
        "pass": passed,
        "score": score,
        "reason": reason,
        "action": action,
        "skill_slug": skill_slug,
    }
