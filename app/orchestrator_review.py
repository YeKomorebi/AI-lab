"""
orchestrator_review.py — 原有三阶段技术评审逻辑（从 orchestrator.py 剥离）
"""

import asyncio
import os
from pathlib import Path
from typing import AsyncIterator
import anthropic

ROOT       = Path(__file__).parent.parent
SKILLS_DIR = ROOT / "skills" / "colleague"
BASE_URL   = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

AGENTS = [
    {"slug":"example_zhangsan","name":"张三","role":"后端技术视角","focus":"代码质量、接口设计、N+1 查询、事务边界、幂等性、命名规范"},
    {"slug":"example_tianyi",  "name":"天意","role":"安全视角",    "focus":"输入校验、注入风险、权限控制、安全评审是否通过"},
    {"slug":"example_jiaxiu",  "name":"佳秀","role":"业务/人员视角","focus":"业务影响、上线时间线、候选人/用户体验、跨团队协作"},
    {"slug":"example_mingzhi", "name":"明志","role":"算法视角",    "focus":"实验严谨性、指标定义、data leakage、training-serving skew、监控方案"},
]

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

async def _call_agent(client, model, system, user) -> str:
    resp = await client.messages.create(
        model=model, max_tokens=1024,
        system=system, messages=[{"role":"user","content":user}],
    )
    return resp.content[0].text

def _phase1_prompt(agent, material):
    return f"""你现在参加一场技术方案评审会，以下是待评审内容：
---
{material}
---
请从你的专业视角（{agent['role']}）独立评审这份材料。
重点关注：{agent['focus']}
评审要求：结论先行，用你自己的说话风格和性格发言。"""

def _phase2_prompt(agent, material, phase1_results, agents):
    others = "\n\n".join(
        f"### {a['name']}（{a['role']}）\n{phase1_results[a['name']]}"
        for a in agents if a["name"] != agent["name"]
    )
    return f"""评审材料：\n---\n{material}\n---\n\n其他评审人意见：\n{others}\n\n请回应上述意见，保持你自己的立场和性格。"""

def _phase3_prompt(material, phase1, phase2):
    def fmt(d, label):
        return "\n\n".join(f"### {n}（{label}）\n{v}" for n,v in d.items())
    return f"""评审材料：\n---\n{material}\n---\n\n## Phase 1\n{fmt(phase1,'独立评审')}\n\n## Phase 2\n{fmt(phase2,'交叉回应')}\n\n请整合输出最终评审报告，包含：✅共识点 / ⚠️风险点 / ❌分歧点 / 📌Next Steps"""

async def run_review_phases(
    material: str,
    model: str = "claude-sonnet-4-6",
    selected_agents: list | None = None,
) -> AsyncIterator[dict]:
    agents = selected_agents if selected_agents else AGENTS
    client = anthropic.AsyncAnthropic(base_url=BASE_URL)
    system_prompts = {a["slug"]: _load_system_prompt(a["slug"]) for a in agents}

    yield {"phase":"1","type":"phase_start","content":f"Phase 1：独立评审开始，{len(agents)} 人并行"}
    for a in agents:
        yield {"phase":"1","type":"agent_start","agent":a["name"],"role":a["role"],"content":""}

    names     = [a["name"] for a in agents]
    responses = await asyncio.gather(*[
        _call_agent(client, model, system_prompts[a["slug"]], _phase1_prompt(a, material))
        for a in agents
    ])
    phase1 = dict(zip(names, responses))
    for a in agents:
        yield {"phase":"1","type":"agent_done","agent":a["name"],"role":a["role"],"content":phase1[a["name"]]}
    yield {"phase":"1","type":"phase_done","content":"Phase 1 完成"}

    yield {"phase":"2","type":"phase_start","content":"Phase 2：交叉碰撞开始"}
    phase2: dict[str,str] = {}
    for a in agents:
        yield {"phase":"2","type":"agent_start","agent":a["name"],"role":a["role"],"content":""}
        opinion = await _call_agent(client, model, system_prompts[a["slug"]], _phase2_prompt(a, material, phase1, agents))
        phase2[a["name"]] = opinion
        yield {"phase":"2","type":"agent_done","agent":a["name"],"role":a["role"],"content":opinion}
    yield {"phase":"2","type":"phase_done","content":"Phase 2 完成"}

    yield {"phase":"3","type":"phase_start","content":"Phase 3：生成评审报告"}
    yield {"phase":"3","type":"agent_start","agent":"synthesizer","content":""}
    report = await _call_agent(
        client, model,
        "你是中立的技术评审会主持人，整合多方意见输出结构化报告。",
        _phase3_prompt(material, phase1, phase2),
    )
    yield {"phase":"3","type":"review_done","agent":"synthesizer","content":report}
