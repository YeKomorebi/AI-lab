"""
extractor.py — Skill 提取器

职责：从外部数据源（飞书文档、URL、纯文本）提取内容，
      经 LLM 提炼后写入 Skill Library。
"""

from __future__ import annotations

import json
import os
import re
from typing import AsyncIterator

import anthropic
import httpx

from app.skill_engine.library import create_skill, search_skills, _jaccard

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# 飞书 Open API 基地址
FEISHU_BASE = "https://open.feishu.cn/open-apis"

DEDUP_THRESHOLD = 0.80   # Jaccard 超过此值视为已有相似 skill，跳过


# ── 飞书 Token ────────────────────────────────────────────────────────────────

async def _feishu_tenant_token(app_id: str, app_secret: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": app_id, "app_secret": app_secret},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"飞书 token 获取失败: {data.get('msg')}")
        return data["tenant_access_token"]


# ── 飞书内容获取 ──────────────────────────────────────────────────────────────

async def _fetch_feishu_doc(doc_token: str, access_token: str) -> str:
    """获取飞书 Doc 的纯文本内容（v3 API）。"""
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{FEISHU_BASE}/docx/v1/documents/{doc_token}/raw_content",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"飞书文档读取失败: {data.get('msg')}")
        return data.get("data", {}).get("content", "")


async def _fetch_feishu_wiki(wiki_token: str, access_token: str) -> str:
    """获取飞书 Wiki 节点的文档 token，再拉内容。"""
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{FEISHU_BASE}/wiki/v2/spaces/get_node?token={wiki_token}",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"飞书 Wiki 读取失败: {data.get('msg')}")
        obj_token = data["data"]["node"]["obj_token"]
        return await _fetch_feishu_doc(obj_token, access_token)


# ── URL 抓取 ──────────────────────────────────────────────────────────────────

async def _fetch_url(url: str) -> str:
    """抓取 URL 页面并提取纯文本（简单去 HTML 标签）。"""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, timeout=20,
                                headers={"User-Agent": "Mozilla/5.0 SkillExtractor/1.0"})
        resp.raise_for_status()
        html = resp.text
    # 去掉 script/style 块，再剥 HTML 标签
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:20000]   # 截断，避免超长


# ── LLM 提炼 Skill ────────────────────────────────────────────────────────────

_DISTILL_SYSTEM = """\
你是一个 Skill Library 管理员。你的任务是：从用户提供的原始文档/内容中，提炼出可复用的 AI 执行 Skill。

每个 Skill 代表「一类可标准化的工作流程或专业能力」，需要包括：
- name：简短中文名（10字以内）
- description：一句话描述这个 skill 的用途（50字以内）
- tags：3-5个关键词标签（逗号分隔）
- prompt_template：给 AI 的执行指令模板（150-400字），描述如何用这个 skill 完成任务，包含占位符 {{task}} 表示具体任务

严格按照 JSON 数组格式输出，不加任何其他文字：
[
  {
    "name": "...",
    "description": "...",
    "tags": ["...", "..."],
    "prompt_template": "..."
  }
]

若文档内容不含可提炼的 skill，输出空数组 []。每次最多提炼 5 个 skill。\
"""


async def _distill_skills(raw_text: str, source_hint: str, model: str) -> list[dict]:
    client = anthropic.AsyncAnthropic(base_url=BASE_URL)
    user = f"来源：{source_hint}\n\n---\n\n{raw_text[:12000]}"
    resp = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=_DISTILL_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text.strip()
    # 提取 JSON 部分（LLM 有时加 markdown fence）
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group())
    except Exception:
        return []


# ── 去重检查 ──────────────────────────────────────────────────────────────────

def _is_duplicate(name: str, description: str) -> bool:
    candidates = search_skills(f"{name} {description}", top_k=5)
    query = f"{name} {description}"
    for c in candidates:
        c_text = f"{c.get('name', '')} {c.get('description', '')}"
        if _jaccard(query, c_text) >= DEDUP_THRESHOLD:
            return True
    return False


# ── 主入口：流式提取 ──────────────────────────────────────────────────────────

async def extract_skills(
    source_type: str,
    model: str = "claude-sonnet-4-6",
    # text
    text: str | None = None,
    # url
    url: str | None = None,
    # feishu
    feishu_app_id: str | None = None,
    feishu_app_secret: str | None = None,
    feishu_doc_token: str | None = None,
    feishu_wiki_token: str | None = None,
) -> AsyncIterator[dict]:
    """
    流式生成提取事件：
      {"type": "fetch_start"}
      {"type": "fetch_done", "chars": N}
      {"type": "distill_start"}
      {"type": "skill_saved", "slug": ..., "name": ..., "description": ...}
      {"type": "skill_skipped", "name": ..., "reason": "duplicate"}
      {"type": "extract_done", "saved": N, "skipped": N}
      {"type": "error", "content": ...}
    """
    try:
        yield {"type": "fetch_start", "source_type": source_type}

        raw = ""
        source_hint = source_type

        if source_type == "text":
            raw = (text or "").strip()
            source_hint = "手动文本"

        elif source_type == "url":
            if not url:
                yield {"type": "error", "content": "url 不能为空"}
                return
            raw = await _fetch_url(url)
            source_hint = url

        elif source_type == "feishu":
            if not feishu_app_id or not feishu_app_secret:
                yield {"type": "error", "content": "飞书 App ID 和 App Secret 不能为空"}
                return
            token = await _feishu_tenant_token(feishu_app_id, feishu_app_secret)
            if feishu_wiki_token:
                raw = await _fetch_feishu_wiki(feishu_wiki_token, token)
                source_hint = f"飞书 Wiki({feishu_wiki_token[:8]}…)"
            elif feishu_doc_token:
                raw = await _fetch_feishu_doc(feishu_doc_token, token)
                source_hint = f"飞书文档({feishu_doc_token[:8]}…)"
            else:
                yield {"type": "error", "content": "飞书来源需提供 doc_token 或 wiki_token"}
                return
        else:
            yield {"type": "error", "content": f"不支持的来源类型: {source_type}"}
            return

        if not raw.strip():
            yield {"type": "error", "content": "内容为空，无法提取 Skill"}
            return

        yield {"type": "fetch_done", "chars": len(raw)}
        yield {"type": "distill_start"}

        skills = await _distill_skills(raw, source_hint, model)

        saved = 0
        skipped = 0
        for s in skills:
            name = s.get("name", "").strip()
            description = s.get("description", "").strip()
            tags = s.get("tags", [])
            prompt_template = s.get("prompt_template", "").strip()

            if not name or not prompt_template:
                skipped += 1
                continue

            if _is_duplicate(name, description):
                yield {"type": "skill_skipped", "name": name, "reason": "duplicate"}
                skipped += 1
                continue

            meta = create_skill(
                name=name,
                description=description,
                prompt_template=prompt_template,
                tags=tags,
            )
            yield {"type": "skill_saved", "slug": meta["slug"],
                   "name": name, "description": description}
            saved += 1

        yield {"type": "extract_done", "saved": saved, "skipped": skipped}

    except Exception as e:
        yield {"type": "error", "content": str(e)}
