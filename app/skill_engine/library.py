"""
library.py — 模块四：Skill Library 管理

职责：存储 / 检索 / 合并 / 淘汰 task_skills
"""

from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SKILLS_DIR = Path(__file__).parent.parent.parent / "skills" / "task_skills"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

DEPRECATE_THRESHOLD = 0.3   # confidence 低于此值且 usage 足够时淘汰
DEPRECATE_MIN_USAGE = 5     # 至少执行过 5 次才考虑淘汰
MERGE_SIMILARITY_THRESHOLD = 0.92  # 关键词 Jaccard 相似度超过此值则合并


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_from_name(name: str) -> str:
    slug = re.sub(r"[^\w一-鿿]", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "skill"


def _token_set(text: str) -> set[str]:
    """把描述拆成词集合，用于 Jaccard 相似度。"""
    tokens = re.findall(r"[\w一-鿿]+", text.lower())
    return set(tokens)


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _ema_confidence(old: float, success: bool) -> float:
    """指数加权移动平均更新置信度。"""
    return round(0.8 * old + 0.2 * (1.0 if success else 0.0), 4)


# ── 基础 CRUD ─────────────────────────────────────────────────────────────────

def get_skill(slug: str) -> Optional[dict]:
    meta_path = SKILLS_DIR / slug / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def get_skill_prompt(slug: str) -> str:
    skill_path = SKILLS_DIR / slug / "skill.md"
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8")
    return ""


def list_skills(status: Optional[str] = "active") -> list[dict]:
    result = []
    if not SKILLS_DIR.exists():
        return result
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status is None or meta.get("status") == status:
            result.append(meta)
    return result


def create_skill(
    name: str,
    description: str,
    prompt_template: str,
    tools: list[str] | None = None,
    tags: list[str] | None = None,
    slug: str | None = None,
) -> dict:
    slug = slug or _slug_from_name(name)
    # 避免重名：加数字后缀
    base_slug = slug
    counter = 1
    while (SKILLS_DIR / slug).exists():
        slug = f"{base_slug}_{counter}"
        counter += 1

    skill_dir = SKILLS_DIR / slug
    skill_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "slug": slug,
        "name": name,
        "description": description,
        "tools": tools or [],
        "tags": tags or [],
        "confidence": 0.5,
        "usage_count": 0,
        "success_count": 0,
        "status": "active",
        "created_at": _now(),
        "updated_at": _now(),
        "version": "v1",
    }
    (skill_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (skill_dir / "skill.md").write_text(prompt_template, encoding="utf-8")
    return meta


def update_skill_meta(slug: str, **fields) -> Optional[dict]:
    meta = get_skill(slug)
    if meta is None:
        return None
    meta.update(fields)
    meta["updated_at"] = _now()
    (SKILLS_DIR / slug / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def update_skill_prompt(slug: str, new_prompt: str) -> None:
    (SKILLS_DIR / slug / "skill.md").write_text(new_prompt, encoding="utf-8")


def delete_skill(slug: str) -> bool:
    skill_dir = SKILLS_DIR / slug
    if not skill_dir.exists():
        return False
    shutil.rmtree(skill_dir)
    return True


# ── 使用记录 ──────────────────────────────────────────────────────────────────

def record_execution(slug: str, success: bool) -> Optional[dict]:
    meta = get_skill(slug)
    if meta is None:
        return None
    meta["usage_count"] = meta.get("usage_count", 0) + 1
    if success:
        meta["success_count"] = meta.get("success_count", 0) + 1
    meta["confidence"] = _ema_confidence(meta.get("confidence", 0.5), success)
    meta["updated_at"] = _now()
    (SKILLS_DIR / slug / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


# ── 检索 ──────────────────────────────────────────────────────────────────────

def search_skills(query: str, top_k: int = 3) -> list[dict]:
    """
    关键词 Jaccard 相似度检索，返回 top_k 个 active skill。
    """
    active = list_skills(status="active")
    scored = []
    for meta in active:
        text = f"{meta.get('name', '')} {meta.get('description', '')} {' '.join(meta.get('tags', []))}"
        score = _jaccard(query, text)
        scored.append((score, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k] if _ > 0]


# ── 合并 ──────────────────────────────────────────────────────────────────────

def merge_similar_skills() -> list[tuple[str, str]]:
    """
    检测两两 description 相似度超过阈值的 skill，保留 confidence 更高的，
    另一个标为 deprecated。返回被合并的 (deprecated_slug, kept_slug) 列表。
    """
    active = list_skills(status="active")
    merged: list[tuple[str, str]] = []
    deprecated_set: set[str] = set()

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            if a["slug"] in deprecated_set or b["slug"] in deprecated_set:
                continue
            sim = _jaccard(
                f"{a['name']} {a['description']}",
                f"{b['name']} {b['description']}",
            )
            if sim >= MERGE_SIMILARITY_THRESHOLD:
                keep, drop = (a, b) if a.get("confidence", 0) >= b.get("confidence", 0) else (b, a)
                update_skill_meta(drop["slug"], status="deprecated")
                deprecated_set.add(drop["slug"])
                merged.append((drop["slug"], keep["slug"]))

    return merged


# ── 淘汰 ──────────────────────────────────────────────────────────────────────

def deprecate_low_confidence() -> list[str]:
    """
    将 confidence 低于阈值且 usage_count 足够多的 skill 标为 deprecated。
    返回被淘汰的 slug 列表。
    """
    deprecated = []
    for meta in list_skills(status="active"):
        if (
            meta.get("usage_count", 0) >= DEPRECATE_MIN_USAGE
            and meta.get("confidence", 1.0) < DEPRECATE_THRESHOLD
        ):
            update_skill_meta(meta["slug"], status="deprecated")
            deprecated.append(meta["slug"])
    return deprecated
