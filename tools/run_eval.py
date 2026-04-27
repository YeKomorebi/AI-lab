"""
eval runner — 对任意 Claude 模型跑 eval cases，输出行为差异报告

用法:
  python tools/run_eval.py --skill colleague/example_zhangsan --model claude-sonnet-4-6
  python tools/run_eval.py --all --model claude-sonnet-4-6
  python tools/run_eval.py --all --compare-model claude-opus-4-7

输出:
  reports/eval_report_{model}_{timestamp}.json
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

try:
    import yaml
    import anthropic
except ImportError:
    print("缺少依赖，请先运行: pip install anthropic pyyaml")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
CASES_DIR = ROOT / "eval" / "cases"
REPORTS_DIR = ROOT / "reports"
SKILLS_DIR = ROOT / "skills"


# ─────────────────────────────────────────────
# SKILL 加载
# ─────────────────────────────────────────────

def load_skill_prompt(skill_path: str) -> str:
    """从 skill_path (如 colleague/example_zhangsan) 加载系统 prompt。
    优先读 SKILL.md，fallback 到 persona.md + work.md 拼接。
    """
    parts = skill_path.split("/")
    skill_dir = SKILLS_DIR.joinpath(*parts)

    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        return skill_md.read_text(encoding="utf-8")

    # fallback: 拼接 persona + work
    persona = skill_dir / "persona.md"
    work = skill_dir / "work.md"
    sections = []
    if persona.exists():
        sections.append(persona.read_text(encoding="utf-8"))
    if work.exists():
        sections.append(work.read_text(encoding="utf-8"))

    if not sections:
        raise FileNotFoundError(f"找不到 skill 文件: {skill_dir}")

    return "\n\n---\n\n".join(sections)


# ─────────────────────────────────────────────
# EVAL CASE 加载
# ─────────────────────────────────────────────

def load_cases(skill_path: str) -> list[dict]:
    """根据 skill_path 找到对应的 YAML case 文件并加载。"""
    slug = skill_path.split("/")[-1].replace("example_", "")
    yaml_file = CASES_DIR / skill_path.split("/")[0] / f"{slug}.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"找不到 eval cases 文件: {yaml_file}")

    with yaml_file.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("cases", [])


def load_all_cases() -> list[tuple[str, dict]]:
    """加载 cases/ 目录下所有 YAML 中的 cases，返回 (skill_path, case) 列表。"""
    results = []
    for yaml_file in CASES_DIR.rglob("*.yaml"):
        with yaml_file.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        skill_path = data.get("skill", "")
        for case in data.get("cases", []):
            results.append((skill_path, case))
    return results


# ─────────────────────────────────────────────
# LLM 调用
# ─────────────────────────────────────────────

def call_model(client: anthropic.Anthropic, model: str, system: str, user: str) -> str:
    """调用 Claude API，返回回复文本。"""
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


# ─────────────────────────────────────────────
# 规则打分
# ─────────────────────────────────────────────

def check_rule(response: str, rule_key: str, rule_value: Any) -> tuple[bool, str]:
    """对单条规则进行检查，返回 (passed, detail)。"""
    resp_lower = response.lower()

    if rule_key == "must_contain":
        for kw in rule_value:
            if kw.lower() not in resp_lower:
                return False, f"缺少关键词: '{kw}'"
        return True, "OK"

    if rule_key == "must_contain_any":
        for kw in rule_value:
            if kw.lower() in resp_lower:
                return True, f"包含: '{kw}'"
        return False, f"应至少包含其中之一: {rule_value}"

    if rule_key == "must_not_contain":
        for kw in rule_value:
            if kw.lower() in resp_lower:
                return False, f"不应包含: '{kw}'"
        return True, "OK"

    if rule_key == "emoji_required":
        emoji_pattern = re.compile(
            "[\U00010000-\U0010ffff"
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]",
            flags=re.UNICODE,
        )
        if rule_value and not emoji_pattern.search(response):
            return False, "应包含 emoji"
        return True, "OK"

    # style / tone 暂时跳过（需要 LLM 评估，rule_based 模式不处理）
    return True, f"规则 '{rule_key}' 跳过（需 LLM 评估）"


def score_case(response: str, expected: dict, method: str) -> dict:
    """对单个 case 的回复打分，返回结构化结果。"""
    results = []
    passed_count = 0
    total_count = 0

    for key, value in expected.items():
        if key in ("style", "tone"):
            # 这两类需要 LLM judge，rule_based 模式记录待评估
            results.append({"rule": key, "passed": None, "detail": "需 LLM judge"})
            continue

        passed, detail = check_rule(response, key, value)
        results.append({"rule": key, "passed": passed, "detail": detail})
        total_count += 1
        if passed:
            passed_count += 1

    score = passed_count / total_count if total_count > 0 else 1.0
    return {"score": round(score, 3), "rules": results}


# ─────────────────────────────────────────────
# 主运行逻辑
# ─────────────────────────────────────────────

def run_single_skill(client: anthropic.Anthropic, model: str, skill_path: str) -> list[dict]:
    system_prompt = load_skill_prompt(skill_path)
    cases = load_cases(skill_path)
    results = []

    for case in cases:
        case_id = case.get("id", "unknown")
        scenario = case.get("scenario", "")
        user_input = case.get("input", "")
        expected = case.get("expected_behavior", {})
        threshold = case.get("scoring", {}).get("pass_threshold", 0.8)
        method = case.get("scoring", {}).get("method", "rule_based")

        print(f"  [{case_id}] 运行中...", end=" ", flush=True)
        try:
            response = call_model(client, model, system_prompt, user_input)
            score_result = score_case(response, expected, method)
            passed = score_result["score"] >= threshold

            print(f"{'✅' if passed else '❌'} score={score_result['score']}")
            results.append({
                "id": case_id,
                "skill": skill_path,
                "scenario": scenario,
                "passed": passed,
                "score": score_result["score"],
                "threshold": threshold,
                "rules": score_result["rules"],
                "response_preview": response[:200],
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": case_id,
                "skill": skill_path,
                "scenario": scenario,
                "passed": False,
                "score": 0,
                "error": str(e),
            })

        time.sleep(0.5)  # 避免触发 rate limit

    return results


def run_all(client: anthropic.Anthropic, model: str) -> list[dict]:
    all_results = []
    skill_paths = sorted(set(sp for sp, _ in load_all_cases()))

    for skill_path in skill_paths:
        print(f"\n── {skill_path} ──")
        results = run_single_skill(client, model, skill_path)
        all_results.extend(results)

    return all_results


# ─────────────────────────────────────────────
# 报告输出
# ─────────────────────────────────────────────

def write_report(results: list[dict], model: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "-").replace(":", "-")
    report_path = REPORTS_DIR / f"eval_report_{safe_model}_{ts}.json"

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed_cases = [r for r in results if not r.get("passed")]

    report = {
        "model": model,
        "timestamp": ts,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total else 0,
        },
        "failed_cases": [
            {"id": r["id"], "skill": r["skill"], "score": r.get("score", 0)}
            for r in failed_cases
        ],
        "details": results,
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def print_summary(results: list[dict], model: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{'─'*50}")
    print(f"模型: {model}")
    print(f"总计: {total} cases  通过: {passed}  失败: {total - passed}")
    print(f"通过率: {passed / total * 100:.1f}%" if total else "")

    failed = [r for r in results if not r.get("passed")]
    if failed:
        print("\n失败 cases:")
        for r in failed:
            print(f"  ❌ [{r['id']}]  score={r.get('score', 0):.2f}  skill={r['skill']}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="对 Claude 模型跑 colleague-skill eval cases")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--skill", help="单个 skill path，如 colleague/example_zhangsan")
    group.add_argument("--all", action="store_true", help="跑所有 cases")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="要测试的模型 ID")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # 读取环境变量 ANTHROPIC_API_KEY

    if args.skill:
        print(f"运行 skill: {args.skill}  模型: {args.model}")
        results = run_single_skill(client, args.model, args.skill)
    else:
        print(f"运行所有 cases  模型: {args.model}")
        results = run_all(client, args.model)

    print_summary(results, args.model)
    report_path = write_report(results, args.model)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
