<div align="center">

# AI Room 工作台

### 多 Agent 协作 · 动态房间 · 自我进化

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Claude](https://img.shields.io/badge/Powered%20by-Claude-blueviolet)](https://anthropic.com)

</div>

---

一个基于 Claude 的多 Agent 协作工作台。用自然语言和 AI 成员对话，按需创建/删除/改造成员，房间之间一键移交，成员能在对话中自我完善技能。

## 核心特性

**动态房间系统**
- 自由创建房间，任意搭配成员
- `@全员` 让所有成员并行响应，`@某人` 点名单独回复
- 一键移交：自动生成结构化摘要，目标房间成员读取后主动开场
- 房间记忆：每 10 条消息由隐形 CEO agent 自动提炼对话要点，注入所有成员的上下文

**造物主（成员设计师）**
- 在对话中描述需求，造物主自动设计 Prompt 架构（Layer 0~3）并创建成员
- 支持在对话中直接删除成员、给指定成员赋予新能力

**Agent 自我进化**
- 任何成员在对话中输出 `agent-patch` 块，系统自动将内容追加到其 skill 文件，下次对话即生效
- 老板在造物主房间说「给 XX 加上 YY 能力」，造物主设计 patch 内容并自动注入

**Skill 提取引擎**
- 从飞书文档、任意 URL 或纯文本中提炼可复用 skill，自动去重后存入 Skill Library
- Skill Library 支持自动合并相似 skill、淘汰低置信度 skill

**资源管理器**
- 每个房间内置类 VSCode 文件面板，支持新建/重命名/拖拽/预览
- 文件可一键发送到对话作为附件

**主题系统**
- 内置 6 套配色预设，支持自定义 CSS 变量，持久化保存

## 快速开始

```bash
git clone https://github.com/YeKomorebi/colleague-skill
cd colleague-skill
pip install -r requirements.txt

# 启动
ANTHROPIC_API_KEY=your_key uvicorn app.server:app --port 8000

# 第三方代理
ANTHROPIC_API_KEY=your_key ANTHROPIC_BASE_URL=https://proxy/v1 uvicorn app.server:app --port 8000

# 局域网共享
ANTHROPIC_API_KEY=your_key uvicorn app.server:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://127.0.0.1:8000

## 项目结构

```
colleague-skill/
├── app/
│   ├── server.py              # FastAPI，所有 API 端点
│   ├── orchestrator.py        # 房间调度、房间记忆、移交摘要
│   ├── static/
│   │   └── index.html         # 前端单页应用（Vanilla JS）
│   └── skill_engine/
│       ├── library.py         # Skill Library CRUD / 合并 / 淘汰
│       ├── extractor.py       # 从飞书/URL/文本提炼 skill（流式）
│       ├── decomposer.py      # 任务分解
│       ├── executor.py        # Skill 执行
│       └── evaluator.py       # 执行结果评估
├── skills/
│   ├── colleague/             # 成员 Skill 文件
│   │   ├── creator_god/       #   造物主：persona.md + work.md
│   │   ├── example_zhangsan/  #   张三
│   │   ├── example_jiaxiu/    #   佳秀
│   │   ├── example_mingzhi/   #   明志
│   │   └── custom_*/          #   用户创建的自定义成员
│   └── task_skills/           # Skill Library（自动管理）
├── data/
│   ├── rooms.json             # 房间配置持久化
│   └── sessions/              # 对话历史 + 房间记忆缓存
└── requirements.txt
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/rooms` | 所有房间 |
| POST | `/api/rooms` | 创建房间 |
| PUT | `/api/rooms/{id}` | 更新房间 |
| DELETE | `/api/rooms/{id}` | 删除房间 |
| GET | `/api/agents` | 所有成员 |
| POST | `/api/agents` | 创建成员 |
| GET | `/api/agents/{key}` | 成员详情 |
| PUT | `/api/agents/{key}` | 更新成员 |
| PATCH | `/api/agents/{key}/skill` | 追加/替换成员 skill 文件（自我进化接口）|
| DELETE | `/api/agents/{key}` | 删除成员 |
| POST | `/api/room/{id}/chat` | 发送消息（返回 job_id）|
| GET | `/api/room/{id}/stream/{job_id}` | SSE 流式响应 |
| GET | `/api/room/{id}/history` | 对话历史 |
| DELETE | `/api/room/{id}/history` | 清空历史 |
| POST | `/api/room/{id}/transfer` | 发起移交 |
| GET | `/api/transfer/stream/{job_id}` | SSE 移交进度 |
| POST | `/api/upload` | 上传文件 |
| POST | `/api/skill-engine/extract` | SSE 流式 skill 提取 |
| POST | `/api/skill-engine/run` | SSE 流式任务执行 |

## 对话内指令块

AI 成员回复中可包含以下特殊块，系统自动执行：

| 块类型 | 触发方 | 效果 |
|--------|--------|------|
| ` ```agent-create ``` ` | 造物主 | 创建新成员 |
| ` ```agent-delete ``` ` | 造物主 | 删除指定成员 |
| ` ```agent-patch ``` ` | 任意成员 / 造物主 | 追加内容到指定成员（或自身）的 skill 文件 |

`agent-patch` 格式：
```json
{
  "target": "成员名（省略则更新自身）",
  "key": "成员 slug（可省略）",
  "section": "work",
  "mode": "append",
  "content": "要追加的 Markdown 内容"
}
```

## 环境变量

| 变量 | 必须 | 说明 |
|------|------|------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API Key |
| `ANTHROPIC_BASE_URL` | 可选 | 第三方代理，默认 `https://api.anthropic.com` |

---

## 致谢

本项目基于 [titanwings/colleague-skill](https://github.com/titanwings/colleague-skill)（dot-skill）构建，原项目提供了 AI 成员的 Skill 双层架构（Persona + Work）设计方案。

---

<div align="center">

**MIT License** © [YeKomorebi](https://github.com/YeKomorebi)

</div>
