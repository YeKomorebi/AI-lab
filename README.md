<div align="center">

# AI Lab 工作台

### 多 Agent 协作 · 需求→评审→算法 全流程

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Claude](https://img.shields.io/badge/Powered%20by-Claude-blueviolet)](https://anthropic.com)

</div>

---

模拟一个小型 AI 化公司团队，四位 AI 成员驻场三个功能房间，你用自然语言描述需求，他们从各自专业视角协同讨论，一键移交到下一个环节。

## 功能

**三个协作房间**

| 房间 | 驻场成员 | 职责 |
|------|---------|------|
| 💡 需求讨论室 | 佳秀、张三、明志 | 业务拆解、工期评估、可行性分析 |
| 🔍 技术评审室 | 张三、天意、明志 | 代码评审、安全检查、算法验证（并行）|
| 🧪 算法实验室 | 明志、张三、天意 | 实验设计、工程实现、数据安全 |

**核心交互**
- `@` 指定成员单独回复，或全员同时响应
- 一键移交：需求讨论室 → 技术评审室 → 算法实验室，自动生成结构化移交摘要，目标房间成员读取摘要后主动开场发言
- 上传文档（txt / md / pdf / docx / 代码文件）或图片（Claude Vision 直接识别），AI 成员能读取内容参与分析

**成员管理**
- 编辑现有成员的基本信息、Persona（性格/说话风格）、Work（职责/技术规范）、所属房间
- 新建自定义成员，立即可以加入房间参与协作

## 四位内置成员

| 成员 | 角色 | 性格 |
|------|------|------|
| 张三 | 后端技术 | 字节跳动 L2-1，INTJ，凡事先问 impact，习惯甩锅 |
| 天意 | 安全 | 安全评审必给替代方案，不只是拦截 |
| 佳秀 | 业务/人员 | 关注时间线和用户体验，招聘进展随时更新 |
| 明志 | 算法 | 没有 baseline 不谈效果，实验必须严谨 |

## 快速开始

**安装依赖**
```bash
git clone https://github.com/YeKomorebi/colleague-skill
cd colleague-skill
pip install -r requirements.txt
```

**启动服务**
```bash
# 官方 Anthropic Key
ANTHROPIC_API_KEY=你的key uvicorn app.server:app --port 8000

# 第三方代理
ANTHROPIC_API_KEY=你的key ANTHROPIC_BASE_URL=https://代理地址/v1 uvicorn app.server:app --port 8000
```

浏览器打开 http://127.0.0.1:8000

**局域网共享（让同一 WiFi 下的人访问）**
```bash
ANTHROPIC_API_KEY=你的key uvicorn app.server:app --host 0.0.0.0 --port 8000
```
对方用你的局域网 IP 访问，如 `http://192.168.1.x:8000`

## 项目结构

```
colleague-skill/
├── app/
│   ├── server.py              # FastAPI 服务，所有 API 端点
│   ├── orchestrator.py        # 房间调度：派发消息、移交摘要、开场发言
│   ├── orchestrator_review.py # 三阶段技术评审（高级模式）
│   └── static/
│       └── index.html         # 前端单页应用
├── skills/colleague/          # 成员 Skill 文件
│   ├── example_zhangsan/      #   张三：persona.md + work.md + meta.json
│   ├── example_tianyi/
│   ├── example_jiaxiu/
│   └── example_mingzhi/
├── eval/                      # 成员行为评测用例
├── tools/
│   └── run_eval.py            # 运行评测
└── requirements.txt
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/rooms` | 获取所有房间配置 |
| GET | `/api/agents` | 获取所有成员列表 |
| GET | `/api/agents/{key}` | 获取成员详情（含 persona/work 内容）|
| PUT | `/api/agents/{key}` | 更新成员信息 |
| POST | `/api/agents` | 创建自定义成员 |
| POST | `/api/room/{id}/chat` | 发送消息（返回 job_id）|
| GET | `/api/room/{id}/stream/{job_id}` | SSE 实时流式响应 |
| POST | `/api/room/{id}/transfer` | 发起房间移交 |
| GET | `/api/transfer/stream/{job_id}` | SSE 移交进度流 |
| POST | `/api/upload` | 上传文件（文本提取 / 图片 base64）|

## 环境变量

| 变量 | 必须 | 说明 |
|------|------|------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API Key |
| `ANTHROPIC_BASE_URL` | 可选 | 第三方代理地址，默认 `https://api.anthropic.com` |

## 自定义成员

在「成员管理」页面点「+ 新建成员」，填写：
- **基本信息**：姓名、角色、专注点、所属房间
- **Persona**：性格描述、说话风格、口头禅（直接写 Markdown）
- **Work**：职责范围、技术规范、工作流程

也可以直接在 `skills/colleague/` 下新建目录，放 `persona.md` + `work.md` + `meta.json`，重启服务后自动加载。

---

## 致谢

本项目基于 [titanwings/colleague-skill](https://github.com/titanwings/colleague-skill)（dot-skill）构建。

原项目提供了四位 AI 成员的完整 Skill 文件（Persona + Work 双层架构）以及 colleague-skill 的技术方案，本项目在此基础上实现了多房间协作 App、一键移交、成员管理和文件上传等功能。

---

<div align="center">

**MIT License** © [YeKomorebi](https://github.com/YeKomorebi)

</div>
