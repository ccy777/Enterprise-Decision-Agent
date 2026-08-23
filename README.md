# 企业决策智能体

> **面向企业知识与经营数据的 AI Agent 平台**

[![CI](https://github.com/ccy777/enterprise-decision-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ccy777/enterprise-decision-agent/actions/workflows/ci.yml)
![Runtime base](https://img.shields.io/badge/runtime%20base-v1.0.2-2563eb)
![Python](https://img.shields.io/badge/python-3.11%2B-3776ab)
![Runtime](https://img.shields.io/badge/runtime-agent%20workflow-0f766e)

这是一个面向企业知识问答、经营数据分析与库存风险诊断的 AI Agent 平台。系统通过 Router 识别 `Knowledge`、`Data` 与 `Mixed` 三类任务，并由对应 Skill 完成知识检索、数据查询或知识与数据联合决策。

项目基于 LangGraph 编排 Agent Workflow，集成 Hybrid RAG、MCP Data Agent、Context / Memory、Tool Calling、Trace 与 Evaluation，并通过 FastAPI 提供统一服务接口。执行链同时保留必要的 Scope、Evidence、Citation 与 Reviewer 边界，使结果可追踪、可评测、可复核。

## 真实运行界面


### 知识与数据联合决策

![知识与数据联合决策的真实运行结果](docs/assets/demo/mixed-result.png)

同一次请求由 Router 选择 `mixed`，执行 `inventory-risk-diagnosis`，回答同时包含数据引用 `[D1]` 与知识引用 `[E1]`、`[E2]`。

### 执行链追踪

![真实请求的执行链追踪](docs/assets/demo/mixed-trace.png)

Trace 展示 routing、planning、MCP Tool Calling、Knowledge retrieval、evidence selection、review 与 answer generation；公开投影不展示 Prompt、SQL、业务记录、Evidence/Audit 正文或凭据。

### 越权请求阻断

![越权数据请求在调用前被阻断](docs/assets/demo/security-fail-closed.png)

`data_scope_denied` 是固定安全评测的真实案例：Provider 调用 0、Tool 调用 0、响应未发布，评测通过。

## 核心能力

- **工作流与提示词工程**：基于 LangGraph 编排 `Router -> Planner -> Skill -> Tool -> Reviewer` 多步骤工作流，通过 Prompt 与 Pydantic 结构化输出约束阶段输入输出。
- **上下文与会话记忆**：支持 Redis / In-Memory 会话状态、TTL、滚动摘要与上下文预算，按租户、用户和会话隔离多轮任务。
- **混合检索与离线评测**：采用 `Dense + BM25 -> RRF -> Cross-Encoder -> Parent Expansion` 检索链，并通过 200 条企业场景查询验证排序效果。
- **MCP 数据智能体与工具调用**：通过 Native Tool Calling 选择并执行 Data Agent，由 Data Planner 生成结构化 SQL 计划，再调用 MCP Tools 完成 Schema 获取、MySQL 查询与结果分析。
- **链路追踪与智能体评测**：记录路由、规划、Skill、MCP、Evidence 与 Reviewer 的阶段状态、错误码和耗时，并建立离线回归与边界评测。
- **三类任务路由**：统一处理企业知识问答、经营数据分析以及知识与数据联合决策。
- **执行边界与工程交付**：使用 SQLGlot、DataScope、Evidence、Citation 与 Reviewer 约束执行范围，CI 覆盖质量、测试、安全和依赖检查。

## 为什么不是普通 RAG 项目

| 路径 | Agent 执行内容 | 典型输出 |
| --- | --- | --- |
| **Knowledge** | Hybrid retrieval、Evidence Selection、Answerability 与 Citation | 带文档依据的企业知识回答 |
| **Data** | Data Planner、Tool Calling、MCP Tool 与 MySQL 数据分析 | 结构化查询结果与经营数据结论 |
| **Mixed** | 将 Knowledge Evidence 与 Data Evidence 组合并交由 Reviewer 检查 | 同时引用制度与业务数据的决策建议 |

因此，这不是简单的 `Question -> Vector DB -> LLM`：系统会先判断任务类型，再执行相应 Skill 和 Tool，并基于知识或数据 Evidence 生成结果。

## 整体架构

```mermaid
flowchart TD
    U["User request"] --> API["FastAPI"]
    API --> X["Formal Request Executor"]
    X --> R["Router"]
    R --> C["Coordinator / Skill Registry"]
    C --> S["Selected Skill"]
    S --> T["Native Tool Calling"]
    T --> K["Knowledge Agent\nHybrid RAG"]
    T --> D["Data Agent\nData Planner"]
    D --> MCP["MCP Tools"]
    MCP --> DB["MySQL"]
    K --> KE["Knowledge Evidence"]
    DB --> DE["Data Evidence"]
    KE --> M["Mixed Synthesis"]
    DE --> M
    KE --> V["Reviewer / Response"]
    DE --> V
    M --> V
    X -.-> CM["Context / Memory"]
    X -.-> O["Trace / Evaluation"]
```

上图展示面试和项目学习所需的主执行链；Scope、Provider governance、Audit 与 release gate 等工程边界保留在正式 Runtime 中。具体设计见 [Architecture](docs/ARCHITECTURE.md) 与 [Agent Workflow](docs/AGENT_WORKFLOW.md)。

![企业决策智能体网页演示界面](docs/assets/demo-ui.png)

> 内置 FastAPI Demo UI 会展示 runtime readiness、answers、citations、route、selected skill、memory state 与 execution trace。截图刻意展示未配置 Provider 时的 unready runtime：系统会 fail-closed，而不是伪造 ready 状态。

## 核心模块

### 1. 工作流与提示词工程

LangGraph 编排 Router、Planner、Skill、Tool、Reviewer 与 Release stages。Router、Data Planner、Evidence Selector、Answerability Reviewer 与 Workflow Reviewer 使用各自的 Prompt 和 Pydantic 结构化输出契约，条件分支与状态对象负责串联知识问答、数据分析和异常处理。

### 2. 混合检索（Hybrid RAG）

```text
Dense + BM25 -> RRF -> Cross-Encoder -> Parent Expansion
                              -> Evidence Selection -> Citation
```

检索链同时使用 semantic 与 lexical recall，对融合候选进行 reranking，扩展相关 Parent context，并区分 candidate evidence 与可发布的 Citation。系统支持 Answerability review，不宣称消除 hallucination。详见 [Hybrid RAG](docs/HYBRID_RAG.md)。

### 3. MCP 数据智能体与工具调用

Native Tool Calling 根据已选路由调用 `run_data_agent`；Data Agent 通过 MCP 获取 Schema 与业务定义，由 Data Planner 将自然语言问题转换为结构化 SQL 计划，再调用 MCP Tool 完成 MySQL 查询与结果分析。查询链使用 SQLGlot、table / column allowlist、`LIMIT`、timeout 与 DataScope 约束查询范围。详见 [Data Agent & MCP](docs/DATA_AGENT_AND_MCP.md)。

### 4. 上下文与会话记忆

Context / Memory 按 tenant、user、session 做有界隔离。runtime 支持 in-memory 或 Redis-backed state、TTL、state version、rolling summary 与 bounded context，而不是无限累积会话内容。

### 5. 链路追踪与智能体评测

系统为 Routing、Planning、Skill、Tool Calling、Retrieval、Evidence 与 Reviewer 建立请求级 Trace，记录阶段状态、错误码和耗时。离线验证覆盖稳定集成测试、确定性边界评测与可独立复算的 Retrieval Benchmark，详见 [Evaluation](docs/EVALUATION.md)。

### 6. 安全边界与结果审查

SecurityContext 与 Scope checks 覆盖 request、workflow、skill、tool、data、knowledge 和 Response Release 边界；缺少授权时默认 fail-closed。Evidence validation、Reviewer、release gate 与本地可验证 Audit hash chain 共同让决策路径可审查。详见 [Security Boundaries](docs/SECURITY_BOUNDARIES.md)。

## 检索评测 v2

**状态：FROZEN / ADOPTED。** Retrieval Benchmark v2 是本项目当前公开 Benchmark。

| 组成 | 数值 |
| --- | ---: |
| Queries | 200 |
| Answerable / Unanswerable | 160 / 40 |
| Enterprise documents | 12 |
| Parent records | 36 |
| Child evidence windows | 101 |

ranking metrics 以 **160 Answerable queries** 为 denominator；40 个 Unanswerable queries 被明确排除在 Hit@K 与 MRR denominator 之外。

| 指标 | RRF | Cross-Encoder Reranker | 增益 |
| --- | ---: | ---: | ---: |
| Child Hit@1 | 85.62% | 96.25% | +10.63pp |
| Hit@5 | 100.00% | 100.00% | — |
| MRR@5 | 91.69% | 98.12% | +6.44pp |

完整公开证据包、冻结相关性集、排名记录和复核命令见 [Retrieval Benchmark v2 public evidence](artifacts/public-evaluation/retrieval-v2/README.md)。

relevance freeze 区分 single-window answer sufficiency 与 multi-evidence retrieval relevance。Cross-Encoder reranking 显著提升前排 Evidence relevance；但 multi-evidence evaluation 也揭示了真实的 precision–coverage trade-off：将 Top-K 过度集中于最相似 Evidence，可能降低 evidence diversity 与 full evidence coverage。因此不能声称 reranking 改善所有 retrieval objective。

### 历史 v1 基线

原 50-query 结果保留为 **Historical Baseline**，而非当前 Benchmark：46 Answerable / 4 Unanswerable queries，RRF Child Hit@1 为 **84.78%**，Cross-Encoder Child Hit@1 为 **93.48%**。当前参考口径以上述 Retrieval Benchmark v2 为准。

> 这些结果来自 frozen synthetic-enterprise benchmark，不是 production accuracy、production SLA、通用现实场景表现，也不代表该 benchmark 之外的 100% retrieval quality。

## 工程质量

项目强调可验证的工程证据，而不是不断变化的测试数量。

| 公开验证项 | 当前结果 |
| --- | ---: |
| 单元测试 | 1,802 项通过 |
| 稳定离线集成测试 | 235 项通过 |
| 确定性边界评测 | 28 / 28 通过 |

- GitHub Actions 对每个 Pull Request 执行 `quality`、`unit`、`security-evaluation`、`secret-scan`、`dependency-scan` 与 `offline-integration`。
- Offline verification 与 selected tests 不需要 Provider、外部数据库或网络服务。
- Retrieval evidence 使用 committed verifier 检查，而非重新运行高成本 model evaluation。
- dependency scanning 是 CI 强制门禁：vulnerability 会在 merge 前被阻断，而不是在发布后才记录。
- secret scanning 与 fail-closed security evaluation 是一等交付门禁。

## 本地运行

### 快速验证——无需配置模型服务

安装 locked environment 后运行仓库提交的 Retrieval Benchmark v2 verifier。该操作从公开的冻结相关性集与排名记录重新计算简历使用的 Hit@1 / MRR@5，不加载模型，也不调用 Provider。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts\verify_retrieval_v2_evidence.py
```

### 完整演示——使用本地配置

请参阅 [Local Demo](docs/LOCAL_DEMO.md) 以及 [`.env.example`](.env.example) 中的字段说明，配置自己的 compatible Provider 与本地基础设施。仓库不包含可用 credentials、model weights、database volume 或 runtime audit log。

```powershell
docker compose config
docker compose up -d

.\.venv\Scripts\python.exe scripts\run_local_demo.py knowledge
.\.venv\Scripts\python.exe scripts\run_local_demo.py data
.\.venv\Scripts\python.exe scripts\run_local_demo.py mixed
.\.venv\Scripts\python.exe scripts\run_local_web_demo.py mixed
```

前三种模式分别演示 Knowledge、Data 和 Mixed CLI 路径；最后一条启动仅绑定 `127.0.0.1`、仅授予固定 Mixed Demo 最小 Scope 的网页入口。Provider calls 可能产生费用；前置条件与停止步骤见 Local Demo guide。

正式 ASGI 入口仍使用默认拒绝身份解析器，不因 Demo 降低安全策略：

```powershell
.\.venv\Scripts\python.exe -m uvicorn decision_agent.main:app --host 127.0.0.1 --port 8000
```

然后访问 `http://127.0.0.1:8000`。

## 项目结构

```text
src/decision_agent/   Agent runtime、security、retrieval、MCP、API 与 skills
tests/                Unit 与 offline integration tests
scripts/              Demo 与 committed verification entry points
datasets/             Sanitized synthetic knowledge、data 与 security fixtures
docs/                 Public architecture、workflow、security 与 demo guides
.github/              CI 与 dependency automation
```

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | 系统层次与端到端 request flow |
| [Agent Workflow](docs/AGENT_WORKFLOW.md) | Router、Planner、Skills、Reviewer 与 release controls |
| [Hybrid RAG](docs/HYBRID_RAG.md) | Retrieval、fusion、reranking、Parent Expansion 与 Evidence Selection |
| [Data Agent & MCP](docs/DATA_AGENT_AND_MCP.md) | MCP、DataScope、SQL Guard 与只读数据访问 |
| [Security Boundaries](docs/SECURITY_BOUNDARIES.md) | Identity、authorization、scope、provider 与 audit boundaries |
| [Local Demo](docs/LOCAL_DEMO.md) | 本地前置条件与 Knowledge / Data / Mixed walkthrough |
| [Limitations](docs/LIMITATIONS.md) | 已知约束与工程边界 |

## 项目范围与限制

这是一个受控工程系统，不宣称 unrestricted autonomy、zero hallucinations 或 production-scale guarantees。公开仓库提交了 Retrieval Benchmark v2 的 synthetic query relevance freeze、ranking-only records、verified metrics 与 failure analysis，用于离线复核简历指标；不会公开 credentials、private configurations、Provider prompts / outputs、runtime logs、private business data 或 internal filesystem paths。

仓库当前没有 project-wide open-source license。公开可见不等同于自动允许复制、修改或再分发代码。
