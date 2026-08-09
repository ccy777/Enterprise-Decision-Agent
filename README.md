# 企业决策 Agent（Enterprise Decision Agent）

> 面向企业知识问答、只读经营数据分析与综合决策场景的受控 AI Agent。

[![CI](https://github.com/ccy777/enterprise-decision-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ccy777/enterprise-decision-agent/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-v1.0.2-2563eb)
![Python](https://img.shields.io/badge/python-3.11%2B-3776ab)
![Runtime](https://img.shields.io/badge/runtime-controlled%20agent-0f766e)

它不是一个“接上大模型就回答”的普通聊天机器人。系统会在回答发布前依次处理身份与权限、路由、规划、知识或数据取证、证据筛选、答案审查、引用校验和发布控制，让每个结论都能说明：**用了什么能力、依据来自哪里、为什么允许返回。**

| 可复现的公开工程证据 | 结果 |
| --- | ---: |
| Unit Tests | **1,802 passed** |
| Stable Offline Integration | **235 passed** |
| Deterministic Security Cases | **28 / 28** |
| Frozen Retrieval Benchmark | **50 queries** |

![企业决策 Agent 内置 Demo UI](docs/assets/demo-ui.png)

> 仓库内置的 FastAPI Demo UI。图中未配置正式 Provider，因此 Runtime 按 fail-closed 规则如实显示“未就绪”；完成本地配置后，页面会展示回答、引用、路由、Skill、Memory 状态和执行轨迹。

## 这个项目解决什么问题

企业问答真正困难的部分通常不是“生成一段文字”，而是同时处理三类边界：

| 场景 | 典型问题 | 系统处理方式 |
| --- | --- | --- |
| 企业知识问答 | 保修、采购、库存制度是什么？ | 在授权文档内检索、筛选证据并生成引用 |
| 经营数据分析 | 哪些产品低于安全库存？ | 经 MCP、DataScope 和 SQL Guard 查询只读 MySQL |
| 综合决策分析 | 结合库存数据与补货制度给出建议 | 区分数据事实和制度依据，再统一审查后发布 |

系统的核心目标不是追求“自由行动”，而是让 Agent 在明确权限和证据边界内完成可追踪的企业任务。

## 效果展示

### 1. Knowledge：有依据的企业知识问答

固定演示问题：

```text
华衡智能科技有限公司的业务定位和本项目演示范围是什么？
```

系统只允许访问 `DOC-ORG-001` 和 `DOC-AGENT-001`，回答需要经过：

```text
KnowledgeScope
  -> Dense + BM25 检索
  -> RRF 融合
  -> Cross-Encoder 重排
  -> Evidence Selection
  -> Answerability
  -> Citation Validation
```

如果授权资料无法支持结论，系统应返回信息不足，而不是用常识补齐企业事实。

### 2. Data：受控的只读经营数据分析

固定演示问题：

```text
截至 2026 年 6 月 30 日，哪些产品低于安全库存？
```

公开仓库中的合成数据断言为：

| 产品 | 当前库存 | 安全库存 | 判断 |
| --- | ---: | ---: | --- |
| P100 · Aster 工业泵 | 40 | 50 | 低于安全库存 |
| P300 · Cirrus 传感器 | 90 | 100 | 低于安全库存 |
| P600 · Flux 工业线缆 | 110 | 120 | 低于安全库存 |

这条链路不是让模型直接连接数据库：

```text
Data Skill
  -> EnterpriseDataMCPClient
  -> MCP Server
  -> EnterpriseDataToolService
  -> SafeQueryService
  -> SQL Guard
  -> SQLAlchemy
  -> read-only MySQL
```

模型不会获得无限制数据库权限，也不会直接执行任意生成的 SQL。

### 3. Mixed：制度证据与数据事实联合分析

固定演示问题：

```text
结合库存数据与补货制度，分析截至 2026 年 6 月 30 日的库存风险并给出建议。
```

Mixed 路径会同时使用受控经营数据和 `DOC-INV-001` 补货制度，并在最终回答中区分：

- 哪些是数据库返回的库存事实；
- 哪些是制度文件给出的阈值或流程；
- 哪些建议仍需要人工审批；
- 当前证据不能支持哪些结论。

CLI/API 的正式响应结构如下（字段来自正式 API 合约，示例值不代表一次真实 Provider 输出）：

```json
{
  "status": "completed",
  "route": "knowledge | data | mixed",
  "skill": "selected-skill",
  "answer": "evidence-backed answer",
  "citations": ["approved evidence references"],
  "error_code": null
}
```

## 一张图看懂架构

```mermaid
flowchart TD
    U["用户请求"] --> S["RequestPrincipal + SecurityContext"]
    S --> F["FormalRequestExecutor"]
    F --> R["Router"]
    R --> C["Coordinator"]
    C --> P["Planner"]
    P --> K["Knowledge Skill"]
    P --> D["Data Skill"]
    P --> M["Mixed Skill"]
    K --> HR["Hybrid RAG"]
    HR --> E["Evidence + Citation"]
    D --> MCP["MCP Tool Boundary"]
    MCP --> SQL["DataScope + SQL Guard"]
    SQL --> DB["read-only MySQL"]
    M --> K
    M --> D
    E --> V["Reviewer"]
    DB --> V
    V --> G["Response Release Gate"]
    G --> A["Trace + Audit"]
```

主链可以概括为：

```text
Router -> Planner -> Skill -> Evidence -> Reviewer -> Release
```

详细设计见 [系统架构](docs/ARCHITECTURE.md) 和 [Agent 工作流](docs/AGENT_WORKFLOW.md)。

## 四个核心工程能力

### 1. 受控 Agent 编排

路由、规划、Skill 调度、证据筛选、审查和发布都是显式阶段。Planner 的输出不能创建权限、扩大 Scope，也不能绕过已注册的 Skill 和 Tool。

系统定义了九个闭集 `ProviderStage`：`routing`、`planning`、`data_planning`、`data_answer`、`evidence_selection`、`answerability_review`、`knowledge_answer`、`inventory_synthesis` 和 `workflow_review`。它们构成调用白名单与预算边界，**不表示每个请求固定调用模型九次**。

### 2. Hybrid RAG

```text
Dense Retrieval + BM25
          -> KnowledgeScope Filter
          -> RRF Fusion
          -> Cross-Encoder Reranker
          -> Parent Expansion
          -> Evidence Context
          -> Evidence Selection
          -> Citation
```

- Embedding：`BAAI/bge-small-zh-v1.5`，512 dimensions
- Reranker：`BAAI/bge-reranker-base`
- Chunking baseline：`fixed-window-v1`
- Vector store：Milvus

冻结的 50-query 工程基准中，Cross-Encoder 将 Child Hit@1 从 **84.78%** 提升到 **93.48%**，MRR@5 从 **91.67%** 提升到 **96.74%**。

详细说明见 [Hybrid RAG](docs/HYBRID_RAG.md)。

### 3. Controlled Data Agent

数据请求必须跨越 MCP Tool 边界，并通过：

- `DataScope`：限制可访问的数据域和资源；
- Safe Projection：只暴露回答需要的安全字段；
- SQL Guard：校验只读语句、表、列、行数和超时；
- Read-only Account：数据库账号不具备写权限。

详细说明见 [Data Agent 与 MCP](docs/DATA_AGENT_AND_MCP.md)。

### 4. Fail-closed Security 与审计

安全检查重复存在于 Request、Workflow、Skill、Tool、Data、Knowledge、Provider 和 Response Release 边界。缺少必要身份、Scope 或授权时，请求默认失败关闭。

覆盖的正式对象包括：

```text
RequestPrincipal / SecurityContext / Tenant / Session
Scenario / Workflow / Skill / Tool authorization
DataScope / KnowledgeScope / ProviderPolicy
Recursive Redaction / Response Release Gate
AuditEvent / Audit Hash Chain
```

Audit 是**单进程、本地文件、可验证 Hash Chain**，不宣称分布式强一致、绝对不可篡改或商业级不可抵赖。

详细说明见 [安全边界](docs/SECURITY_BOUNDARIES.md)。

## 可验证的工程证据

### 六项主分支 CI

```text
quality
unit
security-evaluation
secret-scan
dependency-scan
offline-integration
```

Unit 和 Offline Integration 的数字是测试数量，不是 Coverage 百分比。

### Retrieval 冻结基准

| 50-query benchmark | RRF Child | Cross-Encoder Child | 变化 |
| --- | ---: | ---: | ---: |
| Hit@1 | 84.78% | 93.48% | +8.70 percentage points |
| MRR@5 | 91.67% | 96.74% | +5.07 percentage points |
| Hit@5 | 100% | 100% | — |

数据集包含 46 个 answerable query 和 4 个 unanswerable query。这是合成文档上的冻结工程基准，不是生产准确率。

### M9 冻结系统运行

| Frozen-run observed metric | 结果 |
| --- | ---: |
| Formal Runtime | 8 / 9 |
| Deterministic Boundaries | 4 / 4 |
| Overall | 12 / 13 |
| Unanswerable | 2 / 3 |
| False Positive | 1 |
| Provider Calls | 45 |
| Input / Output Tokens | 38,549 / 4,258 |
| End-to-end P50 / P95 | 9.594 s / 24.250 s |

这些数字是小规模 frozen-run observed metrics，不是生产 SLA、生产成本或线上准确率。完整口径见 [评测说明](docs/EVALUATION.md)。

## 快速体验

### 方式 A：不需要 Provider，验证冻结证据

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

.\.venv\Scripts\python.exe scripts\verify_retrieval_evidence.py
.\.venv\Scripts\python.exe scripts\calculate_m9_metrics.py artifacts\evaluation\m9-final-eval-v1\case_records.jsonl --dataset datasets\agent_tasks\m9_final_eval_v1.json --adjudications artifacts\evaluation\m9-final-eval-v1\adjudications.json --output $env:TEMP\m9-metrics.json --manifest-output $env:TEMP\m9-manifest.json
```

### 方式 B：运行完整本地 Demo

```powershell
Copy-Item .env.example .env
docker compose config
docker compose up -d

.\.venv\Scripts\python.exe scripts\initialize_knowledge_corpus.py
.\.venv\Scripts\python.exe scripts\run_local_demo.py knowledge
.\.venv\Scripts\python.exe scripts\run_local_demo.py data
.\.venv\Scripts\python.exe scripts\run_local_demo.py mixed
```

如需打开截图中的 Web UI：

```powershell
.\.venv\Scripts\python.exe -m uvicorn decision_agent.main:app --host 127.0.0.1 --port 8000
```

然后访问 `http://127.0.0.1:8000`。

完整 Demo 需要用户在本地环境配置自己的兼容 Provider 凭据，可能产生 Provider 费用。仓库不包含可用凭据、模型权重、数据库 Volume 或运行时 Audit Log。详细步骤见 [本地 Demo](docs/LOCAL_DEMO.md)。

## 项目结构

```text
src/decision_agent/          Agent Runtime、Security、Retrieval、MCP、API 与 Skills
tests/                       Unit、Offline Integration 与可选服务测试
scripts/                     Demo、初始化、评测与离线验证入口
datasets/                    合成知识、经营数据与安全用例
docker/mysql/init/           合成 Schema、Seed 与只读用户初始化
artifacts/evaluation/        冻结且已脱敏的工程评测证据
artifacts/public-evaluation/ 公开证据的来源与 Hash 映射
docs/                        架构、工作流、安全、评测与运行文档
.github/                     CI 与 Dependabot 配置
```

## 深入阅读

| 文档 | 内容 |
| --- | --- |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统分层与端到端调用链 |
| [AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md) | Router、Planner、Skill、Reviewer 与 Release |
| [HYBRID_RAG.md](docs/HYBRID_RAG.md) | 检索、融合、重排、Parent Expansion 与评测 |
| [DATA_AGENT_AND_MCP.md](docs/DATA_AGENT_AND_MCP.md) | MCP、DataScope、SQL Guard 与只读 MySQL |
| [SECURITY_BOUNDARIES.md](docs/SECURITY_BOUNDARIES.md) | 身份、权限、Scope、Provider 与 Audit |
| [EVALUATION.md](docs/EVALUATION.md) | Retrieval、M9、测试和指标口径 |
| [LOCAL_DEMO.md](docs/LOCAL_DEMO.md) | 本地环境、服务启动与三类 Demo |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | 已知失败和能力边界 |

## 已知限制

冻结评测有意保留了唯一真实 false positive：`m9-knowledge-unanswerable-003`。一个不可回答的知识请求被错误地带引用发布。这是 Answerability / Evidence Sufficiency 失败，不是权限绕过或敏感数据泄漏。

## 项目边界

这是基于 v1.0.2 冻结版本构建的公开工程快照。Agent 核心 Runtime 和评测证据保持不变，公开仓库只进行文档、Demo 可用性和依赖安全维护。

当前项目没有授予项目级开源 License；源码公开可见不等于获得复制、修改或再分发授权。
