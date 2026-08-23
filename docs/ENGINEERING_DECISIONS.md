# 关键技术选型

## 显式工作流编排

项目使用Router、Planner、Skill、Tool和Reviewer组成显式执行链。LangGraph负责状态流转与条件分支，Pydantic负责阶段契约，使Knowledge、Data和Mixed三条链路能够独立测试和定位问题。

## 混合检索

Dense检索擅长语义匹配，BM25擅长关键词和业务术语匹配。RRF用于融合两路排名，Cross-Encoder进一步优化前排结果，Parent Expansion补回回答所需上下文。这套组合兼顾召回、排序和最终Evidence完整性。

## MCP工具层

Data Agent通过MCP获取Schema、业务定义和查询能力。MCP把Agent流程、SQL校验与数据库执行拆成清晰模块，方便替换数据源、测试工具契约和扩展新的经营数据能力。

## 上下文与会话记忆

会话状态支持Redis与In-Memory两种实现，按租户、用户和会话隔离。TTL、滚动摘要和上下文预算用于控制多轮任务输入规模，并保留当前问题所需的历史信息。

## 链路追踪与离线评测

请求级Trace覆盖Routing、Planning、Skill、Tool Calling、Retrieval、Evidence和Reviewer。离线评测分别验证检索排序、完整工作流和异常场景，便于在修改Prompt、检索参数或工具实现后进行回归。

## 可复核评测证据

Retrieval Benchmark v2保存查询集、相关性标注、排名记录、指标结果与SHA-256校验值。复核脚本可以直接从仓库中的固定记录重新计算Hit@1、Hit@5和MRR@5。
