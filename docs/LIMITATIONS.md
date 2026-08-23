# 项目状态与后续计划

## 已完成能力

- Knowledge、Data、Mixed三类任务路由；
- LangGraph工作流与多阶段Prompt；
- Hybrid RAG、Evidence Selection、Answerability和Citation；
- MCP数据智能体、SQL生成与MySQL查询；
- Redis / In-Memory会话记忆与滚动摘要；
- 请求级Trace、离线评测和GitHub CI；
- FastAPI接口、命令行Demo与网页演示界面。

## 当前评测

- Retrieval Benchmark v2：200条企业场景查询；
- Child Hit@1：85.62%提升至96.25%；
- MRR@5：91.69%提升至98.12%；
- 1,802项单元测试、235项稳定离线集成测试；
- 28 / 28项确定性边界评测通过。

系统任务集中保留了一个Answerability误判案例，相关记录用于继续优化Evidence充分性判断。

## 后续优化方向

- 扩充真实业务问法和多轮任务评测；
- 优化Answerability与Evidence覆盖率；
- 增加并发压测和性能监控；
- 完善分布式Trace与评测看板；
- 扩展新的MCP数据源和业务Skill。

## 使用说明

仓库用于项目展示、学习和评测复核。如需复制、修改或二次发布，请先联系作者。
