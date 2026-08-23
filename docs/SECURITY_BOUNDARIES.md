# 数据访问与结果校验

## 请求上下文

每次请求都携带`RequestPrincipal`和`SecurityContext`，其中包括租户、会话、策略版本以及可选的`DataScope`和`KnowledgeScope`。这些信息用于确定当前请求能够使用的Skill、Tool、知识文档和业务数据。

## 分层校验

```text
身份与会话
  -> 场景和路由
  -> Workflow / Skill / Tool
  -> KnowledgeScope / DataScope
  -> Provider输入处理
  -> Reviewer与Citation
  -> Response与AuditEvent
```

KnowledgeScope在RRF和Reranker之前过滤文档；DataScope在MCP客户端创建和工具结果返回时校验数据资源。Provider输入会进行字段分类、上下文预算和递归脱敏处理。

## 查询与回答

数据查询经过MCP、SQLGlot、SQL Guard、表字段范围、LIMIT和执行超时。知识回答经过Evidence Selection、Answerability Review和Citation Validation。Reviewer负责检查最终工作流状态、答案和引用。

## 链路追踪与审计

Trace通过request ID和trace ID关联路由、规划、Skill、Tool Calling、检索与Reviewer阶段。`AuditEvent`记录关键执行事件，并使用JSONL追加写和哈希链支持本地复核。

## 评测结果

确定性评测集包含28个用例，覆盖数据范围、知识范围、Provider输入处理、工具调用和结果返回等场景，当前结果为28 / 28通过。
