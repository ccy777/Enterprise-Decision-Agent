# MCP数据智能体与工具调用

## 完整执行链

```text
Data Skill
  -> Native Tool Calling
  -> run_data_agent
  -> MCP获取Schema与业务定义
  -> Data Planner生成SQL计划
  -> MCP执行查询
  -> SQLGlot / SQL Guard
  -> SQLAlchemy
  -> MySQL
  -> Data Evidence与分析结果
```

Native Tool Calling负责选择并执行`run_data_agent`。进入Data Agent后，MCP提供企业Schema、业务定义和查询执行工具，Data Planner据此把自然语言问题转换为结构化查询计划。

## MCP工具

MCP Server公开三个工具：

| Tool | 作用 |
| --- | --- |
| `get_enterprise_schema` | 获取允许查询的业务表和字段 |
| `get_business_definitions` | 获取销量、库存、采购和交付等业务口径 |
| `execute_safe_query` | 执行SQL并返回结构化查询结果 |

MCP将Agent工作流与数据库访问解耦，Schema、查询参数和返回结果都使用明确的数据契约。

## 数据查询规划

Data Planner根据用户问题、Schema和业务定义输出：

- 查询状态；
- 业务意图；
- SQL；
- 计划理由；
- 需要补充的信息。

Pydantic负责校验计划结构。计划就绪后，Data Agent调用MCP执行SQL，再将查询结果组织为Data Evidence和带引用的回答。

## 查询规则

SQLGlot与SQL Guard负责解析MySQL语法，并检查单条查询、只读操作、业务表字段、LIMIT、执行超时和结果规模。DataScope用于限定当前请求可访问的业务资源。

## 本地数据与运行方式

`docker/mysql/init/01-schema.sql`和`02-seed.sql`提供产品、供应商、销售订单、库存快照和采购订单等合成数据。`03-create-readonly-user.sh`创建本地查询账号。MCP客户端和服务端由运行时统一启动与关闭，离线集成测试使用确定性的进程和服务替代实现。
