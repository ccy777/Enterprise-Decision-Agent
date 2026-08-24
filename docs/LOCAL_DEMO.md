# 本地运行指南

## 环境准备

- Python 3.11及以上；
- Docker与Docker Compose；
- 能够运行MySQL、Milvus、etcd和MinIO的本地环境；
- OpenAI兼容模型服务配置。

仓库提供合成知识文档、MySQL Schema和Seed数据，可以在本地运行并验证三类任务。

## 安装项目

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
Copy-Item .env.example .env
```

根据`.env.example`填写本地模型服务和基础设施配置，`.env`已加入忽略列表。

## 启动基础设施

```powershell
docker compose config
docker compose up -d
docker compose ps
```

Compose会启动MySQL和Milvus相关依赖。MySQL使用合成Schema与Seed数据初始化。

## 初始化知识库

MySQL、etcd、MinIO和Milvus状态正常后执行：

```powershell
.\.venv\Scripts\python.exe scripts\initialize_knowledge_corpus.py
```

该命令读取配置的数据集，完成文档解析、分块、Embedding生成和Milvus写入，并校验Child记录数量。重复执行会根据稳定记录ID更新已有数据。

检索证据可以单独复核：

```powershell
.\.venv\Scripts\python.exe scripts\verify_retrieval_v2_evidence.py
```

## 运行三类任务

```powershell
# 企业知识问答
.\.venv\Scripts\python.exe scripts\run_local_demo.py knowledge

# 经营数据分析
.\.venv\Scripts\python.exe scripts\run_local_demo.py data

# 知识与数据联合决策
.\.venv\Scripts\python.exe scripts\run_local_demo.py mixed
```

JSON 结果包含请求 ID、执行状态、路由、Skill、回答、Citation 和 Trace 摘要。启动 Web 分析工作台：

```powershell
.\.venv\Scripts\python.exe scripts\run_local_web_demo.py mixed
```

网页中保持同一对话，可依次提问“哪些产品低于安全库存？”“其中风险最高的是哪个？”“结合库存制度给出建议。”；点击“新建对话”后会生成新的会话标识，不继承上一段对话。

## 查看执行结果

重点关注：

- Router选择的任务类型；
- 实际执行的Skill和Tool；
- Knowledge Evidence与Data Evidence；
- 最终答案中的Citation；
- Trace中的阶段状态、耗时和错误码；
- Context / Memory状态变化。

## 离线验证

```powershell
.\.venv\Scripts\python.exe scripts\verify_retrieval_v2_evidence.py
.\.venv\Scripts\python.exe scripts\verify_retrieval_evidence.py
.\.venv\Scripts\python.exe scripts\calculate_m9_metrics.py artifacts\evaluation\m9-final-eval-v1\case_records.jsonl --dataset datasets\agent_tasks\m9_final_eval_v1.json --adjudications artifacts\evaluation\m9-final-eval-v1\adjudications.json --output $env:TEMP\m9-public-metrics.json --manifest-output $env:TEMP\m9-public-manifest.json
```

单独检查MCP与MySQL查询链：

```powershell
.\.venv\Scripts\python.exe scripts\run_safe_query_demo.py
```

## 停止服务

```powershell
docker compose down
```
