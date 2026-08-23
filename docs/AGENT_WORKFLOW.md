# 智能体工作流

## 整体流程

系统采用 Router、Planner、Skill、Tool 和 Reviewer 分工协作的执行方式。模型负责理解问题、生成计划和组织回答，服务端负责流程编排、工具注册、上下文装配、结果校验与状态记录。

```text
请求校验
  -> 加载会话上下文
  -> Router 判断任务类型
  -> Coordinator 选择 Skill
  -> Planner 生成结构化计划
  -> Skill 与 Tool 执行
  -> Evidence 选择与答案生成
  -> Reviewer 检查结果和引用
  -> 返回响应并记录 Trace
```

## 三类路由与 Skill

| 路由 | Skill | 处理内容 |
| --- | --- | --- |
| Knowledge | `enterprise-knowledge-qa` | 企业文档检索、证据选择与引用回答 |
| Data | `enterprise-data-analysis` | 经营数据查询与分析 |
| Mixed | `inventory-risk-diagnosis` | 联合知识规则与经营数据生成库存风险建议 |

Knowledge和Data走各自的单一Skill；Mixed会组合知识与数据子任务，再将两类Evidence交给综合与审查阶段。

## 提示词与结构化输出

项目为不同阶段分别设计Prompt，并使用Pydantic校验结构化结果：

1. `routing`：识别Knowledge、Data或Mixed任务；
2. `planning`：生成工作流执行计划；
3. `data_planning`：结合Schema生成数据查询计划和SQL；
4. `evidence_selection`：从检索结果中选择回答所需证据；
5. `answerability_review`：判断现有证据是否足够；
6. `knowledge_answer`与`data_answer`：生成带引用的答案；
7. `inventory_synthesis`：综合库存数据和补货规则；
8. `workflow_review`：检查最终结果、引用和执行状态。

不同路由只调用当前任务需要的阶段，不会固定执行全部流程。

## 异常处理

路由、计划、工具调用、检索和Reviewer都返回明确状态与错误码。执行失败时由工作流进入对应异常分支，并将阶段状态写入Trace，便于通过同一请求ID定位问题。
