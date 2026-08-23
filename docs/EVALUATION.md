# 项目评测

## 评测组成

项目使用三类评测验证不同层次的能力：

1. 单元测试验证模型、服务和工具契约；
2. 离线集成测试验证完整工作流；
3. 检索Benchmark与系统任务集验证检索和Agent执行结果。

## 公开验证结果

| 验证项 | 结果 |
| --- | ---: |
| 单元测试 | 1,802项通过 |
| 稳定离线集成测试 | 235项通过 |
| 确定性边界评测 | 28 / 28通过 |

GitHub Actions在Pull Request和`main`分支运行`quality`、`unit`、`security-evaluation`、`secret-scan`、`dependency-scan`与`offline-integration`六项检查。

## 检索评测 v2

Retrieval Benchmark v2包含200条合成企业场景查询，其中160条可回答、40条不可回答。Hit@K和MRR使用160条可回答查询作为分母。

| Child指标 | RRF | Cross-Encoder |
| --- | ---: | ---: |
| Hit@1 | 137 / 160 = 85.62% | 154 / 160 = 96.25% |
| MRR@5 | 146.7 / 160 = 91.69% | 157.0 / 160 = 98.12% |
| Hit@5 | 160 / 160 = 100% | 160 / 160 = 100% |

Cross-Encoder使Child Hit@1提升10.63个百分点，MRR@5提升6.44个百分点。`scripts/verify_retrieval_v2_evidence.py`会校验固定记录的哈希，并重新计算公开指标。

原50-query v1结果保留为历史基线：RRF Hit@1为84.78%，Cross-Encoder Hit@1为93.48%，RRF MRR@5为91.67%，Cross-Encoder MRR@5为96.74%。

## 系统任务评测

| 指标 | 结果 |
| --- | ---: |
| Formal Runtime | 8 / 9 |
| Deterministic Boundaries | 4 / 4 |
| Overall | 12 / 13 |
| Unanswerable | 2 / 3 |
| Provider调用 | 45 |
| 输入 / 输出Token | 38,549 / 4,258 |
| E2E P50 / P95 | 9.594s / 24.250s |

任务集中保留了一个Answerability误判案例，用于后续优化Evidence充分性判断和拒答策略。

## 复核方式

Retrieval v1/v2与M9指标计算均支持离线运行。公开复核不需要重新加载Embedding、Reranker或调用Provider，具体命令见README和`artifacts/public-evaluation/`。
