# 检索评测 v2 公开证据

该证据包用于独立复核简历中的200-query检索指标，不需要加载Embedding、Reranker或调用Provider。

## 核心结果

评测集包含200条合成企业场景查询，其中160条可回答、40条不可回答。排序指标使用160条可回答查询作为分母。

| Child指标 | RRF | Cross-Encoder |
| --- | ---: | ---: |
| Hit@1 | 137 / 160 = 85.62% | 154 / 160 = 96.25% |
| Hit@5 | 160 / 160 = 100.00% | 160 / 160 = 100.00% |
| MRR@5 | 146.7 / 160 = 91.69% | 157.0 / 160 = 98.12% |

## 复核命令

在仓库根目录执行：

```powershell
python scripts/verify_retrieval_v2_evidence.py
```

脚本会校验固定文件哈希、200 / 160 / 40数量、Evidence模式、Case顺序和排名字段，并从`ranking_records.jsonl`与`relevance_freeze.jsonl`重新计算Hit@1、Hit@5和MRR@5。

## 文件说明

- `relevance_freeze.jsonl`：固定查询和Child相关性标注；
- `ranking_records.jsonl`：200条查询的RRF与Cross-Encoder排名；
- `metrics.json`：完整指标、切片和覆盖率分析；
- `experiment_report.md`：实验过程与指标解释；
- `failure_cases.md`：Cross-Encoder Top1未命中案例；
- `relevance_audit.md`：相关性标注复核记录；
- `runtime.json`：模型版本、配置和离线运行信息；
- `manifest.json`：数量、身份和文件哈希。

## 指标说明

Hit@1表示第一条Child是相关Evidence。对于需要多条Evidence共同回答的问题，还需要结合Hit@5和Evidence覆盖率理解检索结果。
