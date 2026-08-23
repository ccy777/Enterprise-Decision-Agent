# 混合检索（Hybrid RAG）

## 检索链路

```text
Dense Retrieval + BM25
  -> KnowledgeScope过滤
  -> RRF融合（k=60）
  -> Cross-Encoder精排
  -> Parent Expansion
  -> Evidence上下文构建
  -> Evidence Selection
  -> Answerability Review
  -> 带引用的回答
```

语料采用版本化Parent/Child分块。Dense检索使用`BAAI/bge-small-zh-v1.5`生成512维归一化向量，BM25负责关键词召回；RRF融合两路Child排名，`BAAI/bge-reranker-base`对候选结果精排，Parent Expansion补充相邻上下文。

默认配置为Dense Top 10、BM25 Top 10、RRF Top 10、Reranker Top 5、Parent Top 5，最终Evidence最多5条、总文本最多6,000字符。

## 检索评测 v2

当前公开Benchmark包含200条合成企业场景查询：

| 数据组成 | 数量 |
| --- | ---: |
| Queries | 200 |
| Answerable | 160 |
| Unanswerable | 40 |
| Documents | 12 |
| Parent records | 36 |
| Child evidence windows | 101 |

| Child指标 | RRF | Cross-Encoder | 提升 |
| --- | ---: | ---: | ---: |
| Hit@1 | 85.62% | 96.25% | +10.63个百分点 |
| MRR@5 | 91.69% | 98.12% | +6.44个百分点 |
| Hit@5 | 100% | 100% | — |

排序指标以160条可回答查询为分母。96.25%对应Cross-Encoder阶段的Child Hit@1，用于衡量首条检索结果命中情况。

## 证据复核

运行以下命令即可校验固定SHA-256、数据集数量和Evidence模式，并从相关性标注与排名记录重新计算Hit@1、Hit@5和MRR@5：

```powershell
python scripts/verify_retrieval_v2_evidence.py
```

## 证据构建

KnowledgeScope在结果融合前完成文档范围过滤。精排后的Child通过Parent Expansion恢复上下文，再由Evidence Selector选出用于回答的证据，最终引用与文档ID、版本和来源保持关联。
