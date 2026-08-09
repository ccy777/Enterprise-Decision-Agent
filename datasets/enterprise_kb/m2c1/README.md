# M2C-1 企业知识库资料包

本目录描述虚构企业“华衡智能科技有限公司”的制度与运营规则，仅用于离线工程测试、检索评测蓝图和人工审查，不代表任何真实企业、个人或生产规则。

## 目录职责

- `entity_dictionary.json`：统一公司、部门、产品、配件、地区、客户等级、供应商与分类 ID。
- `business_fact_registry.json`：跨文档必须保持一致的固定业务事实及其条款引用。
- `document_manifest.json`：12 份 Markdown 文档的版本、责任部门、分类和密级登记。
- `query_blueprint.jsonl`：60 条自然中文问题蓝图、参考答案、相关条款及强负例。
- `documents/`：带稳定人工 Clause ID 的企业制度长文档。

Clause ID 是人工维护的稳定业务标签，不是检索运行时生成的 Chunk ID。M2C-1A 禁止手工填写 Parent ID 或 Child ID；这些 ID 及最终 Ground Truth 将在 M2C-1B 使用真实 Markdown Parser 和 `ParentChildChunker` 生成。本资料包不能作为生产企业制度直接使用。

## 规范源格式与多格式能力

正式 Benchmark 的 12 份文档统一以 Markdown 作为规范源格式，以便稳定维护 Clause ID、字符 offset、版本差异和后续 Ground Truth。统一源格式不表示项目只支持 Markdown：`tests/fixtures/ingestion/mixed_format/` 使用三份内容不同的静态 TXT、Markdown 和 PDF，验证真实 `ParserRegistry` 能将三种格式转换为统一 `DocumentBlock`，并继续进入 `ParentChildChunker`。这些 Fixture 不登记到 Manifest、事实注册表或 60 Query，也不参与检索指标，避免重复内容污染召回结果。

## M7.1A 企业画像与能力边界

`DOC-ORG-001` 为企业画像的权威来源，说明华衡智能科技有限公司是项目中的虚构演示企业，明确智能设备制造与运营、产品 A/B 与原装电池、采购/库存/销售履约/售后替换场景，以及知识库和经营数据覆盖边界。`DOC-AGENT-001` 为企业 Agent 能力边界的权威来源，说明 Knowledge、Data、Mixed 三类受证据约束的能力，及不支持通用互联网搜索、任意闲聊、长期用户画像和业务写操作等边界。

两份文档与新增的 10 条 `enterprise_profile` 查询蓝图均进入同一套真实 Markdown Parser、`ParentChildChunker` 和既有 fixed-window-v1 Retrieval Pipeline。`generated/` 只能通过 `scripts/build_m2c1_parent_child_ground_truth.py` 刷新；不得手工编辑 Parent/Child 内容或用资料概览替代具体制度、数据证据和引用。

## Parent/Child Ground Truth

Clause ID 继续作为人工稳定语义标签；`generated/` 中的 Parent/Child ID 和四个 JSONL 文件
由真实 Markdown Parser 与 `ParentChildChunker` 生成，禁止手工修改。Clause 与 Chunk 使用
正文半开区间的正向字符重叠建立映射。Query 的 relevant 与 hard-negative 集合分别稳定聚合；
同时覆盖两类 Clause 的 Chunk 会从纯 hard-negative 集合移出并显式记录为 overlap，不会被
静默丢弃。当前字符窗口分块会产生跨 Clause 的混合 Chunk，因此 Parent 级纯强负例分析能力
受到限制；该碰撞保留为可审查的真实结果。本阶段只生成 Ground Truth，不产生 HitRate、Recall 或 MRR；下一阶段 M2C-2 使用
这些标签运行完整 Retrieval Pipeline。
