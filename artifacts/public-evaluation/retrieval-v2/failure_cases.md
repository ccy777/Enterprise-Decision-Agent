# Retrieval Benchmark v2 — Final Failure Cases

All correctness decisions use the blind-frozen `adjudicated_relevant_child_ids`.

## Movement summary

- RRF wrong → Reranker correct: 20
- RRF correct → Reranker wrong: 3
- Unchanged correct: 134
- Unchanged wrong: 3
- Final Reranker Top1 misses: 6
- Pattern counts: `{'ranking ambiguity': 2, 'window-boundary issue': 4}`

## Detailed final Reranker Top1 misses

### M2C1-Q006

- Query: 产品 A 使用非原厂充电设备后损坏，是否属于基础保修？
- Type / mode: `multi_constraint` / `single_window_sufficient`
- Relevant Children: `child_adfd58e8fe3b19d7f2a226bf758d5ca657d87e88ebf82dc4a951d840c8fb669e`
- RRF Top5: `1:child_30d69939410d5911ac2e11f1b35e4e02120b324eac268a6119c247f108cfc412[DOC-CS-001], 2:child_adfd58e8fe3b19d7f2a226bf758d5ca657d87e88ebf82dc4a951d840c8fb669e[DOC-CS-001], 3:child_926711a50bc164c70b2bf00a6381fc917c6e0d2cd2e5bb1450e9e4b617c4fcb7[DOC-CS-001], 4:child_5b9119374263bddc6ca8c517b151f885ab51047885dc80e4bec2ab918c7acf14[DOC-CS-001], 5:child_a3e67116cd32664d76a565a377211d029e5acf98ac2b03becbaeeed9c5df4152[DOC-CS-002]`
- Reranker Top5: `1:child_926711a50bc164c70b2bf00a6381fc917c6e0d2cd2e5bb1450e9e4b617c4fcb7[DOC-CS-001], 2:child_adfd58e8fe3b19d7f2a226bf758d5ca657d87e88ebf82dc4a951d840c8fb669e[DOC-CS-001], 3:child_30d69939410d5911ac2e11f1b35e4e02120b324eac268a6119c247f108cfc412[DOC-CS-001], 4:child_5b9119374263bddc6ca8c517b151f885ab51047885dc80e4bec2ab918c7acf14[DOC-CS-001], 5:child_63c8476cc32b8ff4835aaadd92e0655cb325c045607bcfc94b3b2bd9501fce37[DOC-CS-002]`
- Diagnosis: **ranking ambiguity** — Top1 is a different clause/window in the correct document; intra-document policy similarity dominates.

### M2C1-Q022

- Query: 产品 B 可用量为 70 台时，安全线和紧急线分别意味着什么？
- Type / mode: `multi_constraint` / `multi_evidence_required`
- Relevant Children: `child_9f8677e667e3e63fa30462a8e182a393fd0d87d5249441402254e60ce6f1485d, child_31c696e938321f353b25de4372e737bf1387d213151eaab78341637e891e603b`
- RRF Top5: `1:child_9f8677e667e3e63fa30462a8e182a393fd0d87d5249441402254e60ce6f1485d[DOC-INV-001], 2:child_c6e9a327e34acbda5db4704a19155aab9ee55ec418bd275b74c8562e0cfaabd6[DOC-INV-001], 3:child_31c696e938321f353b25de4372e737bf1387d213151eaab78341637e891e603b[DOC-INV-001], 4:child_588a6ad82d0ed0412bd4aad296083c8b177f37bd43b6061d1767f388d8d1da95[DOC-INV-001], 5:child_277e0c36787d2673c2cf933a408012505443bbaa5f2f80e9aceca2fdaf780440[DOC-INV-001]`
- Reranker Top5: `1:child_c6e9a327e34acbda5db4704a19155aab9ee55ec418bd275b74c8562e0cfaabd6[DOC-INV-001], 2:child_9f8677e667e3e63fa30462a8e182a393fd0d87d5249441402254e60ce6f1485d[DOC-INV-001], 3:child_31c696e938321f353b25de4372e737bf1387d213151eaab78341637e891e603b[DOC-INV-001], 4:child_feb15cb4d3980a91c2c0b4cb9da317a706ceb9b8d63bd24b6cbbbc612e02d7d3[DOC-INV-001], 5:child_277e0c36787d2673c2cf933a408012505443bbaa5f2f80e9aceca2fdaf780440[DOC-INV-001]`
- Diagnosis: **ranking ambiguity** — Top1 is a different clause/window in the correct document; intra-document policy similarity dominates.

### M2C1-Q034

- Query: 预算超支 8% 需要哪些人审批？
- Type / mode: `single_fact` / `single_window_sufficient`
- Relevant Children: `child_6d84db62631e8f53ac76494b189c30e1364e9099c86fc6f9bbdd05c214a3447b`
- RRF Top5: `1:child_6d84db62631e8f53ac76494b189c30e1364e9099c86fc6f9bbdd05c214a3447b[DOC-FIN-001], 2:child_998df9cb28390fc1f7e4b9f8b0575020e0be29fc89675db4b7c9478ccbb6e233[DOC-FIN-001], 3:child_c54f77cfceecbd7ba626983085aa76fd70404baf1a219b6c3a6683f8da036944[DOC-FIN-001], 4:child_73afcd2517b1eb22d810639f73543c0f8655cc41f3a314866620ea1f2395dca3[DOC-PROC-001], 5:child_abf5b07c7f017d471726319539bc4ab313af894cdabc9e69afbee0c95e1353dd[DOC-FIN-001]`
- Reranker Top5: `1:child_998df9cb28390fc1f7e4b9f8b0575020e0be29fc89675db4b7c9478ccbb6e233[DOC-FIN-001], 2:child_6d84db62631e8f53ac76494b189c30e1364e9099c86fc6f9bbdd05c214a3447b[DOC-FIN-001], 3:child_73afcd2517b1eb22d810639f73543c0f8655cc41f3a314866620ea1f2395dca3[DOC-PROC-001], 4:child_abf5b07c7f017d471726319539bc4ab313af894cdabc9e69afbee0c95e1353dd[DOC-FIN-001], 5:child_c54f77cfceecbd7ba626983085aa76fd70404baf1a219b6c3a6683f8da036944[DOC-FIN-001]`
- Diagnosis: **window-boundary issue** — Top1 is a sibling window in a relevant Parent but does not meet the frozen evidence rule for this query.

### RBV2-N022

- Query: 华东区普通客户的常规基础折扣上限是多少？
- Type / mode: `entity_numeric_policy_fact` / `single_window_sufficient`
- Relevant Children: `child_1c6cef2b2e0f80cafc312e17ccb7af994e21a666d05f95c968fae41918ffa553`
- RRF Top5: `1:child_dbabd97ac74144001ee37a21e26c4cff45f85ed750d6127777e7a1c6b1ad0708[DOC-SALES-001], 2:child_80f5496075bed0b62b07fc0e674a97dcd3b00204693d240929c3bb355a2ffeae[DOC-SALES-001], 3:child_1c6cef2b2e0f80cafc312e17ccb7af994e21a666d05f95c968fae41918ffa553[DOC-SALES-001], 4:child_2fe69363a6cf3956b7d37a384605d905d3317b36945bc57b83131914c102d105[DOC-SALES-001], 5:child_870ab642b2faccd4bcc5795bcb2d662431b532a6dc9bb25ae93010cb00a12c11[DOC-SALES-002]`
- Reranker Top5: `1:child_dbabd97ac74144001ee37a21e26c4cff45f85ed750d6127777e7a1c6b1ad0708[DOC-SALES-001], 2:child_1c6cef2b2e0f80cafc312e17ccb7af994e21a666d05f95c968fae41918ffa553[DOC-SALES-001], 3:child_870ab642b2faccd4bcc5795bcb2d662431b532a6dc9bb25ae93010cb00a12c11[DOC-SALES-002], 4:child_80f5496075bed0b62b07fc0e674a97dcd3b00204693d240929c3bb355a2ffeae[DOC-SALES-001], 5:child_2fe69363a6cf3956b7d37a384605d905d3317b36945bc57b83131914c102d105[DOC-SALES-001]`
- Diagnosis: **window-boundary issue** — Top1 is a sibling window in a relevant Parent but does not meet the frozen evidence rule for this query.

### RBV2-N038

- Query: 集团统一采购临时共享价格需要谁批准，会自动把关联公司升为战略客户吗？
- Type / mode: `hard_distractor` / `single_window_sufficient`
- Relevant Children: `child_b8e3363ea7c319add9ab1e6cf4facd80f3cd4546ddae08ee13ce6a994d461f43`
- RRF Top5: `1:child_87a9a193463a4aa02aa3d9fa2badac22d49414e706b69e44a6bda18fddde9bec[DOC-SALES-002], 2:child_2fe69363a6cf3956b7d37a384605d905d3317b36945bc57b83131914c102d105[DOC-SALES-001], 3:child_b8e3363ea7c319add9ab1e6cf4facd80f3cd4546ddae08ee13ce6a994d461f43[DOC-SALES-002], 4:child_6c54f64b4a3acdaa0b45400ac9b07558d00e529b3a73bdbc9d839fdefdc62b4c[DOC-SALES-002], 5:child_9e4ca864745af0d5270721987d3fe2b87af47f6e447756024d331786581302b6[DOC-SALES-002]`
- Reranker Top5: `1:child_87a9a193463a4aa02aa3d9fa2badac22d49414e706b69e44a6bda18fddde9bec[DOC-SALES-002], 2:child_b8e3363ea7c319add9ab1e6cf4facd80f3cd4546ddae08ee13ce6a994d461f43[DOC-SALES-002], 3:child_2fe69363a6cf3956b7d37a384605d905d3317b36945bc57b83131914c102d105[DOC-SALES-001], 4:child_6c54f64b4a3acdaa0b45400ac9b07558d00e529b3a73bdbc9d839fdefdc62b4c[DOC-SALES-002], 5:child_4608b2a1bf0364e7b1888ae3868d348a63f4a8ea36f7bdf4acf73c4e642240fc[DOC-SALES-001]`
- Diagnosis: **window-boundary issue** — Top1 is a sibling window in a relevant Parent but does not meet the frozen evidence rule for this query.

### RBV2-N098

- Query: ‘华衡科技’‘本公司’和‘示例企业’这些称呼在当前资料中统一指向什么？
- Type / mode: `exact_keyword` / `single_window_sufficient`
- Relevant Children: `child_ce98419eebc79f068a359797b9374f584e0c37a48be8dce2bf7d30de763dbb1d`
- RRF Top5: `1:child_ce98419eebc79f068a359797b9374f584e0c37a48be8dce2bf7d30de763dbb1d[DOC-ORG-001], 2:child_4f6d1f1fd8f554a200c09e0193726d0e1c0d65dff904aba528087ddec88e8a78[DOC-ORG-001], 3:child_974a917dbe7975167036c28ca320d7e027666f8a75d82f8dfb344d1994e6e342[DOC-AGENT-001], 4:child_eb3432a80541008456a5ba69c74540ab03f6ebcd4f6f9d8914d245eb87fb01f0[DOC-AGENT-001], 5:child_fc65c6f53f725078fcbeb1ce4e4aeecdde77b100fa45d6dfe70aaf7a5b78f67f[DOC-SEC-001]`
- Reranker Top5: `1:child_4f6d1f1fd8f554a200c09e0193726d0e1c0d65dff904aba528087ddec88e8a78[DOC-ORG-001], 2:child_ce98419eebc79f068a359797b9374f584e0c37a48be8dce2bf7d30de763dbb1d[DOC-ORG-001], 3:child_eb3432a80541008456a5ba69c74540ab03f6ebcd4f6f9d8914d245eb87fb01f0[DOC-AGENT-001], 4:child_974a917dbe7975167036c28ca320d7e027666f8a75d82f8dfb344d1994e6e342[DOC-AGENT-001], 5:child_fc65c6f53f725078fcbeb1ce4e4aeecdde77b100fa45d6dfe70aaf7a5b78f67f[DOC-SEC-001]`
- Diagnosis: **window-boundary issue** — Top1 is a sibling window in a relevant Parent but does not meet the frozen evidence rule for this query.
