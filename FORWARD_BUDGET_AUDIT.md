# v11：原目标前向支持集候选与全局预算诊断

版本 `gpu-eval-v11-forward-budget-audit`；独立输出 `results/gpu-eval-v11-forward-budget-audit/`。项目仍是 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究**，保留 BMAD 六数据集与 normal memory exact cosine 1-NN。本文件是开发交接；实验数据、结果、图表与分析报告按 AGENTS 第 28 节仅留本地。

## 授权与协议

延续用户此前“能否换一种性能更好的稀疏方式？再改进一下”的授权，在**同一 U-VaRFS 目标**下改进支持集求解候选。ASLS patch geometry、sigmoid/Adam/loss、组合搜索和层预算保持；不是 Hard-Concrete、监督选择、PCA 替换或新 detector。

当前 Main 输入继续为 `representation.layer_normalization=none`，完整 `layer_l2_*` 消融保留。将 L2 输入固定为 Main 的协议切换仍按 AGENTS 第 19 节等待明确批准；不能按返回 test AUROC 在 raw/L2 或 ASLS/全层之间挑选 Main。

原 beta/lambda grid、box `[0,1]`、geometry tolerance、32–256 维预算、normal fit/variability/memory 图像和 patch 预算、扰动定义、五 seeds、Top-1%、pixel 评价及 Frozen DINOv3 全层候选均保持。所有新 U-VaRFS 方法采用相同候选流程与 refit 迭代上限；旧算法只作明确对照。

## 为什么增加另一个求解起点

已有流程先求完整连续权重，再通过删除和局部交换得到预算内支持集；固定支持集重优化无法恢复被遗漏的维度，单坐标交换也不保证跳出局部支持集。原目标/训练几何改善不能自动解释为异常性能收益。

v11 增加从零权重构建的独立候选。记 `A=H+2*beta*rscale*P*P^T`，原梯度为 `g=A*w-H*1+lambda*1`。对遗漏维度 j 加入系数 t 的精确目标变化为 `g_j*t + 0.5*A_jj*t^2`，在 `[0,1]` 上的最优 t 为 `clip(-g_j/A_jj,0,1)`。每步加入精确目标下降最多的一维，最多使用既定 max_features；达到无有效下降时停止。候选中的现有系数在插入阶段保持，随后重优化固定支持集。

插入/refit 均使用原 H、P/rscale、beta/lambda，以及原线性项 `(H*1)_S`，不重新定义选中维度的 Gram 目标。增量梯度每 64 步完整重算；最后以原完整目标和几何重新评估。

每个 lambda 同时比较 **v10 exchange/refit 候选、前向候选、前向/refit 候选**。只有目标与原 weighted-Gram 误差均不变差，并保持原最小非零约束的候选才可接受。最终仍选满足容差的最稀疏 path 点；无可行点时仍取 geometry error 最小者，明确 `feasible=false`。没有用异常标签/masks或 normal cosine 诊断另设选择准则。

`forward_candidate_insertions` 是独立候选的构建次数，不代表最终 Main 接受了该候选；应同时读取 `forward_candidate_accepted` 和 `forward_source`。固定支持 gap 仅在实际采用已 refit 的前向候选时报告，不能借用旧支持集证书。该启发式不保证全局最佳支持集或异常性能提升。

## 全部支持集的预算必要条件

新增 `budget_geometry_audit`，只读且 `selection_candidate=false`。设 `b=H*1`、`B=1^T H 1`，把参考 Gram 作为投影方向可得：

```text
geometry_error(w) >= 1 - b^T w / B
                  >= max(0, 1 - sum(topK(b)) / B)
```

第二个下界覆盖**所有至多 K 个非零维度、权重位于 `[0,1]` 的支持集**，不同于已有 fixed_support_geometry_floor。保存每维参考 Gram 对齐份额，以及达到既定容差所需的必要维数下界。下界超过容差加报告 margin，说明拟合正常几何在当前预算下可被排除；下界未超过容差不能证明存在可行解，必要维数也不是充分预算建议。

使用 fitted H 的 float64 reductions，并以 1e-6 margin 报告工作精度结论，不声称区间算术证明。零参考 Gram 单独标记，不能据此宣称正常结构有意义。诊断不改变候选、lambda、beta、维度预算或几何容差。weighted Gram 与 detector 再归一化 cosine 仍须分别报告；后者误差小不等于原目标可行。

每个 U-VaRFS JSON 保存诊断、输入归一化模式和按源层汇总的正常参考对齐量；CSV 保存全局下界、必要维数、预算排除状态及前向候选来源。只有下一轮真实 fit 保存的新诊断才能用于该数据集的全局判断，不能用旧支持集下界冒充全局下界。

## 同轮对照、断点与分析

保留原 37 方法，新增 `asls_exchange_refit_uvarfs` 精确保留前向升级前的同层结果，每输入分支 38 方法，默认 **76 方法**、全六 456 行。旧 Top-weight、prune/refit、prefix、legacy_main、Raw、PCA、Random 与全层 U-VaRFS 都保留；新候选的 nested exchange_refit 输出可与单独旧策略逐位比较。

raw/L2 分支继续共享正常图像、patches、扰动、DINO forward 和 memory 抽样；variance/fit/memory/query 在分支内统一输入。新算法的支持集与维度由正常数据决定，PCA/Random 和 Random-K 按各分支 Main 的维度/K 匹配，统一上限保持。

完整候选构建/refit、全局诊断和 normal cosine audit 均计入整套方法的 fit 运行开销。嵌套旧候选的中间时间不能作为独立运行效率；所有方法的共享总时间和峰值显存不能重复分配或宣称单方法加速。

分析脚本增加每个输入分支对**自己的 baseline**的配对 image bootstrap，记录 target_method，避免将跨输入改善全部归因于 U-VaRFS。逐图 Image AUROC/AP、五 seeds mean/sample SD、正常清单/预算、原目标与逐 lambda 保护、全局下界派生量均独立核查。图像区间没有患者/slide 分组或非劣界值，不能宣称患者级显著性或非劣。

版本/配置/代码/数据/方法完整性限制仍适用，v10 及历史结果不能 resume 到 v11。旧结果不覆盖；主输入切换也不能续接当前 raw Main 的目录。

## 第 19 节检查

| 项目 | 结论 |
| --- | --- |
| 研究问题 | 不变，BMAD 六数据集与两阶段稀疏选择 |
| ASLS 定义 | 不变，原 loss/gates/patch/search/预算 |
| U-VaRFS 目标 | 不变，新增原目标支持集求解起点，沿用用户稀疏升级授权 |
| 异常 labels/masks 拟合 | 无 |
| detector | 不变，cosine 1-NN、Top-1%、bilinear map |
| main/baseline 数据预算 | 完全相同；旧算法是明确的计算策略对照 |
| 新旧比较 | 使用 v11 同轮旧策略控制；不得把跨版本变化归因于单一算法 |
| 版本 | bump v11，独立目录；L2 主输入升级仍待批准 |

## 运行与验证

92 项本地检查通过，3 项 CUDA 检查因本机无 runtime 跳过；shell/diff 检查通过。覆盖样本空间原目标的独立 scalar oracle、全部小型支持集的下界检查、同列例子的紧下界与 cosine/weighted Gram 差异、旧策略精确复现、每 lambda 保护、诊断篡改拒绝，以及两个 primary 配置的 76 方法 fit → memory → image/pixel 流程。本机没有真实 BMAD、模型与 CUDA，新版真实性能和服务器开销仍须验证。

服务器先 Liver（CT 调试数据集），再 BMAD 全六：

```bash
git pull --ff-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

核查 76 方法、forward/source 接受状态、全局/固定支持/continuous 三类证书、两分支 manifest、memory 与 pixel 指标。预算被下界排除时如实报告，不自动增维、放宽容差或改损失。
