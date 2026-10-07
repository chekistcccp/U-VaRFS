# v9：原目标下支持集交换与正常余弦几何诊断

版本 `gpu-eval-v9-support-exchange-audit`，独立输出 `results/gpu-eval-v9-support-exchange-audit/`。本轮延续用户已授权的稀疏方式改进；实验回传、分析、图表与日志只保留本地，本文仅记录实现与运行协议。

## 修改范围检查

| 强制检查 | 本轮答案 |
| --- | --- |
| 改变研究问题？ | 否；BMAD 全六、Frozen DINOv3、ASLS、U-VaRFS、normal memory |
| 改变 ASLS 定义？ | 否；patch 输入、等权单位层 Gram、sigmoid/Adam/loss、组合搜索及原预算/容差保持 |
| 改变 U-VaRFS 数学目标？ | 否；H、P/rscale、beta、lambda path 保持；改进已授权的支持集求解 |
| 引入异常标签或 masks 拟合？ | 否；交换与诊断只读取 normal fit patches 和 variability |
| 改变主 detector？ | 否；exact cosine 1-NN、Top-1%、bilinear map 保持 |
| 给 Main 更多数据/表示预算？ | 否；所有新版 U-VaRFS 使用同一交换/refit 配置与抽样；旧求解器仅作明确消融 |
| 与旧结果完全直接可比？ | 否；支持集求解和扰动随机流实现发生变化，新增同轮 v8 控制 |
| 需要 bump version？ | 是；v9 禁止 resume/覆盖任何 v8 及更早产物 |

不根据 test AUROC 回退层规则、修改 beta/lambda 或容差，也不改变特征归一化和主目标。新增 solver 不保证异常指标提升。

ASLS 额外保存均匀 gates 方向的解析诊断：令 p=t·1，原 loss 为 `(1-t) + gamma mean(v) + rho t`，斜率是 `rho-1`。该字段解释全开方向的优化趋势，仅为诊断；不据此改 rho、gate loss 或选择规则，也不将离散压缩当成学得稀疏 gates 的证明。

## 精确交换损失

令 `A = H + 2 beta rscale P P^T`，`g = A w - H1 + lambda`。对当前支持中的维度 i（权重 a），和支持外维度 j（当前权重 0），将 i 删除、j 赋权 b：

```text
delta F = 0.5 A_ii a^2 - g_i a
        + (g_j - A_ji a) b + 0.5 A_jj b^2
b* = clip((A_ji a - g_j) / A_jj, 0, 1)
```

这是原完整目标的精确两坐标变化，保留遗漏维度的交叉项。使用 H 的同式计算 representation term 变化，只选择原目标下降超过数值阈值、representation term 不增加的交换。每个 lambda 独立搜索，支持大小保持；候选分块，R 使用低秩乘法，不构造大 R，不使用随机候选或异常监督。

默认最多 8 次交换，chunk=256，最小原目标改进 1e-8，refit 上限仍为 2000。数值计算预算对所有新版 U-VaRFS 一致，不按数据集/检测结果设置不同参数。搜索为有限次数的局部启发式：不宣称全局最佳稀疏支持集或严格 L0 最优。

refit 仍求解原完整目标，线性项必须是 `(H1)_S`，不能替换成 `H_SS 1`。每个 lambda 比较 v8 候选、交换候选、交换/refit 候选，仅接受原目标和预算后几何误差均不比 v8 候选差、且满足原最小非零约束的表示（工作数值精度下）。最终 lambda 仍由原 label-free 的最稀疏可行/最小几何误差 fallback 规则确定；逐 lambda 保护不保证不同 lambda 的最终解互相支配，更不保证 AD 性能。

连续、完整预算后、固定支持集三种 optimality gap 分开。交换后固定支持集 gap 针对实际保留的非零支持重新计算。`exchange_steps` 记录启发式搜索路径次数，最终来源另由 `exchange_source` / `budget_source` 记录。

## 公平对照与断点

原 36 方法全部保留，新增 `asls_prune_refit_uvarfs`，默认共 37 方法。其表示由本轮同一 ASLS 输入、同一完整连续求解的 v8 prune/refit 候选直接保存，供 Main 对照；`asls_top_weights_uvarfs` 和 `legacy_main` 继续保留旧特征/整体规则。同一有序层输入缓存求解，五个随机 seeds、Random-K、memory 共享抽样都保持。

新的 `uvarfs_main.json` 包含 `prune_refit` 和 `legacy_top_weights`，每个 lambda 记录交换前后原目标/几何改进、最终来源和 gap。指标 CSV 增加交换来源/次数/改进。`asls.json` 保持已授权的 geometry_search 定义。

## 正常扰动随机流

Gaussian noise 使用显式 `torch.Generator`，种子由稳定 SHA256 recipe `uvarfs-normal-noise-v1:seed:dataset:image_offset` 产生，写入 `fit_manifest.json`。因此在同一配置、数据顺序、设备下，前一个数据集的 PCA/求解操作、全局 RNG 消耗或断点跳过都不会改变该批正常噪声。其他扰动定义、幅度、图像数量与 variability 公式保持。

不声称 CPU/CUDA 噪声逐位一致或跨硬件完全确定。v9 与 v8 的噪声实现不同，跨版本变化不能全部归因于交换；优先使用 v9 同轮对照。

## 实际余弦几何：仅诊断

`normal_geometry_audit.json` 复用全部已有 normal fit patches，`selection_candidate=false`，不参与层/特征/lambda 选择：

- 记录每层 raw patch norm，以及 ASLS 单位层 Gram consensus 与 raw 全层 concat cosine 的差异；
- 对 Main、v8 候选、旧规则对照分别计算实际 cosine Gram 对 raw 全层/ASLS consensus/选中层 raw 的误差；
- 保存原加权 Gram 误差及加权正常向量范数，区分 U-VaRFS 原损失与 detector 行归一化后的几何；
- 比较 fit patch 的 cosine 1-NN 保持率及 reference cosine regret，仅排除自身，包含同图 patches，对 ties 敏感，不能当成官方 memory/test 指标。

只读诊断不改 normal data/memory 预算、输入归一化、原损失或 NN backend；仍需真实数据验证其差异。矩阵差值按 row chunks 累加，最高 float32 matmul 精度；新增诊断的时间/显存计入共享 fit 开销，不宣称 backbone 加速。

## 验证与服务器运行

本地检查包括：交换损失与独立完整目标穷举、cardinality/逐 lambda 双保护、原 v8 候选复现、固定支持 gap、扰动 RNG 独立性及正常 fit 重复、cosine 几何直接矩阵对照、完整 37 方法 CPU fixture 与旧结果保护。CUDA 检查仅在 CUDA runtime 可用时运行；真实 BMAD 性能需要服务器实验。

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

Liver 是首个调试数据集，最终仍是 BMAD 全六。核对 exchange 日志、37 方法、`asls_prune_refit_uvarfs`、`perturbation_rng`、`normal_geometry_audit.json`、共享 memory manifest 和 pixel 指标。预算无可行解时保留 `feasible=false`。结果审计继续只写入新的本地 `reports/` 目录。
