# ASLS patch 主方法升级（2026-10-06）

用户已明确批准：**“批准修改patch主方法”**。当前默认主方法为 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究**，其中 ASLS 改用 normal patch geometry；BMAD 全六数据集与其他研究约束保持。

版本：`gpu-eval-v6-patch-asls`。输出目录：`results/gpu-eval-v6-patch-asls/`。v4/v5 结果与原回传文件保留，不能用旧 checkpoint 代替新实验。

## 批准范围与主线检查

| 检查 | v6 处理 |
|---|---|
| 研究问题、BMAD 六数据集、Frozen DINOv3 全层候选 | 保持 |
| 主 ASLS 定义 | 经用户批准，将 geometry 输入由 pooled 改为对应位置的 normal fit patches |
| ASLS loss、sigmoid gates、Adam、概率排序与几何约束前缀 | 保持，未改为 Hard-Concrete |
| U-VaRFS 目标、beta、lambda path 与无标签选择规则 | 保持 |
| 异常标签/mask 参与 fit 或超参数选择 | 不允许，仍仅用于最终评价 |
| Detector、Top-1% 与 bilinear pixel map | 保持 normal memory + cosine 1-NN |
| Main/baseline 数据、表示及 memory 预算 | 保持统一 |
| 与旧实现互换结果或断点 | 不允许，版本与目录已更新 |

## 实际切换

- `main` 使用 normal patch geometry 选层，再使用原 U-VaRFS 选择 latent dimensions；`asls_raw`、`asls_pca`、`asls_random_seed*` 使用同一组 patch ASLS 选层。
- 默认 `geometry_representation=patch`，配置及 fit/metadata 的缺省行为一致。
- `asls_pooled_raw`、`asls_pooled_uvarfs` 为明确的旧输入消融。它们替换 v5 中已成为主分支的两项 patch 消融，默认仍为 30 方法，原 28 项方法类别全部保留。
- Random-K 的 K 仍匹配当前 Main 选层数，并报告全部五 seeds（42、123、3407、2026、2027）。K 由 normal-only 几何规则决定，不固定为 v4 的两层。
- Patch 与 pooled 使用同一次 Frozen DINO 全层特征提取、同一 normal train 采样和 variability；同轮全部方法共享 memory 图像/patch/reservoir sampling。

ASLS patch geometry 复用已有 256 张正常 fit 图像 × 每图 16 patches，即默认 4096 个对应 normal patch rows。Pooled 消融使用这些图像的 256 个 pooled rows。没有增加 fit 图像、patch、扰动或 memory 数据预算。图像/patch 数不足时以实际样本数为准，记录在 manifest/diagnostics 中。

所有层 patch rows 必须对应。对于单位归一化特征 `F_l`，原 cosine Gram 为 `K_l=F_l F_l^T`，原 full-layer consensus 为 `K_C=mean_l K_l`。继续使用已有等价计算：

```text
Q_lm = <K_l,K_m>_F = ||F_l^T F_m||_F^2
||sum_l a_l K_l||_F^2 = a^T Q a
```

改变的是主 ASLS 的正常几何输入；loss 公式、扰动稳定性项、稀疏项、2–6 层预算、0.05 tolerance 和离散规则均不变。U-VaRFS 仍使用 beta=0.002、原 15 个 lambda、32–256 维预算、2000 次上限与 tol=1e-5。Memory 仍为 1024 张正常图像、每图 32 patches、20000 bank rows；主 detector 不变。

v5 的 highest FP32 拟合、FISTA 动量重启、目标/box gap 诊断、非候选 lambda=0 诊断与像素评价等价优化继续保留，详见 [v5 修改记录](RESULT_DRIVEN_FIXES.md)。

## 输出与比较口径

- `asls.json` 和 `asls_patch.json` 为当前 Main patch 选层；`asls_pooled.json` 为旧输入消融。
- `fit_manifest.json` 与 `run_metadata.json` 记录 `asls_geometry_representation=patch`；`asls.json` 的 `normal_geometry_samples` 反映实际 patch rows。
- `uvarfs_main.json` 的层应与 patch `selected_layers` 一致；各方法 CSV 分别记录自己的 ASLS geometry 和可行性。
- 若 ASLS 无预算内可行前缀，仍按既有规则返回上限前缀并记录 `feasible=false`；U-VaRFS 无可行解时仍用 minimum-error fallback。切换 patch 不保证消除 gate 饱和或保证几何可行。
- v6 同轮 `main` 与 `asls_pooled_uvarfs` 使用同一数值实现，是几何输入消融的主要对照。v4→v6 同时包含数值/评价工程修正，不能将全部差异只归因于 patch。
- 不根据 test AUROC 在 pooled/patch 间择优选择主方法，不用异常标签调 layer/lambda/budget。

## 验证与运行

本地通过 42 项回归检查，包含主方法实际 patch rows、pooled 消融 rows、选层到 U-VaRFS 的传递、normal-only guards、同 K 随机层与共享 memory、30 方法 CPU fixture 最终评价、Gram 误差/梯度等价及 v4/v5 覆盖保护。

本地仍缺真实 BMAD 数据、DINOv3 权重与 CUDA PyTorch，**没有 v6 真实性能结果**，不能宣称本次切换已提高 AUROC/AUPRO。历史分析见 [v4 报告](reports/2026-10-06-v4-analysis/analysis.md)。

服务器先验证 Liver（CT 主分析/调试数据集，最终仍须六数据集）：

```bash
git pull --ff-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

日志应包含 `ASLS start: geometry=patch`、`ASLS pooled geometry ablation`、`U-VaRFS main`、memory 与 test。核对 normal-only manifest、各模块 geometry feasibility、三个 pixel 指标及版本。

Liver 完成后运行全六：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

已有处理数据与模型时无需重复下载或安装；`run.sh` 继续不安装依赖。历史结果不能与 v6 合并为同一个实验表。
