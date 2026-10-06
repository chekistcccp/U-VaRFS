# v4 结果分析与 v5 修改说明（2026-10-06）

本文保留批准前的 v5 实现记录。用户随后明确批准 patch 主方法；当前版本与验证步骤见 [v6 升级说明](PATCH_ASLS_UPGRADE.md)，以下 pooled Main/等待选择属于历史状态。

本轮分析对象是 `results/gpu-eval-v4-protocol-fixes/`，六数据集 × 28 方法共 168 行。完整报告、五 seed mean ± SD、配对 bootstrap、来源指纹、压缩/耗时表与图见 [v4 分析](reports/2026-10-06-v4-analysis/analysis.md)。原始结果目录与用户回传的 `run.log` 保留原样。

研究保持 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究**。最终 benchmark 仍为 BMAD 全六数据集，Liver 是 CT 主分析/首轮调试数据集。

## 结果说明了什么

- 完整性修复已在服务器得到验证：Brain/Liver/RESC 异常 mask 分别覆盖 3075/3075、660/660、764/764；全部图像与像素指标齐全。逐图 AUROC/AP、五 seed 汇总和记录的 Git 源码/config 指纹均已复核。
- Main Macro Image AUROC 为 **74.26%**，Fixed-4 Raw 为 **78.12%**，同 K Random Raw 五 seed 均值为 **79.47%**。Main 仅在 Liver/OCT2017 高于同 K Random Raw/U-VaRFS；不能宣称跨模态整体优于固定/随机层。
- 同一 ASLS 层输入下，U-VaRFS 高于 Random feature 均值（6/6）、PCA（4/6）和 Raw（5/6）；RESC 相对 Raw 有下降。层选择是优先诊断对象，两个主模块都保留。
- Main 采用 2/12 层、87–250/4608 维，表示压缩明显；Liver Pixel AUROC/AP/AUPRO 为 98.34%/17.32%/95.10%。Brain Image AUROC 比 Fixed-4 Raw 低约 17.76 个百分点，定位也较差。压缩不能代替性能证据。

上述压缩以全 12 层 4608 维为候选，包含层与特征两步裁剪；仅 U-VaRFS 阶段的输入是选中两层的 768 维，再压缩到 87–250 维。
- Pooled consensus Gram 非对角均值约 0.93–0.994；Brain/Liver/RESC 接近常数。Gates 全部接近 0.993，微小差异与 variability 排序同向。类似 Gram 下，共同 gate 的几何项近似 `1-p`，推动全开；满足 pooled 几何约束并不保证局部 patch 几何得到保留。
- 六个 Main 均选最小 lambda=1e-6，几何误差约 0.052–0.071，**全部不可行**。投影残差约 2e-8–1.6e-7，但迭代变化未达容差：数值收敛、feature support 稳定性和几何可行性须分别记录，不能由任一标记推断另外两者。
- X-ray 异常比例约 95.46%；其 Main Image AP=97.86% 需结合这一比例与 AUROC=71.33% 解读。

Bootstrap 使用 2000 次按正常/异常分层的 image-level 配对重采样，Random 比较使用每次重采样内全部五 seed 的 AUC 均值。没有患者/slide 分组 manifest、预设非劣界值或多重比较校正，不能据此宣称患者级显著性或非劣。评价标签不回流拟合。

## 修改前主线检查

| 检查 | 本轮处理 |
|---|---|
| 是否改变研究问题、全六 benchmark 或 frozen backbone | 否 |
| 是否改变主 ASLS 定义 | 否；默认仍 pooled geometry、sigmoid gates、原 loss、Adam 与几何约束离散前缀 |
| 是否改变 U-VaRFS 目标函数 | 否；原 representation/variability/L1、已有 normalization/scaling 保留 |
| 是否引入异常标签/mask 到训练或参数选择 | 否；只用于最终评价与固定方法统计比较 |
| 是否改变 cosine 1-NN detector/Top-1%/插值 | 否 |
| 是否给 Main 与 baseline 不同数据/表示/memory 预算 | 否；原全部预算和五 seeds 保留 |
| 是否允许旧结果断点与新实现混用 | 否；严格精度和迭代实现改变数值输出，另立版本目录 |
| 是否 bump experiment version | 是：`gpu-eval-v5-normal-audit` |

## 已实施的改进

### 原目标的数值求解与可行性诊断

1. ASLS/U-VaRFS 拟合时使用 `float32_matmul_precision=highest`，退出后恢复原设置。Frozen DINO bf16 与 exact cosine NN 设置保留。TF32 的吞吐优化不应掩盖拟合目标的细小变化。
2. Batched FISTA 对每个 lambda 独立重启失配的动量，保留保守 Lipschitz 步长、原 proximal/box 投影、lambda grid、beta=0.002、2000 次上限与 tol=1e-5。
3. 记录原目标的 representation/variability/sparsity 分项及 box linearization gap。对凸目标 `F`，`gap = max_{z in [0,1]^M} <grad F(w), w-z>` 在精确算术下上界 `F(w)-min F`；当前记录为浮点数值诊断。
4. `objective_converged`（gap ≤ tol）、原 `solver_converged`（迭代变化与投影残差）和预算后 `feasible` 分开；lambda 选择仍遵循原最稀疏可行/最小几何误差 fallback 规则，没有新增最优性标记筛选规则。
5. 不可行时额外计算 `lambda=0` 的 **normal-only 诊断**，保持相同 beta 与 feature budget。它标记 `selection_candidate=false`，不加入原 lambda path、不参与 Main 选择，用于区分 L1、variability、预算和数值误差的影响。该诊断不保证得到可行解或性能提升。

ASLS 增加各 loss 项对 logits 的梯度轨迹，帮助区分几何、扰动和稀疏项的作用，没有替换成 Hard-Concrete，也没有把权重和归一化改为除以 gate 总和。

### Normal patch geometry 消融

新增 `asls_patch_raw` 和 `asls_patch_uvarfs`，默认共 **30 方法**，原 28 方法全部保留。消融使用与 U-VaRFS 相同的既有 4096 个 normal fit patches，不新增图像/patch/扰动预算；其余 loss、sigmoid/Adam 和 layer budget 完全相同。

为控制内存，使用单位归一化特征 `F_l` 的恒等式：

```text
K_l = F_l F_l^T
Q_lm = <K_l, K_m>_F = ||F_l^T F_m||_F^2
||sum_l a_l K_l||_F^2 = a^T Q a
```

`Q` 仅为 L×L；构建时使用已有 L·D feature-space cross-products，不保存 L 个 N×N patch Gram。这是同一 cosine Gram 误差的等价计算，已验证误差、梯度和离散 subset error 与直接 Gram 一致。

**默认 Main 仍用 pooled geometry。** 按 [AGENTS.md 第 19 节](AGENTS.md)，主 ASLS 输入切换需要用户明确指令，当前等待该选择；已经实现可复核的 patch 消融。若以后批准切换，要明确标记新的主方法版本、保留 pooled 消融，并以 normal-only 固定规则选择层；不得在测试结果中择优选择 pooled/patch 版本。

### 像素评价与结果保护

- 同图像 mask 的连通域索引只计算一次，供全部方法的最终评价复用；mask 不进入 fit。
- Pixel histogram 与 normal PRO threshold counts 共享一次排序，保持全部边界/ties 语义、1024 bins、100 thresholds 和 FPR=0.3，不改变原指标近似定义。
- 新版本默认输出 `results/gpu-eval-v5-normal-audit/`。运行前拒绝向含其他版本 CSV 的目录写入，`FORCE_EXPERIMENT=1` 也不能覆盖历史版本；代码指纹统一 POSIX 路径与 LF，跨平台不再因 CRLF 产生差异。
- 每个方法的 ASLS geometry/feasibility 分别记录；不会把 Main 的 ASLS 状态写到固定/随机层 baseline 上。

## 验证边界

本地工作机缺少真实 BMAD 数据、DINOv3 权重和 CUDA PyTorch，**未取得 v5 真实性能结果**。

- 41 项回归检查通过；数学与 CPU fixture 覆盖两种 ASLS 输入、全部 30 方法、normal fit、共享 memory、最终图像/像素指标、随机对照、来源保护和 heatmap I/O，`run.sh` Bash 语法检查通过。
- 直接加载 Git `0295e55` 的 v4 pixel 实现，与新版在 4 图 × 28 方法 × 448² 的独立 fixture 比较：全部 histogram、FP/PRO counts 与最终指标完全一致；包含正常/双连通域 mask、阈值 ties 和 score clipping。
- 三次交错计时的 CPU 中位耗时比约 1.63，详见 [局部对照记录](reports/2026-10-06-v4-analysis/pixel_equivalence_benchmark.json)。这是局部像素累积验证，不能解释为整套服务器或 detector 加速。
- v4 源目录全部 207 个产物的 SHA256 已复核；原 CSV/JSON/heatmap 未改写。v3 报告保留。

数据集级 `fit_seconds/memory_seconds/eval_seconds/peak_gpu_gb` 包含全部方法共享开销；新增消融与诊断也有计算成本。单个 U-VaRFS `fit_seconds` 是选择路径求解时间，不包含随后 lambda=0 诊断；整数据集 fit 时间包含该诊断。不得用重复的共享耗时/显存证明 Main 单独加速。

## 服务器验证顺序

本轮 v4 已确认 mask 归一化完整，已有数据/模型时先单独验证新版 Liver：

```bash
git pull --ff-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

应看到 pooled ASLS、patch geometry 消融、各 U-VaRFS path 与可选 `lambda0-diagnostic`、PCA、memory、test heartbeat；检查选层/几何/证书和三个 pixel 指标。Liver 完成后运行全六：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

比较新版与 v4 时保留各自版本、config/code 指纹和准确方法身份；不直接合并为同一实验。服务器环境由用户维护，`run.sh` 不安装依赖。

只复现本轮结果审计（不加载模型/不拟合）：

```bash
python scripts/analyze_results.py --results results/gpu-eval-v4-protocol-fixes --out reports/2026-10-06-v4-analysis --bootstrap 2000
python scripts/analyze_results.py --out reports/2026-10-06-v4-analysis --plots-only
```

该报告模板明确限定 v4，避免将归档文字误用于新的性能结果。输出目录必须位于原结果目录之外。
