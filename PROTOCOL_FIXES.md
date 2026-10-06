# 原始研究协议下的实现修正（2026-10-06）

主线保持 **BMAD 全六数据集 + Frozen DINOv3 ViT-S+ 全层候选 + ASLS + U-VaRFS + normal memory exact cosine 1-NN**。

这次修改属于数学统计实现纠错、求解精度、评价完整性和公平对照。ASLS 的 sigmoid/Adam 目标、U-VaRFS 的 representation/variability/L1 目标、beta、lambda grid、geometry tolerance 和所有采样/表示预算保持原样。没有用 anomaly labels、test AUROC 或 lesion masks 选择层、lambda 或超参数。

## 已修正的具体问题

| 问题 | 修正与记录 | 性质 |
|---|---|---|
| variability 为每个 batch 的比值取平均，遗漏 batch 间方差 | Welford 合并总体方差；全体采样 patch 的平方差均值除以总体方差加 epsilon | 按原公式修正统计估计 |
| lambda 挑选检查连续权重，之后截断到 feature budget 可能失去几何可行性 | 每个 lambda 先应用同一维度预算，再检查最终加权表示；记录连续/最终误差和完整 lambda path | 按原选择规则修正可行性判断 |
| 原结果所有 U-VaRFS fit 均耗尽 400 次迭代 | 保守 Lipschitz 上界、投影固定点残差和每 lambda 收敛标记；默认上限 2000，容差仍为 1e-5 | 同一凸目标的数值求解改进 |
| ASLS gates 接近全开 | 保留目标与优化器，增加 loss/gradient/gate range/normal geometry 诊断 | 可追溯诊断，未重新设计 ASLS |
| mask 可能遗漏或错配 | released AD 的 label/mask 目录归一化；优先相对路径，兼容唯一 stem、扩展名与常见后缀；歧义报错 | 数据/评价纠错 |
| Brain/Liver/RESC 无 pixel metrics 仍能被视为完成 | 异常 test mask 完整性预检；缺失即报错；输出覆盖清单、pixel 指标和示例 heatmaps | 补齐原始评价协议 |
| 完美像素排序的 PR 积分可能为 0.5，AUPRO 可能缺少末段或为 NaN | histogram average precision，补齐 ROC 起点；AUPRO 插值至 FPR=0.3 | 评价数学纠错 |
| Random-4 与 ASLS 两层的层数预算不同 | 增加 Random-K Raw / U-VaRFS，K 等于 ASLS 层数，全部 5 seeds；保留所有原 baseline | 原计划中的公平消融 |
| 不同方法 reservoir seed 不同 | 所有方法使用同一正常图像、patch 行和 reservoir 抽样 | 公平 memory protocol |
| 配置/代码/数据变更仍可能复用已有 CSV | 检查版本、配置/代码指纹、数据路径/大小/mtime、准确方法集合、完成标记和指标完整性 | 防止不兼容断点混用 |
| Windows 类目录别名导致异常样本重复 | 去重解析后的类目录；raw OCT2017 disease folders 走原始确定性拆分 | 跨平台纠错 |

U-VaRFS 保留已有 row normalization、H normalization、P column normalization 与 R/H Frobenius scaling；未引入新的权重标准化或损失。`max_iter` 是数值求解上限，不能据此宣称算法方法创新，也不保证每个数据集的每个 lambda 都会收敛。

预算后无可行解时，仍按原协议选最终 geometry error 最小者；`selection_reason=minimum_geometry_error_fallback` 与 `feasible=false` 明确记录这一情形。数值尚未收敛时同样记录和提示。论文不能把这些解描述为满足 0.05 几何约束。

像素指标仍为 streaming histogram/threshold 近似：1024 bins、100 PRO thresholds、cosine distance 范围 0–2、AUPRO 上限 FPR=0.3。像素 mask 按既有 letterbox 与 nearest-neighbor 处理到 model input size。升级未将它们改成原图分辨率的精确排序指标。

## 新结果与复现信息

版本：`gpu-eval-v4-protocol-fixes`。

默认输出：`results/gpu-eval-v4-protocol-fixes/`。原回传 v3 CSV、JSON、run.log 和原分析报告保持原样，不能与新版结果合并为一张可直接比较的表。

每个数据集额外输出：

- `fit_manifest.json`：normal training 样本、fit patch 索引、扰动列表和原始 variability 向量。
- `memory_manifest.json`：统一 memory 图像/patch 抽样、seed 与 bank 行数。
- `run_metadata.json`：完整配置、代码/数据/配置指纹、Git revision、backbone 参数和计算环境。
- `mask_coverage.json`：异常 test 样本 mask 完整性与缺失路径。
- `asls.json`：原选层结果及饱和、几何、梯度诊断。
- `uvarfs_*.json`：所有 lambda 的预算后几何误差、可行性和数值收敛状态。
- `image_predictions.csv`：每幅 test 图像的固定方法分数与最终评价标签，供后续配对统计；不回流拟合。
- `heatmaps/`：确定性选取的最终评价示例，显示 Input、mask（如有）、All Raw 与 Main；可视化使用固定 cosine distance 0–2 映射。

`fit_seconds`、`memory_seconds`、`eval_seconds`、`peak_gpu_gb` 仍是整个数据集全部方法共享的运行量，不能拿同一重复值证明单个方法加速。新增 `matching_and_aggregation_seconds` 只测该方法的 NN 查询和 image Top-K 汇总，排除共享 DINO forward、feature transform 和 pixel metric/绘图时间；其中第一轮可能含 CUDA/GEMM 预热，不是严格的独立延迟基准。`memory_vector_fp16_mib` 是理论 FP16 向量容量，不包括索引、query 临时张量、backbone 与完整显存占用。

Random baseline 按全部 5 seeds 报 mean ± sample SD。默认共 28 个方法；单数据集调试时全局 CSV 只汇总本次请求的数据集，六数据集总表应由最终 `DATASETS=all` 运行生成。

数据指纹基于文件身份、大小与纳秒 mtime，未对数十 GB 原始 archive 做内容哈希。原始 archive 不会被修改或删除。

## 服务器运行顺序

先验证 Liver，这是 CT 调试步骤；最终实验仍须 BMAD 全六数据集。第一次 v4 运行需要重新做数据归一化（`NORMALIZE_VERSION=3`），因此保留 preprocessing 阶段：

```bash
SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

可先单独预处理与预检，确认 train normal-only 和 mask 覆盖后再拟合：

```bash
PREPROCESS_ONLY=1 DATASETS=liver bash run.sh
python scripts/run_all.py --dataset liver --check-data-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

关注 `asls.json` 的 gate span/variability rank、`uvarfs_main.json` 的最终 `feasible` 和 `solver_converged`、`metrics.csv` 的 image 与全部 pixel 指标。不能用 test 指标挑 lambda、层或 feature budget。

Liver 验证后运行六数据集，第一次同样保留 preprocessing，以更新其他数据集 mask 布局：

```bash
SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

后续才使用 `SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh` 断点续跑。`run.sh` 不安装依赖，由用户现有环境提供。数据预检选项无需加载 KerasHub/DINOv3 权重或使用 GPU；它会把覆盖清单写入新版结果目录。

## 验证边界

本地 regression suite 覆盖总体统计的 batch 不变性、独立 box-constrained quadratic 优化对照、最终预算几何检查、mask 唯一性与 Windows 路径去重、像素指标、normal-only guards、checkpoint 不兼容拒绝，以及全部 28 个方法的 CPU 测试流程（小型特征提取 fixture + exact cosine）。

```bash
python -m pytest -q -p no:cacheprovider
```

工作机没有实际 BMAD 数据、DINOv3 权重与 CUDA PyTorch；因此未运行真实 Liver、服务器 GPU 或六数据集新版性能实验。实现测试通过不能视为 AUROC/AUPRO 已改善。先重跑得到完整 v4 结果，再据 normal-only 诊断决定后续工作。
