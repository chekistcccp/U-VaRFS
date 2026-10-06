# v8 原目标驱动的稀疏选择升级

用户于 2026-10-07 明确要求：“能否换一种性能更好的稀疏方式？再改进一下”。据此升级主稀疏求解步骤，并保留旧规则消融。研究仍是 Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测，最终覆盖 BMAD 全六数据集。

版本 `gpu-eval-v8-objective-sparsity`，新结果目录 `results/gpu-eval-v8-objective-sparsity/`。本文记录算法与开发协议，实验结果与分析报告只保留本地。

## 方法边界检查

- 研究问题、Frozen backbone、全部 DINO 层候选、normal-only 训练、U-VaRFS 数学目标、cosine 1-NN、Top-1% 和 pixel evaluation 定义保持。
- ASLS 主离散规则由 gate prefix 改为正常几何组合搜索；U-VaRFS 由直接 Top-weight 截断改为原目标驱动剪枝与固定支持集重优化。这是用户授权的主稀疏算法升级，不能描述为仅工程等价加速。
- 所有正常 fit/variability/memory 数据预算、ASLS 2–6 层、feature 32–256 维、0.05 几何容差、beta 和原 15 个 lambda 均保持。
- 新增支持集重优化最多 2000 步，所有采用新版 U-VaRFS 的固定/全层/随机层对照使用同一配置。旧算法消融明确保留原求解流程；计算成本分别记录，不能声称与旧算法相同耗时或默认更快。
- 新旧主方法定义不同，必须单独版本、同轮消融；旧结果不可覆盖或混用。没有根据 test AUROC 选择规则或调参数。

## ASLS：正常几何约束下搜索层组合

默认 `discrete_selection: geometry_search`。保持原 sigmoid/Adam、loss 与扰动估计，复用正常 patch Gram 内积 Q 和已学 gates。

枚举 k=2,…,6 的组合，在第一个包含可行解的 k 中选择 gate 总分最高者，误差与层编号作确定性 tie-break。若整个预算均不可行，选择几何误差最小者并报告 `feasible=false`。12 层且无可行解时检查 2497 个组合。

原设备计算得到的 gate-prefix 选择和路径单独保存，旧规则对照直接复用它们，避免用重新计算的近似值改变历史对照。此处没有 Hard-Concrete。

## U-VaRFS：精确删除损失与支持集重优化

原归一化后的目标仍为：

```text
F(w) = 0.5*(1-w)^T H (1-w) + beta*rscale*||P^T w||² + lambda*sum(w)
0 <= w <= 1
```

连续 lambda path 仍用原 batched FISTA。对超出预算的连续解，令 `A=H+2*beta*rscale*P P^T`、`g=Aw-H1+lambda`，将维度 j 置零的精确原目标变化为：

```text
Delta F_j = 0.5*A[j,j]*w[j]² - g[j]*w[j]
```

依次删除损失变化最小的维度。每次删除后更新完整梯度，保留特征间的交互作用，周期性重算梯度控制浮点累计误差；R 仍通过低秩乘法计算，不构造大矩阵。

随后对旧 Top-weight 与新剪枝支持集分别做 batched box refit。固定支持集 S 的线性目标必须使用 **`(H1)_S`**，不能改成 `H_SS 1`，否则会丢失被删除维度的交叉项并改变原目标。P 的归一化、rscale、beta 与 lambda 都保持原值。最多 2000 次重优化迭代仅改善同一目标的求解，不新增损失项。

本实现与按损失曲率评价重要性的研究思路相关，参见 [Hassibi 与 Stork 的原论文](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)。本项目使用已知二次目标的精确单维删除变化，没有实现逆 Hessian OBS，也不训练或剪枝 DINO backbone 参数。

## 每个 lambda 的正常训练保护条件

四类候选为旧 Top-weight、新逐步剪枝、旧支持集 refit、新支持集 refit。每个 lambda 以旧候选为参照，接受候选必须满足：

1. 原目标值不增加；
2. 正常训练几何误差不增加；
3. 不破坏原候选的最小非零维度约束，且不超过统一最大维度预算。

在满足条件者中按原目标值选择；若启发式搜索没有改善，则保留旧候选。条件在实际零填充权重上计算，包含最终阈值处理，严格以当前数值精度评价。没有用异常标签决定接受与否。

最终 lambda 仍按原规则取最稀疏可行解；若没有可行解，取最终几何误差最小者。保护条件是 **逐 lambda** 的，不保证选择更稀疏可行解后与旧最终解的误差完全相同，也不保证 Image AUROC/AUPRO 提升。

`fixed_support_box_gap` 只证明当前固定支持集的 box 最优性程度；`budgeted_box_optimality_gap` 仍相对完整连续 box 问题。两者都不证明找到了全局最佳稀疏支持集。`lambda=0` 继续只作诊断，不参与正式 path。

## 同轮对照与公平性

保留原 32 方法，新增四项，默认共 36 方法：

| 对照 | 层选择 | 特征稀疏化 | 用途 |
| --- | --- | --- | --- |
| main | geometry_search | objective_prune_refit | 升级主方法 |
| asls_gate_prefix_raw | 旧 gate-prefix | Raw | 原层规则 |
| asls_gate_prefix_uvarfs | 旧 gate-prefix | 新特征稀疏化 | 隔离层规则差异 |
| asls_top_weights_uvarfs | 新层规则 | 旧 Top-weight | 隔离特征稀疏化差异 |
| legacy_main | 旧 gate-prefix | 旧 Top-weight | 完整旧流程 |

原固定/全层、PCA、Random feature、随机层五 seeds、pooled 消融均保留。`asls_geometry_search_raw/uvarfs` 保留为明确同规则对照，默认与对应 Main 层输入重合，不删除旧方法名称。

同一有序层输入缓存一次完整连续求解，旧截断对照复用该 path。所有方法共享同一 memory 图像、patch 与 reservoir 抽样；Random-K 仍匹配 Main 层数，Random feature/PCA 仍匹配 Main 维度。fit/memory/eval/peak 为整套方法共享量，额外 refit 与消融成本不能隐藏。

JSON 保存连续权重、实际 `budgeted_weights`、旧结果、所有 lambda 的候选来源、目标/几何改善和固定支持集证书；CSV 同步策略与选中候选的诊断。连续证书与实际 detector 表示保持区分。

## 验证与复跑

60 项 CPU 检查通过，1 项 CUDA 对照因本机没有 CUDA runtime 跳过。检查包括直接完整目标逐维删除对照、独立 SciPy box 优化、遗漏特征交叉项、padding 对第 0 维的影响、原算法数值复现、全部 lambda 的保护条件、36 方法正常训练→memory→pixel evaluation、五 seed 公平性和旧目录覆盖保护。

本机缺少真实 BMAD 数据、模型与 CUDA PyTorch，未验证新版医学 benchmark 性能，也不能从局部 CPU 检查推断服务器耗时。

先在服务器验证 Liver（CT 调试/主分析数据集，最终仍是全六）：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

确认新版本/目录、geometry_search Main、prune/refit 日志、三个隔离对照与 legacy_main、normal manifests、每个 lambda 保护条件、完整 image/pixel 指标后，再运行全部 BMAD 六数据集：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

run.sh 不安装依赖；实验产物、日志、图表与结果分析保持本地，代码/配置/测试/开发文档按持续授权提交推送。
