# v16：同支持的相对权重对照与可行性作用域（历史开发记录）

当前 v17 性能优先定义见 [PERFORMANCE_FIRST.md](PERFORMANCE_FIRST.md)，本节控制与诊断继续保留。

版本 `gpu-eval-v16-weighting-control`；新目录 `results/gpu-eval-v16-weighting-control/`。
本文件记录开发协议；真实回传指标、分析与图表仅留本地 `results/`、`reports/`。

## 方法身份与公平性

正常表示目标下降不能单独证明异常检测收益。需在同层、同实际 K、同支持索引、正常数据和 memory 下，区分 U-VaRFS 支持选择与相对权重的作用。

新增 `asls_selected_raw`：复制本输入/目标分支 Main 的**有序层与 active 索引**，保留这些坐标的原输入值，按原流程行 L2 归一化。这等价于该支持上的均匀正相对权重；不重新排序、选维、拟合或优化。支持仍由 normal-only Main 产生，无额外正常数据或 backbone forward。它是 fixed-support weighting ablation，不是新主方法或 random feature baseline。

raw/L2 × cosine/Gram 四支各有自己的控制：`asls_selected_raw`、`gram_asls_selected_raw`、`layer_l2_asls_selected_raw`、`layer_l2_gram_asls_selected_raw`。原 152 方法、五随机 seeds 全部保留；默认 **156 方法，全六 936 行**。各分支 K 可以不同，控制只匹配自己的 Main。PCA/Random 保持各支原匹配规则。

全部方法共享 normal memory 图像/patch/reservoir 抽样规则、cosine 1-NN、image Top-1% 与 pixel 插值。新增对照的 memory/匹配/统计成本计入整套共享成本，不将方法数变化后的总耗时当作单方法加速证据。

## 审计与报告

- 各支 `selected_support_control.json` 保存 Main 源方法/目标/输入、有序层、active、K、均匀权重身份、`extra_fit=false` 和 `selection_candidate=false`。
- CSV 控制行保存 `support_source_method` 与 `representation_weighting`；不借用 Main 的 lambda、feasible 或求解证书。
- normal geometry 比较控制与 selected input、full hierarchy、unit-layer consensus，以及 weighted Main 的 cosine error 与正常 1-NN agreement；排除自身，包含同图 patches，不代替官方 memory/test 评价。零行按现有 detector 行为记录，不作为新选解规则。
- 报告核验控制与 own Main 的 exact layers/active/输入/目标/K/源方法一致，加入同支持异常指标和配对 image bootstrap。fit-only 仍不声明异常性能。
- 完整 resume 除原四支文件还须有四个控制 JSON。新版本不能覆盖、混合或续接 v15。

## 有限可行性的准确作用域

原 `feasible` 与选择语义保持：从规定 lambda grid 各点的**有限生成池完整目标最小解**中选最稀疏几何可行点；无可行路径点则按原路径最小几何 error fallback。

只读 helper 统计全部已生成候选的可行数量、最小 error、最稀疏可行 K，以及 lambda-path 可行点数，区分：

- `feasible_lambda_path_point`
- `feasible_generated_candidates_outside_lambda_path`
- `no_feasible_generated_candidate`

生成池中有满足容差的支持，不保证它是任一规定 lambda 的完整目标最小点。无可行路径或无可行生成候选都不证明全部预算内支持不可行，`global_budget_infeasibility_claimed=false`。helper 不进入求解/选解路径、不修改旧 JSON。不得通过直接选池可行解、延伸 lambda grid、放宽容差或切换 Main 输入掩盖 fallback。

## 第 19 节检查

1. 研究问题保持 BMAD 六、Frozen DINOv3、ASLS、无异常标签 U-VaRFS、normal memory。
2. ASLS 定义与全部层候选不变。
3. Main cosine/simplex/cardinality 目标、原 Gram 控制与全部 solver 保持 v15；K、floor、beta/lambda 与容差不变。
4. fit 不使用 anomaly labels、test labels 或 lesion masks。
5. detector/Top-1%/pixel 定义不变。
6. Main 与所有控制的数据/表示/memory 预算规则一致。
7. 四个新增消融改变整套方法数和总成本；相同资源的原方法数值应保持，不能直接比较整套耗时。
8. bump EXPERIMENT_VERSION 与独立目录，缺少新对照的旧 checkpoint 不视为完整 v16。

raw Main 固定，L2 消融保留，不据 test 自动挑主输入、目标、lambda 或支持。该修改增加验证研究主张所需的控制，不宣称已改善真实 AUROC/AUPRO。

## 验证与运行

本地检查包括新控制的均匀权重数学、两种 primary 配置的原 152 方法前后数值与逐图预测、共享 fit/perturbation/memory 清单、错配审计拒绝、四支 resume、历史报告与全部已有回归。检查结果见 AGENTS 第 41 节。

本机缺真实 BMAD、模型与 CUDA；fixture 不能证明真实性能。服务器先验证 Liver：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

再固定配置全六：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

Liver 是运行验证/CT 主分析数据集，不按其 test 指标调参。核验四个同支持控制、正常抽样、fallback 作用域与异常指标。代码/配置/测试/开发交接依第 28 节提交推送，实验产物仅留本地。
