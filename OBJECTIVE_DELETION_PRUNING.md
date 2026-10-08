# v15：同一 cosine 目标的有限删维求解修正

项目继续为 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究**。Main 保持第 5.4/37 节已批准的 cosine/simplex/cardinality 目标及 raw 输入；L2 仍为完整消融。本轮只修改支持缩小提案的数值算法。版本为 `gpu-eval-v15-objective-deletion-pruning`，独立输出 `results/gpu-eval-v15-objective-deletion-pruning/`。

回传科学结论、实验数字与图表仅存本地 `reports/`，本文件记录实现、协议和可验证边界。优化正常训练目标不等于异常检测指标提高。

## 为什么需要修改删维

固定支持 p 的旧删维分数为：

```text
d_j = p_j / (1-p_j) * (dot(p,g)-g_j)
g = gradient of representation + beta*relative_variability
```

这是一阶删除方向近似。simplex 内点驻点有各 g_j 相等，因此全部分数为零，无法区分有限删除的二阶及更高阶代价。floor 边界上的完整梯度并非所有维度相等，不应宣称所有驻点都严格为零；但内部坐标的排序仍可能退化为很小的数值差异。更准的固定支持求解不能自行解决有限支持缩小的问题。

正常合成特征的独立探针复现了严格退化；精确有限删除可在该例中提出更低的相同正常目标。该证据只证明算法弱点，不能证明它是所有真实性能退化的唯一原因，也不能证明新算法一定提高 BMAD AUROC/AUPRO。

## 实际算法与保护

默认 `cosine_pruning_strategy: exact_objective_deletion`。在每次原预设 K 轨迹的缩小阶段，保留旧一阶提案，同时计算当前支持每个维度 j 的有限删除代价：

1. 删除 j，对其余相对权重重新归一化，并使用新 K−1 的相同总 floor mass 作 Euclidean simplex 投影。
2. 用相同完整正常行、full-reference cosine Gram、P、beta 和解析目标计算实际 base objective；不使用随机 pairs 或较小数据子集。固定 K−1 上 lambda*(K−1) 相同，因此排名无需根据 lambda 分别拟合。
3. 加权 norm 为零或目标数值无效的单维删除标记无效，不能用 epsilon 归一化伪造有效 loss；此类删除在排名内优先保留相应坐标。
4. 按实际删除损失排序，沿用原 coverage repair 提出当前目标 K 的联合删维支持，重新归一化并使用该 K 的 floor 投影。
5. 分别实际评价完整旧提案与新提案。只有新提案在同一 base objective 上 **严格低于旧提案** 才采用；若新提案没有改善则保留旧提案。旧提案无效而新提案有效时可采用有效新提案；两者均无效时记录并按原零行候选规则继续/明确失败。

单维排序并不等于多个坐标同时删除的最优组合；测试包含联合删维使新排名更差的正常反例，并验证实际目标保护回用旧提案。该保护只比较**同一当前点上、同一目标 K 的初始化目标**，不保证新旧非凸轨迹最后得到相同解，不保证 cosine geometry 单项不增加，也不保证 test 性能不退化。

原 projected BB/Armijo、固定支持 SLSQP 精修、支持交换、完整目标有限候选池 lambda 比较、实际 geometry 容差下最稀疏选解/显式 infeasible fallback 保持。仍无全局非凸最优保证。旧 `gradient_direction` 保留为显式正常求解控制；不会依据 test 指标自动切换策略。

## 成本、审计与控制

逐维评价重用同一 objective engine/full reference，逐个处理完整候选，避免并行生成 K 份 N×N Gram 的额外内存。每次支持缩小会增加至多当前 K 次有效单维评价及两次完整提案评价，另含原梯度评分；所有 normal data 预算不变，数值计算成本可能增加。必须在服务器确认实际耗时；不称为工程加速或论文创新。长评分过程有进度日志。

每个 pruning event 保存当前支持与 parent candidate ID、原实际目标/几何/残差/一阶分数跨度、所有单维有效性/loss/geometry、两个完整提案、采用来源、实际初始化目标、评价次数和耗时。candidate 保存 event ID，summary 包含所有有效/无效尝试成本。分析器核验父子/支持预算、loss 差、联合保护、全部成本与原 lambda/几何选择；旧 v12/v14 无 pruning 字段的记录仍可审计，带新配置而缺 manifest 的结果拒绝。

原 quadratic/FISTA/v11 forward solver、ASLS、DINO backbone、fit/eval/memory/metric 实现保持。raw/L2 × cosine/Gram 四支 152 方法、五 seeds、按各支实际 Main K 配对 PCA/Random、同 normal 图像/patch/扰动、ASLS 与 memory 抽样保持。冻结 backbone、全部层候选、BMAD 六数据集、normal fit/variability/memory/feature 预算、beta/lambda 数字、cosine 1-NN/Top-1%/bilinear pixel protocol 保持。没有切换 L2 Main、增加异常标签、删基线或增加新损失。

## 第 19 节检查

| 检查 | 结论 |
| --- | --- |
| 研究问题、ASLS、U-VaRFS 精确目标 | 保持；只升级同一目标的支持缩小提案 |
| 训练中的异常 labels/masks | 无；回传标签仅用于评价与配对分析 |
| 主输入、detector、全六 | 保持 raw Main、原 normal-memory detector、BMAD 六 |
| main/baseline 数据/维度预算 | 保持；所有 cosine U-VaRFS 使用同一数值策略，新增求值成本完整记录 |
| 原控制、PCA/Random、五 seeds | 保持完整；按各支实际 K 公平比较 |
| 新旧可比与版本 | 权重/支持/K 可能改变，bump v15、独立目录和新指纹；不能位级续接旧结果 |

## 检查和运行

完整回归 **149 passed、6 skipped**（本机无 CUDA）。独立 sample-space oracle 覆盖有限删除 loss、变化后的 floor 与 variability；驻点退化、联合删除反例、zero-row 失效、实际目标 guard、全部成本与候选链接检查通过。两个 primary 的 152 方法 fit/memory/image/pixel、同数据独立原 Gram 控制及 resume/report 通过。v12/v14 共 48 个主分支的历史审计通过，原产物未重写。

本机没有真实 BMAD、权重或 CUDA，不能据数学/CPU fixture 宣称新的检测收益。服务器先 Liver 核验 finite-deletion/guard/refit/精修/交换日志、成本、四支拟合、152 方法和评价，再固定设计全六；不得看 Liver test 后调整 beta、lambda、K、ASLS 或 input。

```bash
git pull --ff-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

旧结果不覆盖、不混合、不续接。代码/配置/测试和本开发交接按第 28 节持续授权提交推送；实验、分析、checkpoint、图像与日志默认仅留本地。
