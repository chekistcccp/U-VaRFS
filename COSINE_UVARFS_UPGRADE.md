# v12：已批准的 cosine/simplex U-VaRFS

用户明确批准：**“批准设计的改进，修改对应代码并同步到仓库”**。此前 [目标提案](UV_COSINE_OBJECTIVE_PROPOSAL.md) 的待批准状态由本授权取代。本版本实际接入新的主目标，不能称为原公式的等价工程优化。

项目仍为 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究**，最终为 BMAD 全六。版本 `gpu-eval-v12-cosine-simplex`，默认输出 `results/gpu-eval-v12-cosine-simplex/`；旧结果不改写、不续接或混合。

## 目标与尺度约束

ASLS 后拼接正常 patch 输入 X，对非负相对权重 p 定义：

```text
sum(p) = 1; min_features <= ||p||_0 <= max_features
Y(p) = X diag(sqrt(p))
C(p) = row-normalized Y(p) cosine Gram
C_full = cosine Gram of full selected-layer input X
p0 = uniform weights 1/M

F_lambda(p) = 0.5 * ||C_full-C(p)||_F² / ||C_full||_F²
            + beta * ||P^T p||² / ||P^T p0||²
            + lambda * ||p||_0
```

P 来自原正常 perturbation variability（V×M 输入转置，沿用 `norm+1e-8` 按列归一化）。仅 nuisance stability，不生成异常；P 全零时 variability 项为零。非零非负 P 的 uniform 分母为正，整体缩放 P 不改变该项。beta=0.002、原 15 点 lambda grid、geometry tolerance=0.05 和 32–256 非零维预算数字保持，但目标单位与原 quadratic objective 不同。

Simplex 防止权重整体趋零；L1 在 simplex 上恒为 1，因此这里用真实 cardinality，不能声称 L1 产生了稀疏。支持集 K 维权重下限为 `1e-4/K`，即固定总 floor mass=1e-4；报告实际 K 与有效权重维数 `1/sum(p²)`，不以极小权重冒充有效维数。任何训练行的加权 norm 为零时，该候选无效；没有有效候选时明确失败，不回退为 Raw/PCA 或更复杂 detector。

## 实际求解流程

`uvarfs/cosine_uvarfs.py` 实现精确 sample-space 目标及解析梯度。缓存一份 full-reference normal Gram；默认 4096 行 FP32 reference 约 64 MiB，按 512 行分块计算候选误差/梯度，不保留 N×N autograd 图。没有改成随机几何 pairs 或降低 normal sample budget。

固定支持上用 Euclidean simplex 投影、Armijo 下降 line search 和 BB 步长。每个接受步骤必须实际降低 cosine representation + variability；cardinality 在固定 K 上为常数。单支持 refit 上限 2000、line search 最多 20 次、初始步长 0.05、残差 tolerance=1e-5，均写入配置与结果 manifest。数值失效拒绝；line search 停滞与达到迭代上限分别记录，不强称收敛。

候选生成固定为：

1. **精确原 v11 选中支持**及其归一化权重起点，再做 simplex refit/交换。
2. 按原 v11 budgeted weights 排序的起点，从 max_features 逐步剪枝到更小支持。
3. 独立 normal feature-energy 排序起点，使用相同预算与求解。

K 候选为 min_features 起每 32 维、max_features，以及原 v11 Main 实际维数；不是穷举所有 K 或支持组合。剪枝用新目标一阶删除方向估计排序，仅用于生成候选；交换用全维梯度提出至多 8 个替换，必须按新目标精确评价并下降至少 1e-8，再 refit。每个 K 最多 8 次交换。零行初始支持使用确定性 coverage repair；这是启发式，不证明所有预算组合可行/不可行。

原 v11 solver 在 `u_varfs.py` 中保持原公式和选择，其结果既是明确起点，也是独立完整对照；按有序层集合与输入模式缓存，原分支复用结果，避免重复求解。旧 Top-weight/prune/exchange/legacy 控制继续属于原目标。

每个 lambda 在生成的有限池中选 **完整新目标最小**候选；固定支持优化不依赖 lambda，因为 λ·K 在该支持上恒定。随后在这些 lambda 点中选择 actual cosine geometry 容差内最稀疏解；没有可行点时选择 actual cosine error 最小者，明确 `feasible=false`。这只是在有限生成池内的求解与选解，不是全局非凸最优证书，也不能用原 Gram 的 budget lower bound 排除新 cosine 解。

保存 loss 三分项、每个候选的支持/权重、effective dimension、simplex 残差、固定支持一阶残差、refit/交换下降历史、初始化来源和实际求解配置。`relative_change` 明确只描述交换前的 initial refit；最终是否达到驻点以实际最终支持的一阶残差为准。没有给新目标填入旧 continuous convex box gap。

## 完整同轮控制

Raw 主输入保持，L2 主输入切换仍未批准。本轮目标授权不捆绑输入协议变化。

| 前缀 / 目录 | 输入 | 主 U-VaRFS 目标 |
| --- | --- | --- |
| 无 / 根目录 | raw | 新 cosine/simplex/cardinality |
| `gram_` / `gram/` | raw | 原 quadratic Gram + v11 forward/refit |
| `layer_l2_` / `layer_l2/` | 每层 L2 | 新 cosine/simplex/cardinality |
| `layer_l2_gram_` / `layer_l2/gram/` | 每层 L2 | 原 quadratic Gram + v11 forward/refit |

每支完整保留原 38 个方法，默认 **152 方法**，全六共 912 行。Raw、PCA、五 Random feature seeds、随机层、fixed4/all U-VaRFS、pooled/prefix、旧特征算法和 legacy 都保留。每个分支 PCA/Random feature 独立匹配该分支 Main 的 K；Random-K 匹配 ASLS 层数。`objective_branch` 表示分支，`uvarfs_objective_definition` 表示该方法真实目标，显式原算法控制不会被重命名成新目标。

四支共享正常图像/patch IDs、扰动 RNG、DINO forward 和 memory 抽样；每个输入模式只拟合一次 ASLS 并复用原结果。原 normal fit=256 图×16 patches、variability=96 图、memory=1024 图×32 patches/cap20000、随机五 seeds 均保持。输入/variance/fit/memory/query 在每支一致；新相对权重在 memory/query 中用 FP32，再走原 exact cosine 1-NN、Top-1% image score 与 bilinear pixel map。

完整控制增加整套拟合与评价成本，所有 candidate/control/audit 时间都计入共享 dataset fit；不从中间候选计时宣称独立 Main 加速。DINO backbone 仍 frozen，没有异常监督或新 classifier/decoder。

## 结果、断点与分析

`scripts/run_all.py` 保存 objective identity/metric、有效维数、simplex 与固定支持残差。原 Gram 的 box gap/fixed-support floor/全局预算下界仅写原目标方法。resume 必须匹配版本、配置/代码/数据指纹、152 方法以及四支拟合文件；缺失原控制也必须重跑，不能静默缩减对照。

`scripts/analyze_results.py` 对新目标独立检查全部候选的支持、simplex/floor、scales、loss 分项、lambda 有限池最优与 normal-only 选择；核查原控制实际选中支持、共享 ASLS/数据和每支同维 baseline。新报告使用目标对应的证书解释，原结果报告仍支持。所有 image bootstrap 仅用于评价，不反馈到拟合。

## 第 19 节检查

| 检查 | 结论 |
| --- | --- |
| 研究问题 | 不变，BMAD 六数据集与 normal-only 两阶段稀疏 |
| ASLS | loss、patch 输入、全层候选、组合规则、层预算保持 |
| U-VaRFS 目标 | 按用户明确批准升级；原公式保留完整控制 |
| anomaly labels/masks 拟合 | 无 |
| detector | 原 normal memory cosine 1-NN |
| main/baseline 数据预算 | 完全相同；各支同维 PCA/Random |
| 新旧可比 | v12 同轮原目标控制；不混用旧结果或归因于输入切换 |
| 版本 | bump v12，独立目录，严格指纹与完整控制 |

## 服务器运行

本地完整回归为 **115 passed，4 skipped**；CUDA 检查因缺少 runtime 跳过。独立 autograd/有限差分核验解析梯度，SciPy 核验固定支持 simplex 解；两种 primary 配置均验证 152 方法的 fit、memory、image/pixel 评价与完整分支 resume。新候选池选解和报告字段通过检查，原求解器独立对照与历史分析兼容性检查通过，原回传文件未改写。shell/diff 检查通过。

本地可检查数学、CPU/I/O 与小型完整流程；真实 BMAD 数据、模型及 CUDA 不在本机。不能据数学例子或 fixture 宣称正向 AUROC/AUPRO 效果。

先 Liver 验证运行与协议，再同一固定设计运行六数据集。不要看 Liver test 指标再调 beta、lambda、K 或输入。

```bash
git pull --ff-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

检查 main cosine/refit、gram control、四支 manifest、候选 feasible/残差/有效维数、152 方法、memory 与 pixel metrics。所有实验结果/报告/日志继续仅留本地；代码、配置、测试和本开发交接按授权提交推送。
