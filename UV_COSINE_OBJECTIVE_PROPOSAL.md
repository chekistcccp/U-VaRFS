# U-VaRFS 的 cosine 几何目标升级提案（待批准）

项目仍为 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究**，最终保留 BMAD 六数据集。此文件是一份可审核的方法升级提案，不是已经启用的新主方法。

当前 `main` 继续使用 v11 的原 quadratic Gram objective 与 `objective_forward_refit`。本次只新增只读评审脚本和数学检查，未接入 fit、memory、query、lambda 选择或默认配置。v11 真实性能尚须回传。最新回传数据及其分析均只留本地，不将实验结果写入本文件或同步仓库。

## 要修正的问题

原损失保留 `X diag(w) X^T`，detector 则对 `X diag(sqrt(w))` 的每行重新归一化后做 cosine 1-NN。原 geometry tolerance 控制幅度与方向的混合误差，不能直接解释为 detector 的几何保证。删去完全重复的列也可能违反原幅度容差，同时仍精确保留所有 cosine。

这是目标与 detector 的错配，不是已证明的所有性能下降原因。真实数据上还需检查 nuisance stability 是否排除了有用维度、normal fit 的泛化、近邻排序以及 localization。对原目标求得更低的 loss 并不推出 anomaly AUROC/AUPRO 更高。

**不能只把 representation loss 换成 cosine，沿用原 L1/variability。** 对任意 `a>0`，`C(a*w)=C(w)`；而原惩罚随 a²/a 缩小。权重可趋近全零以降低目标，零向量的 cosine 又未定义。也不能在 simplex 上用 L1 宣称稀疏：`sum(p)=1` 时 L1 恒为 1。

## 拟议数学定义

ASLS 仍先选层，X 为对应的 normal patch 特征拼接。定义非负相对权重 p，选出的表示及其实际 cosine Gram 为：

```text
p >= 0, sum(p) = 1
Y(p) = X diag(sqrt(p))
C(p)_ij = <Y_i, Y_j> / (||Y_i|| ||Y_j||)
C_full = cosine_gram(X)
```

每个训练行的 `||Y_i||` 必须大于零；零行候选明确无效，不能靠 epsilon 宣称保持几何。统一特征预算仍为 `min_features <= ||p||_0 <= max_features`（默认 32–256，小型 fixture 按候选维数截断）。Simplex 固定整体权重尺度，detector 对该公共尺度本来不敏感。

沿用原 normal-only mild perturbations 得到的非负 variability 矩阵 P（M×V，保留原按列归一化）。参考相对权重 `p0=1/M`。建议使用：

```text
V(p) = ||P^T p||² / ||P^T p0||²

F_lambda(p) = 0.5 * ||C_full - C(p)||_F² / ||C_full||_F²
              + beta * V(p)
              + lambda * ||p||_0
```

P 全零时 `V=0`；非负 P 只要非零，uniform baseline 分母严格为正。按基准归一化时整体 rescale P 不改变结果，避免人为 epsilon 成为稳定性系数。初始 beta 与 lambda grid 数值保留现有设置，但**损失的单位已经变化，不能声称与原目标等价**。不根据 test AUROC 重新标定 beta/lambda。

这仍由 `normal representation preservation + nuisance variability regularization + latent sparsity` 构成；变动的是 U-VaRFS 的精确数学定义和求解，不能写成工程加速。若批准，论文须给出新目标，原 quadratic U-VaRFS 必须保留为完整消融。

## 批准后的求解与选解边界

拟在固定支持上用 simplex 投影与下降 line search 优化实际 cosine/stability，再用预算内支持删除和交换生成稀疏候选。参考 Gram 只使用既有 normal fit patches，按行分块精确计算，不增加图像、patch 或 memory 预算，不引入异常样本。全部新版 U-VaRFS 对照使用同一候选与迭代预算。

保留原 v11 支持集的归一化权重作为明确初始化，并加入独立正常能量起点；不因它们的 test 指标选择起点。固定支持上 cardinality 不变，但删除/交换/比较不同支持大小时必须评价完整新目标，不把 L1 当作 L0，也不把原 FISTA 的 convex box gap 当新目标证书。

正权重下限、候选支持大小的生成规则及数值停止规则在首次实验前写入配置和 manifest。保持已固定的 32–256 上限，报告实际非零维数以及 `1/sum(p²)` 的有效权重维数，避免用极小权重充数。所有接受候选须有限、满足 simplex/预算并没有零行；没有满足条件的解时报告失败，不能静默退回另一个主方法。

新目标为非凸且包含离散支持，不保证全局最优；记录实际损失分项、约束残差、固定支持一阶残差、迭代上限与初始化来源。梯度须独立有限差分和 sample-space oracle 核验。固定支持驻点不代表全局稀疏最优。

沿用 label-free lambda 选择原则：在预设 **actual cosine geometry** tolerance 下取最稀疏可行解；没有可行解则取实际 cosine geometry error 最小的 fallback，明确 `feasible=false`。原容差数字可以保留为初始协议值，但度量已变，不能将新旧 feasible 混为一谈。不可通过异常 labels/masks 选择参数或版本。

## 同轮对照与验收

本提案**不捆绑 L2 主输入切换**：当前 raw Main 和完整 L2 消融继续分别固定，先在相同输入、相同 ASLS 层集合上判断目标升级。L2 Main 的既有待批准状态保持。

若批准目标升级，则在新的 experiment version 与独立目录启用，保留原目标和 v11 稀疏求解控制。Raw/PCA/Random feature、全层 U-VaRFS、固定层和五随机 seeds 保留；PCA/Random feature 必须额外匹配新变体的实际维数，不能沿用旧 Main 维数冒充公平比较。Frozen backbone、ASLS 定义/层预算、perturbations、normal memory、cosine 1-NN、Top-1% 与 pixel map/evaluation 均保持。

先 Liver 做运行和协议验证，再固定设计跑全部六数据集。不要在看 Liver test AUROC 后改设计，再把同轮其余数据集当成未见测试。全部数据集使用相同 preset 设置。

“正向效果”需要分开验证：

1. 与同输入、同层、同维 PCA/Random 比，特征选择是否有收益；随机结果报告五 seeds mean ± SD。
2. 与原目标比，是否改善 image 与 pixel 指标，报告全部六数据集的差值及区间。
3. 与相同层 Raw 比，是否得到有价值的压缩—性能折中，同时报告维数、memory/matching 开销；未预设非劣界值就不宣称 non-inferior。

不能要求、保证或事后筛选六个数据集全部提升。正常几何改善是机制证据，真实 positive AD effect 必须由固定设计下的 BMAD 结果验证。

## 本次已提供的可复核内容

`scripts/review_uvarfs_objective.py`：

- 只读正常 fit 诊断，明确区分原 weighted Gram 与 actual cosine；不读取 AUROC/prediction labels/masks 来选方法。
- 计算重复列幅度错配与 naive cosine 权重缩放退化的数学例子。
- 为给定 p 计算本提案的候选 loss，**没有 optimizer、feature selector 或接入训练入口**。
- 验证返回支持/scales 与作用域；读取前后核查原始回传文件 SHA256，报告仅写新本地目录。

数学例子不冒充 BMAD 实验，不证明异常检测收益。旧回传未保存原始 fit feature 矩阵时，已返回的正常 Gram 诊断只能按来源引用，不能宣称独立重算。

本轮全套本地检查为 102 passed、3 个 CUDA 检查因无 runtime 跳过；其中 10 项为新增评审检查。独立 scalar oracle 核验实际 cosine 与 variability 公式，覆盖尺度约束、cardinality、无效零行、来源/覆盖保护和实验指纹隔离。原实验代码、配置和 runner 与本轮修改前逐文件核对一致，数学评审文件不进入训练代码指纹。

```bash
python scripts/review_uvarfs_objective.py \
  --results results/gpu-eval-v10-layer-normalization-ablation \
  --out reports/uvarfs-objective-review
```

## 第 19 节与批准范围

| 检查 | 本次交付 | 若批准后实施 |
| --- | --- | --- |
| 研究问题 | 不变 | 不变 |
| ASLS 定义 | 不变 | 不变 |
| U-VaRFS 目标 | main 不变，仅数学评审 | 明确升级为 cosine + simplex + cardinality |
| anomaly labels/masks 拟合 | 无 | 无 |
| detector | 不变 | 不变 |
| 数据预算 | 无新拟合 | 维持所有方法统一原预算 |
| 可比性 | 原实验指纹/预测入口不变 | 同轮原目标与同维 baseline 控制 |
| experiment version | 仍 v11，评审独立输出 | 必须 bump 新版并使用独立结果目录 |

AGENTS 第 5.2 节把原公式锁定为主线，第 19 节明确：**“如果 2–7 任一答案为‘是’，除非用户明确要求改变实验设计，否则不要直接实施。”** 因此本提案等待用户明确批准上述 U-VaRFS 目标升级后再接入主训练流程；原有 patch/支持集求解授权仍持续有效，无需重复批准。
