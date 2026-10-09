# v18：固定预算支持交换先重拟合再判断

项目仍为 Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究，最终是 BMAD 全六。用户要求以检测性能为改进依据，更强压缩不能抵偿性能下降。当前版本 `gpu-eval-v18-refitted-support-exchange`，独立结果目录同名；回传成绩、分析、图表与日志只留本地。

## 方法边界

Main 仍是 [PERFORMANCE_FIRST.md](PERFORMANCE_FIRST.md) 已批准的固定预算 capped cosine/simplex 目标：K=min(256,M)，sum(p)=1，floor/K≤p≤4/K，原正常 cosine/variability 项、beta=0.002、geometry_tolerance=0.05 不变。没有新的loss、异常标签、病灶mask、test调参、主输入切换或更强压缩。

固定支持收敛不等于支持搜索已经充分。旧交换规则把被移除维的权重原样转给新维，仅当这个未重拟合初值已有下降时才调用固定支持求解。它可能提前拒绝最终能够下降的支持。正常独立小型矩阵可复现这一现象；这证明算法有漏解可能，不把它归因为所有真实检测差异的唯一原因。

## 新求解顺序

1. 完整执行原 v17 求解，保留三个起点的 uniform reference 和 optimized 端点。旧 finite-pool 选解原样作为同轮 `asls_fixed_budget_uvarfs` 控制，不受新交换筛选影响。
2. 从各 optimized 端点出发，使用原完整 normal cosine/variability 梯度提出固定 K 的替换。提案排名仍是原 mass-transfer 一阶预测；每步统一8提案、每条最多8步，不穷举所有支持，也没有离散全局证书。
3. 对每个有效支持先按原 capped simplex、原 refit/SLSQP 与下降端点保护重拟合，再按相同正常完整目标判断。保留默认 min improvement=1e-8。零行提案无效，不能成为 detector fallback。
4. 在这一步的提案中选择重拟合后目标最低且严格下降的点。若父状态满足原几何容差，接受点也必须满足；不牺牲已有正常几何可行性。未接受时保留父状态。
5. 将有接受步骤的最终端点追加到原 v17 池，仍先筛几何可行候选再按 Phi/error/ID 选择；无可行候选仍取最小error并明确 `feasible=false`。

保留完整旧池意味着已有可行参考不会因新搜索丢失。在同一正常目标、数据、工作精度下，旧点可行时，新有限池的选解目标不会高于旧点；无可行旧点时保留原 fallback。该保护不保证异常AUROC/AUPRO不下降，也不证明样本外稳定性、支持全局最优或更低正常loss就意味着更强检测性能。

## 成本与可复核记录

新增 `exchange_search`，记录每个父候选、预设提案排名、移除/加入维、重拟合前后完整loss、几何、eligible/accepted、原refit/polish diagnostics、接受支持/相对系数与停止原因。拒绝提案的全部refit迭代、polish次数、目标评价与时间都包含在共享fit成本，不能只报告已接受部分。增加正常求解计算，不增加fit/variability/backbone forward/memory数据。

固定支持残差仍按 capped feasible set 计算，只是非凸固定支持的一阶诊断。交换每步接受严格下降的端点，SLSQP内部试探点仍不作为单调轨迹；optimizer success仍不等于最终残差收敛。

分析器独立核验固定K/cap、原候选池完整保留、同目标分项、接受支持的精确单维变化与几何保护、停止拒绝、全部拒绝refit成本和最终有限池选解；兼容无新字段的旧回传。不含原始fit矩阵时，审计不能重算服务器实际normal Gram或每个提案loss，其作用边界如实保留。

## 完整实验对照

原 v17 fixed-budget Main：`asls_fixed_budget_uvarfs`；各支保存 `uvarfs_asls_fixed_budget.json`，同normal输入的performance/Gram分支共享该求解缓存。Main已有old计算也复用缓存，不重复DINO提取。

旧 cosine/cardinality/lambda Main：`asls_compression_uvarfs`；完整 original quadratic/Gram、所有历史baseline、五随机seed、PCA/Random同K、Main同支持去权重对照全部保持。raw/L2×performance/Gram四支默认 **164方法，全六984行**。新控制在Gram分支也如实记录真实PERFORMANCE身份，不伪装成Gram。

ASLS patch输入、sigmoid/Adam/geometry-search定义、Frozen DINOv3全部层候选、raw Main/L2消融、normal-reference无异常标签约束、fit/扰动/memory/维度预算、cosine1NN/Top1%/pixel插值与评价定义保持。ASLS或normal geometry代理仍可能存在真实检测缺口；本轮不据test选层、选主输入或调beta/cap。

## 第19节检查

1. 研究问题不变，检测性能仍为首要依据。
2. ASLS定义不变。
3. U-VaRFS目标与固定K/cap/geometry有限池选择不变，只扩大同目标求解候选；这是求解算法改进，不是全局等价结果承诺。
4. 不引入异常标签、masks或test scores到拟合、支持、权重和参数选择。
5. detector不变。
6. normal数据、DINO forwards、memory与表示上限不变，各支同维对照保持；增加的求解时间完整报告。
7. Main数值权重/支持可能变化，旧v17同轮保留用于直接比较，不把旧目录当作新版结果。
8. 新版本/目录/指纹，resume还须四支v17控制文件；禁止覆盖、混合或续接旧回传。

## 验证与运行

必要检查包括正常有限交换反例、先refit后比较、完整旧池/原控制保护、严格几何约束、全支持/禁用/零行、被拒绝求解成本、报告/断点/篡改拒绝、两种primary输入164方法的正常清单/memory/image/pixel完整流程与历史回归。检查结果见AGENTS第43节。

本机没有真实BMAD、模型或CUDA，不能将本地测试或更低正常loss解释为真实检测提升。服务器先验证Liver运行，再以统一预设设计完成全六；不按Liver测试成绩调参数：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

代码/配置/测试/开发交接按授权提交推送；实验产物默认只留本地。
