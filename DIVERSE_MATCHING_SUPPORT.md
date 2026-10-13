# v22：同一正常匹配目标的互补稀疏支持搜索

用户要求分析完整实验、按照原研究目标优化并继续实验。当前Main仍是Frozen DINOv3 + ASLS + U-VaRFS的无异常标签医学异常检测；本轮只扩大同一目标的有限支持搜索并做数值精修。数学目标、输入、数据、K256和detector保持[v21定义](NORMAL_MATCHING_DISTRIBUTION.md)。实验成绩、诊断与分析保存到项目reports，不进入Git。

## 固定目标与预算

真实student normal cosine 1-NN分数保持、全部跨图normal reference概率KL、原β=.002 nuisance；temperature、teacher、归一尺度与查询权重均由正常fit按v21定义计算。ASLS原2–6层全部组合，主输入逐层L2，raw完整配对控制。192正常fit/64正常holdout、96扰动图像、每图16patch、memory20000与原共享抽样保持。K=min(256,M)，sum(p)=1，floor=1e-4/K、cap=min(1,4/K)。异常标签与mask只用于评价。

## 三个互补拟合起点

原v21局部贡献、能量、稳定局部贡献三个起点及其全部uniform/refit/交换候选保持。增加三种原始坐标起点：

1. 正常fit特征去拟合均值后的列残差能量支点选择。
2. 正常fit patch与跨图最近/rank8邻居的差向量，按原query权重加权后选择互补列。
3. 上一差向量除以sqrt(1+原fit逐维nuisance敏感性)，再选择互补列。β=0完全不读取扰动。

每步选择残差能量最大列，再投影去除该列方向；仅生成原始latent坐标索引，推理不旋转、不中心化。秩不足时按原列能量确定性补满K。不强制层配额，不按test改变层、输入或规则。独立SciPy pivoted QR作数值oracle，参见[官方列支点分解说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html)；实际执行采用Torch列投影。

六起点均使用原150步projected BB/refit、两轮每轮四个全坐标交换提案。原v21子池先哈希并保留；不以正常holdout选中的旧支持作为新搜索起点。扩大后的训练池全部候选继续保留。

## 同一固定支持目标的数值精修

每个唯一支持按拟合目标选已有最佳权重，以mass=K*p运行SLSQP最多200步、ftol=1e-12。边界为floor_mass至min(K,cap_factor)，sum(mass)=K，梯度除以K；同一目标与可行集仅做变量缩放。参见[SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)。可行点独立投影/评价，真实训练目标严格下降才增加候选，原候选保持。保存全部尝试，包括无改进、optimizer状态与真实成本；非有限末端不得覆盖原点。

起点、交换、精修及warm start仅访问fit；完整训练池哈希后才构造holdout validator，沿用原正常holdout匹配损失加训练nuisance的有限池选择。只报告当前最近分支投影残差；max切换非光滑，SLSQP成功不证明非凸/离散全局最优。正常代理改善不证明未知异常分离或检测收益。

## 控制与独立实验

`main`采用扩大搜索。完整v21在相同fit/memory上保留为`v21_*`和`raw_input_v21_*`，关闭新起点/精修；`configs/v21_control.yaml`保存原配置。全部v20 `local_*`、Gram、历史目标、同输入Raw/PCA/五Random、β0/无score/无distribution及同支持uniform保持。默认352方法/数据集，独立fit/memory种子42/123/3407，全六6336结果；五Random不是五次Main重复。

版本`gpu-eval-v22-diverse-matching-support`，新根直接在两端当前项目内，不覆盖/混合旧结果。四卡dataset worker先验证工程与协议，再固定配置自动完成18任务；不能按Liver或其他test成绩改后续任务。报告mean/样本SD、同输入Raw/PCA/Random/β0、完整v21/v20和定位差异、全部拟合/评价成本及局限。成功依据真实检测/定位改善；压缩、低正常loss或提速不能补偿下降。启动核验一次，后续预计窗口及数小时间隔检查；完整SHA回传项目。

## 第19节检查

研究问题、ASLS、U-VaRFS数学目标、标签约束、detector与统一数据/维度预算不变；支持搜索/数值轨迹变化，故升级版本/目录并保留完整旧控制。用户最新优化/继续实验指令及第20节工程授权适用；没有新研究目标、层配额、PCA Main或监督训练。代码/配置/限制规则检查后提交同步；实验产物Git忽略。
