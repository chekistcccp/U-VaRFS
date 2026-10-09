# v17：性能优先的固定预算 U-VaRFS

用户于 2026-10-09 明确要求：“不要使用更强的压缩，就需要更强的性能，依此为依据进行改进”。本轮依据该要求改变 Main 的 feature sparsity 取舍；不是仅增加诊断，也不是等价提速。版本 `gpu-eval-v17-performance-first`，独立目录 `results/gpu-eval-v17-performance-first/`。

本文件记录方法/工程协议，不上传回传成绩、分析或图表。性能是当前首要成功判据；更低 K、正常训练目标下降、正常几何可行均不能替代异常检测收益。

## 当前精确定义

保持 ASLS 选中层、raw 输入、normal patches X、cosine reference 和原 variability 标定。令 `K=min(max_features,M)`，默认 max_features=256，支持 `|S|=K`，定义：

```text
C(p) = cosine_gram(X diag(sqrt(p)))
p0 = uniform full-input weights
Phi(p) = 0.5 ||C_full-C(p)||_F² / ||C_full||_F²
         + beta ||P^T p||² / ||P^T p0||²

min Phi(p)
subject to sum(p)=1, p_j=0 outside S, |S|=K,
           floor_mass/K <= p_j <= min(1,cap_factor/K) on S,
           actual cosine relative error <= geometry_tolerance
```

beta=0.002、floor_mass=1e-4、geometry_tolerance=0.05 保持；全六统一 cap_factor=4。P 全零时 variability 为零，零行候选无效。硬固定 K 是稀疏预算；不再把维数减少作为优化收益，不再使用 λ·K 在不同 K 之间选解。lambda_grid 保留给旧控制；新结果的 lambda=0、sparsity_term=0 和单个兼容存储点仅用于统一输出，明确 `lambda_optimization_scope=not_used_fixed_budget`。

cap 限制每个相对系数至均匀权重的 4 倍：由 `sum(p²)<=max(p)sum(p)<=4/K` 得 `1/sum(p²)>=K/4`。默认 K256 下有效权重维数至少64。这是系数集中约束，不是独立信息维数、实际坐标数或性能保证。没有给 detector 添加新 head、classifier 或训练参数。

## 正常求解与有限候选规则

- 原 CosineObjective 与解析梯度、P 的列归一化/基准、row scaling 不变。
- capped simplex 使用 2K 个 clamp breakpoints 的分段线性根，fixed-support 投影下降与 SLSQP 同时使用 floor/upper cap。下降/端点接受规则保持；实际残差针对 capped feasible set 计算，不借用无上限 simplex 的残差或旧凸证书。
- 三个正常初始化：原 Gram 权重排序、正常 feature energy、原 Gram 已选支持扩展至 K。覆盖修复与交换都保持 K，不生成较小维数，不按 test 排序。
- 每个起点保存未精修 uniform reference 与优化/交换候选，防止正常 variability 精修后更差的几何把已有可行参考丢掉。正常目标优化不保证异常收益。
- 有限池先筛实际 cosine error≤原容差的候选，再比较 Phi、geometry error、candidate ID；无可行候选时明确取池中最小 geometry error fallback，`feasible=false`。这实现新约束下的有限池选择，不是原 lambda 规则的等价改写；不证明离散全局最优。
- 可行性作用域 `fixed_budget_generated_pool`；未做 lambda 扫描，lambda-path 可行点统计为空。旧控制仍是 prescribed_lambda_path，不混用作用域。

## 完整对照与实验公平性

旧 v15/v16 cosine/cardinality Main 的原目标、K 路径、lambda grid、pruning/refit/polish/exchange 与选解规则原样作为 `asls_compression_uvarfs` 保留。各支 `uvarfs_asls_compression.json` 记录真实 COSINE 身份；normal input 相同的两目标支复用该控制。该控制不受新 cap 和固定 K 约束。

原 quadratic/FISTA/v11 保持完整 gram_* 两支。raw/L2 × performance/Gram 四支各保留既有全部方法，新增旧 cosine Main 控制；默认 **160 方法，全六960行**。显式 Top-weight/prune/exchange/legacy 仍是 quadratic 目标；旧 cosine 控制即使在 gram 分支，也在 CSV 记录实际 COSINE 目标，不伪装成 Gram。

Main 的 PCA、Random feature 五 seeds 和 selected_raw 同支持控制匹配各支 Main 实际 K；不同目标的已选 K 可以不同。相同正常 fit images/patches/perturbations/DINO forwards、ASLS、memory sampling/capacity、evaluation 与 masks 协议保持。每个输入共享一份正常抽样与 ASLS，不新增 Main 数据。整套成本包括完整控制，不能当作单 Main 成本。

Frozen DINOv3、全部层候选、ASLS、normal-only training、BMAD 六、cosine 1-NN/Top-1%/pixel interpolation 均保持。raw Main 固定；本轮性能优先授权不表示按历史 test 挑 L2 Main。

## 第19节检查与版本

1. 研究问题不变；性能优先取代更强压缩作为当前第一成功判据。
2. ASLS 不变。
3. 变化的是 U-VaRFS 的固定 K、capped feasible set、取消 Main λ·K 跨预算选择、正常几何约束的有限池筛选；这是用户当前要求支持的方法调整，明确记录为新目标 identity。
4. 无异常标签/masks/test scores 进入拟合、支持、权重、参数选择。
5. detector 不变。
6. 原正常数据/feature上限/memory预算不变；新 Main 使用原上限，所有同支同维对照匹配。
7. Main 表示与性能会改变，旧结果不视为新结果；旧压缩控制与新 Main 同轮比较。全部指标与共享开销如实报告。
8. bump 版本、新目录与指纹，resume 必须有完整四支及新增旧 cosine 控制文件；不得覆盖、混合或续接 v16。

## 验证与运行

本地检查包括独立 NumPy bisection/SciPy projection oracle、cap/有效维数边界、稳定性项导致集中权重的正常反例、固定预算与可行参考保护、无可行/零行 fallback、四支160方法的 fit/memory/image/pixel/同维控制、正常数据共享、历史控制、报告及 resume。完整检查数见 AGENTS 第42节。

本机缺真实 BMAD/模型/CUDA，不能声称已提高真实 AUROC/AUPRC/AUPRO。服务器先验证 Liver 运行，再固定配置全六；不按 Liver test 调参数。新参数统一预设，不根据各数据集 test 个别挑选。

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

以检测性能判断本轮改进是否成功，不能用 K、更低训练损失或该版本 toy fixture 成绩抵偿异常指标下降。代码/配置/测试/开发交接按默认授权提交推送，实验产物依第28节仅留本地。
