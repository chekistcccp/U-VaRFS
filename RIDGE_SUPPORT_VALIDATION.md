# 岭杠杆互补坐标：固定目标的独立支持拟合与验证

用户2026-10-17明确要求继续思考改进、修改代码并继续服务器实验。本轮保持Frozen DINOv3 + ASLS + U-VaRFS医学异常检测研究；注册Main仍为第48节v22。新增显式`ridge-support-continuation-v1`分支，默认配置和原数学目标不变，不由test自动升级Main。

## 观察、候选解释和可反驳预测

完整v22及尾部候选结果保存在项目reports，非本开发文档。既有观察是：降低正常平均匹配损失、再优化正常图像尾部，尚未弥补相对同输入Raw的检测与定位缺口。

- 候选解释A：现有能量/未白化QR起点可能偏重强正常变化方向，漏掉较弱但独立的坐标；扩大这类方向的覆盖可能改善选择。**这是探索性假设，未证明其为失败原因，也未证明弱正常方向就是异常方向。**
- 竞争解释B：正常代理与未知异常分离本身失配；新增互补支持即使改善正常谱覆盖/损失，也不会带来检测改善。
- 竞争解释C：数据/实现/抽样差异制造表面改善；严格重复v22与β0锚点、来源/内存/指纹审计用来检查它。

先固定算法及全部18区组，后看新检测结果。主要比较对象始终`ridge_matching`，不能看test后改成ridge_only、β0、uniform、另一个输入或最好seed。若正常指标改变而Raw性能缺口保持，A不能被宣布成立，B仍需进一步研究，不能以更多搜索或压缩宣布成功。

## 新支持生成，原匹配目标保持

ASLS有序层、逐层L2主输入、正常192fit/64holdout、96扰动图、每图16patch、K=min(256,M)、floor=1e-4/K、cap=min(1,4/K)和最终memory/detector保持。只读取SHA一致的既有正常缓存，不额外增加正常图像或backbone预算。β0的全部新支持和权重拟合不读取nuisance值。

对每种fit-only矩阵A（正常去均值坐标、正常跨图nearest/rank8差分、按原nuisance敏感性缩放的差分），计算：

```text
C = AᵀA = V diag(e) Vᵀ
λ = max(sum(e below largest K)/K, max(e)*float64_eps*max(A.shape), float64_tiny)
Z = A V diag((e+λ)^(-1/2)) Vᵀ
ZᵀZ = C(C+λI)^(-1)
```

Z各列的平方范数为ridge column leverage。对Z执行固定K贪心投影：设当前残差列Gram为G，每步选使`sum_i G_ij²/G_jj`最大的未选列j，再做`G <- G - G_:j G_j:/G_jj`。该数值是单列正交投影带来的实际总残差能量下降，区别于只按该列自身能量的普通QR支点；不保证全局K最优。返回**原始latent坐标索引**。白化与去均值只用于拟合期支持提案；推理不进行PCA旋转、白化或decoder重建。秩不足确定性补满K；零稀疏行候选无效。正常诊断/实现修正不访问新test结果，具体证据保存项目reports。

λ只由fit谱与原K确定，没有test/holdout超参扫描。采用[岭杠杆列选择](https://arxiv.org/abs/1511.07263)的谱动机；本项目固定K支点启发式及AD任务**不继承该论文随机采样/近似保证**。相关确定性列选择见[McCurdy, NeurIPS 2018](https://arxiv.org/abs/1803.06010)，本实现并非该文算法的复现，不能声称新颖性或理论AD保证。

新支持各保留uniform参考，再用**原MatchingObjective** refit150步、同约束SLSQP最多200步/ftol1e-12；仅真实原训练目标下降时添加精修点。不新增损失项，不删除1-NN/KL/nuisance，不提高压缩，不改变温度或原损失标定。新支持可去重，旧完整v22候选池原样保留，并在使用前复现其正常training/holdout目标。全部新生成/精修先完成并哈希，之后才构造64图holdout验证器；正常候选选择仍为原holdout匹配损失加fit nuisance。

## 配对分支与对照

每区组评价10个预先指定表示：

- `ridge_matching`：旧完整v22池加新ridge池，按原正常规则选择，**本轮主要对象**。
- `ridge_only`：只从新增有效ridge候选按同规则选择；无有效新候选明确回退并记录，用来区分提案与选择规则的作用。
- 上述两者各自β0，以及四者各自相同支持uniform，共8个新表示。
- `v22_anchor`、`v22_beta_zero_anchor`重复原表示，共10个新评价/区组。

原352方法/区组保持原版本、全部原指标/预测/成本与来源，6336行单独controls保存；上一尾部6表示/区组、108行亦单独保留。两项锚点逐图预测必须完全相等、适用image/pixel指标差<=1e-12、正常memory图像/patch/reservoir/容量均一致。失败保留日志、定位工程问题，不能放宽门槛或按异常成绩修选择器。

BMAD全六、来源fit/memory seeds42/123/3407、18配对区组，物理GPU0–3。新增支持与权重确实重新拟合；ASLS/缓存/抽样来自已有三个独立来源fit，不冒充额外三次独立抽样或新患者验证。历史test多轮用于研发、64正常留出曾用于ASLS等限制保持。

## 执行与审计

1. 本地独立SVD/逆矩阵oracle、弱独立方向反例、秩不足/尺度等值、原池保护、β0隔离、选择前不读取test表、10方法worker与全六/无mask分析测试。
2. 提交同步GitHub，安全FF部署；保存并核对nohup.sh、服务器未提交文件及原数据/结果，不安装依赖。
3. GPU0先完成真实Liver normal-only工程验证，不查看新test分数来决定后续配置；再冻结全部源码/配置/脚本SHA，四卡正式18区组。
4. `scripts.run_ridge_support_validation`调度、`scripts.evaluate_ridge_support`拟合/评价、`scripts.analyze_ridge_support_validation`严格全量审计与2000次条件图像bootstrap。失败停止后续分派，已活跃任务安全退出。
5. 全部完成后完整快照、整包与每文件SHA主动取回、本机独立审计、两端报告/索引同步。所有新实验产物实际位于当前项目results/reports/logs，Git忽略；被引用的原正常缓存已在两端项目且完整SHA回传，不能删改。认证/必要GPU锁和调度控制继续单独保存，遵守[存储限制](EXPERIMENT_STORAGE.md)。

报告主要对象相对同输入Raw/PCA/五Random、β0、uniform、v22/v21/v20、上一尾部方法的全六图像/三mask定位均值与样本SD及逐集方向。Random先在fit内部平均；n=3不是患者数。图像bootstrap条件于单fit，多比较未校正，没有像素地图/患者CI。完整拟合尝试/无效提案/阶段开销均记录，不能将复用旧昂贵候选的增量时间当端到端训练成本。

## 第19节变更检查

研究问题、ASLS、U-VaRFS数学目标、normal-only、detector、统一数据与256维预算均不变。新增支持生成算法与数值轨迹，使用独立身份/目录并保留所有原控制；不混合旧结果，不改默认Main/EXPERIMENT_VERSION。用户本轮明确改进/修改代码/继续实验的授权适用。正常谱覆盖、低loss或工程通过均不证明真实性能成功。

## 技能来源

Hypothesis Generation与Experimental Design用于候选/竞争解释、配对单位及预设比较，未自动给假设评分或把假设当发现。Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)，2026-10-17核验当前arXiv记录，未发现期刊替代记录。技能的使用不是性能或科学有效性证明。
