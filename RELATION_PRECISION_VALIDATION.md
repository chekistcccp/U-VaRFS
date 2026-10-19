# 正常关系固定支持精度与效能归因验证

用户于2026-10-19批准依据最新复核修改代码、同步并在服务器继续研究。先实施已建议的受控数值验证；不直接切换新loss、支持规则或注册Main。独立身份`clean-relation-fixed-support-precision-v1`，配置`configs/relation_precision.yaml`。

## 固定对象与问题

来源为完整封闭的`relation-validation-20261018T175358Z`；保留其clean可靠多正图、fit-only温度、.002实际nuisance、原始坐标支持、逐层L2、ASLS、Frozen DINOv3 S+和正常144fit/48selection/64audit、96扰动×16patch。只访问144fit及原β>0扰动，重建图必须与两β父图逐SHA、全部元数据一致；β0不读取扰动。48/64特征不再用于选支持或权重。保留K256、floor=1e-4/K、cap=4/K、memory20000/cosine1NN/Top1%。不增加数据、不重跑支持搜索，不把三组来源fit/memory种子称为新独立抽样或患者。

问题是：同样800步未达到同样数值精度，能否通过有限、统一、同原目标的固定支持精修，使主支持、随机和简单选择的比较更可解释？相同局部残差不是相同泛化误差、全局最优或检测证书。此前已达局部标准仍退化的OCT结果完整保留，不能预设本轮必然成功。

## 19项权重与统一预算

每个区组固定19项：主支持及其独立搜索β0支持2项、同主支持β0一项、dense两β2项、五Random两β10项、Laplacian与repeatability两β4项。各自从父800步权重继续，不能按当前test或新正常留出重选起点。六数据集×来源fit/memory seeds42/123/3407共18区组、342正常拟合。所有旧738关系评价、机制630和更早6336/尾部108/岭180/pursuit180保留来源与成本。

使用已验证的可行有限记忆secant/投影梯度及实际Armijo，精确变量换元mass=K*p，梯度按g_p/K转换。原图关系概率+nuisance数学目标不变。原p单位投影残差`||p−Proj(p−g_p)||∞<=1e-5`；每项最多3000个接受迭代、12000次solver实际loss-gradient调用，历史12、每方向线搜索24、Armijo1e-4、curvature floor1e-10。父loss重现、同支持uniform参考和独立最终gradient各额外一次，最多12003实际调用，分别记录。实际loss严格下降才接受；初值保留，不更改支持或floor/cap。记录全部接受loss/残差轨迹、拒绝数、时间、调用与停止原因。图重建、读取、校验与阶段总耗时另外保存。

原solver代码`uvarfs/precision_refit.py`原样复用，新包装明确这是**光滑固定图固定支持**目标；旧hardmax精度实验不被追认通过。额外报告uniform参考、父权重与同支持β0，区分支持、权重和nuisance；dense与sparse cap不同，不宣称纯删除因果效应。

## 先全局正常门控，再条件检测

先Liver42正常工程门控，属于正式18组；其数值未通过不算程序失败，也不改变其余组。其后物理GPU0–3并行剩余17组，不抢占他人GPU。全部18正常结果冻结并逐文件SHA后，一次全局门控：342/342均达标才新做正常全网格审计和全部检测。任何未达标则终态`precision_inconclusive`、exit0，完整报告，不跳过困难控制、不放宽阈值或自动重试，不解析本轮/旧轮test成绩决定下一步。

通过后每区组22个表示：19新权重与父relation_sparse、父dense、Raw三个重复锚点，共396行。三个锚点逐图完全相等、全部适用指标误差<=1e-12及memory抽样一致后，才引用父738及其历史控制。64audit query从audit memory全部排除。先Liver工程验证，再固定其余17组；不得依Liver测试成绩改变计划。

唯一主比较为`precision_sparse`；报告与Raw/PCA、精修五Random均值/SD、精修dense、简单选择、同支持β0、父主支持/其uniform及全部父控制的全六Image AUROC/AP、三mask Pixel AUROC/AP/AUPRO。三个来源fit样本SD和五Random within-fit SD分开；2000次分层图像bootstrap条件于单fit，不是患者或跨fitCI，未做多比较校正和完整像素地图重采样。独立原始梯度复算全部342残差、父权重SHA、支持、单调性与调用成本；正常门控未过时只报告正常数值与局限。历史test和正常图已暴露、外部完整方法尚未复现，不能由低loss/精度/工程通过宣布方法有效。

## 执行、存储与停止

新入口：`python -m scripts.run_relation_precision --parent-batch <封闭关系批次> --cache-batch <封闭正常缓存> --run-root results/remote-experiments/relation-precision-<时间> --expected-head <冻结提交> --control-dir <外置必要GPU锁>`。不修改原nohup.sh或服务器未提交文件、不安装依赖、不在活动期间pull或改源码/config。所有产物在两端当前项目results/logs/reports且Git忽略；认证和必要锁/控制指针外置。完整终态、exit0且原身份全部退出后，完整封闭快照、主动回传、逐成员SHA/安全路径核验和独立审计，双端报告/索引闭环后暂停本批监控。注册Main仍第48节，长期目标未证明不标成功，不自动启动另一轮。

第19节检查：研究问题、ASLS、关系数学目标、标签约束、detector、数据和K预算不变；固定支持求解方式和计算上限明确改变，用户本次已授权。新evaluation_version和独立目录避免与父结果混合，完整记录新增计算成本；默认EXPERIMENT_VERSION和旧科学源码不改。若本轮仍不支持性能收益，按复核报告重新审查关系选择依据，不继续无停止规则的测试集驱动调参。

技能程序参考：Kassis et al. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。本轮使用experimental-design v1.5；程序规范不是效能证据。
