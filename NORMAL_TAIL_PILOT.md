# 正常图像尾部保真：独立候选选择试验

这是用户于2026-10-16授权改进后增加的**可选研究分支**。当前注册Main仍按AGENTS第48节；本试验没有修改default配置、原目标、ASLS、候选生成、权重求解或detector，不触发服务器部署。只有显式运行本工具才产生独立产物。新增选择规则属于待验证的科学设计候选，不称为等价提速，也不预先声称检测提升。

## 要检验的假设

平均正常匹配误差可能掩盖个别图像的较大分数失真。检验：在正常平均匹配损失的不确定性范围内，改选能够同时改善平均和最差图像尾部分数保真的固定256维候选，是否有助于后续检测/定位。

它只修正“平均patch拟合与逐图尾部聚合不一致”这一问题。它**没有解决**正常数据无法识别所有未知异常方向的不可辨识性，不假定医学语义或病灶信息可由正常loss保证。不能以本试验正常代理改善宣布成功。

## 固定选择规则

1. 读取已经保存的完整fit-only matching候选池。校验生成池SHA、原平均准则winner、每个候选的固定K、simplex/floor/cap及正常来源。不生成新支持、不更新权重、不增加DINO forward。K仍为原256上限，候选池包含原uniform与全部拒绝/接受记录。
2. 在原正常留出query对原fit reference上，重新计算dense和每个student自己的cosine 1-NN；跨图reference排除规则保持。逐query重算score/KL损失，必须与原记录吻合，否则拒绝缓存/结果错配。
3. 对每张正常图像的已采样patch，分别计算dense与student最高1%分数的均值。两者各自取Top-K，不能固定teacher的Top-K位置来代替student。K采用原detector的`max(1,ceil(P*.01))`。
4. 图像误差为两种Top-K均分之差的平方，归一尺度只来自正常fit图像的dense Top-K分数平方均值，下限1e-4。
5. 计算平均图像误差及最差25%图像的CVaR；样本数不整除时，边界图像按剩余质量加权。保持平均图像误差和CVaR均不劣于原mean winner。
6. 候选的原`heldout score+KL+training nuisance`不得超过原mean winner加一个正常**图像cluster SE**。SE按图像聚合patch残差，支持不等patch数；不是把patch当独立样本。该one-SE宽容带是预先固定的选择启发式，不是置信区间或非劣界。
7. 合格池内最小化CVaR，以原平均选择分数及candidate ID打破并列。没有严格尾部改善时返回原winner，不为了得到“新方法”强行换支持。所有候选诊断及成本完整保存。

默认固定Top-1%、CVaR25%、one-SE系数1、chunk256，不扫描参数，不按test挑输入或分支。旧mean winner完整输出，可直接作为同池配对对照。

## 运行与隔离

入口只接收正常缓存及两份拟合解，不接收test路径、异常标签、mask或测试指标；不扫描测试数据。`torch.load(weights_only=True)`加载缓存。缓存和结果的正常路径/图像清单、patch抽样、fit/holdout groups、扰动RNG清单必须一致。

```bash
python -m scripts.run_normal_tail_pilot \
  --normal-cache results/<normal-cache>/<dataset>/normal-inputs.pt \
  --solution results/<batch>/results/seed42/<dataset>/uvarfs_main.json \
  --beta-zero-solution results/<batch>/results/seed42/<dataset>/uvarfs_asls_no_variability.json \
  --config configs/normal_tail_pilot.yaml \
  --out results/normal-tail-pilot-v1/seed42/<dataset> \
  --device cpu
```

需要CUDA时，先人工/调度核对占用，仅分配物理GPU0–3中的一张并设定`CUDA_VISIBLE_DEVICES`，再指定`--device cuda`。本工具不抢卡、不启动其他任务、不安装依赖。默认CPU；真实完整池成本应记录。

所有输入必须在当前项目，输出真实路径必须在当前项目`results/`内，拒绝symlink/junction逃出项目、覆盖历史输出或遗留`.partial`。先写`.partial`，来源前后SHA一致且全部输出完成后发布；记录逐文件SHA。结果、日志与分析不进Git，遵循第47.1节及EXPERIMENT_STORAGE.md。

输出独立`normal-tail-selection-pilot-v1`身份，不冒充v22 Main结果：

- `nuisance-selection.json`、`beta_zero-selection.json`：原/新winner、全部候选图像风险、one-SE门槛、支持来源及实际选择成本。
- `representation-specs.json`：原mean、tail候选在β>0/β=0下四个表示，以及两份tail同支持uniform对照。格式兼容现有`pipeline_eval.transform`；仅供独立评价适配，**当前run_all不加载它来替换Main**。
- `fixed-config.json`、`report.json`、`files-sha256.json`：配置、来源、完整性及明确的`anomaly_performance_assessed=false`。

β=0使用原β0拟合池，支持生成已经忽略特征阶段扰动；新selector不访问任何perturbation tensor。ASLS的nuisance不变，不能把β0称为整个系统不使用扰动。

## 预设评价设计

后续如决定进入检测验证，应先锁定评价适配与配置，独立目录、独立fit/memory seeds42/123/3407、BMAD全六；以dataset×fit为配对区组，所有方法共享正常抽样、memory20000、输入、backbone及detector。当前全部352旧方法控制继续完整保留，不根据pilot或test删除不利控制。

必须比较tail与同池mean、同输入Raw/PCA/五Random、β0及同支持uniform；五Random先在fit内平均，再汇总三个fit，不能当Main重复。图像全六等权，定位仅Brain/Liver/RESC，同时报告逐集方向、均值/样本SD及真实成本。主要问题是是否改善Raw及mean的检测/定位；更低正常CVaR不构成成功判据。不事后挑最有利分支、种子或数据集，也不据测试成绩调one-SE或CVaR比例。

本机没有完成该候选的全六三fit检测验证，新增normal-only pilot不计为完成的新BMAD检测实验。不能据输出解或数学检查声称性能已改善。

## 必须报告的局限

- 当前每图仅16个拟合/留出patch，Top-1%实际为采样patch最大值；它不是完整patch网格的真实Top-1%。工具明确保存这一限制。
- 原64图留出已经被ASLS使用，本selector也在其上选择；不能称作独立最终验证。患者ID不可用，不宣称患者隔离。
- 最终memory20000与fit reference密度仍不同，本试验不偷偷改变预算来掩盖问题。
- one-SE门槛来自原winner的小样本条件估计，原winner本身已经经过候选选择，不能作为已校准的泛化界或正式显著性检验。
- 当前支持池可能根本没有更好的候选。明确回退是合法结果；不能通过更强压缩、更大池或test搜参强行制造正向结果。

## 第19节变更检查

研究问题、默认Main、ASLS、原U-VaRFS数学目标、标签约束、detector、原始预算及旧控制不变；新增独立候选选择规则，不宣称与旧结果可混合。采用独立pilot身份，原EXPERIMENT_VERSION/default结果目录保持。原数据和现有结果只读，运行期间不pull服务器或改冻结配置。新Main/数学主目标的后续升级须有具体可审查提案，本次不自动激活。

## 使用的技能

采用Scientific Critical Thinking的“观察/假设/机制分开”和Experimental Design的配对区组、正确重复单位及完整控制原则。工具不使用药物试验随机化模板，也不把三个fit当独立患者群体。

技能来源：[Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065)，doi:10.48550/arXiv.2609.00065。2026-10-16核对arXiv当前条目的作者、标题及年份；该文是流程技能来源，不作为本算法性能或新颖性的证据。
