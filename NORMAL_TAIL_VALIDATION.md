# 正常图像尾部保真：服务器配对检测验证

用户于2026-10-17要求“按照上次的改进继续在服务器上进行实验”。本轮明确授权[已实现的尾部候选](NORMAL_TAIL_PILOT.md)进入真实检测/定位验证；注册Main仍是第48节v22，不将pilot静默改名为Main。沿用Experimental Design与Scientific Critical Thinking的配对区组、正确重复单位及代理/真实性能分离原则，技能来源记录见上述方案。

## 冻结的科学配置

- Frozen DINOv3、ASLS、L2主输入、原正常192/64图划分、96扰动图像、256维/floor/cap、memory20000及原抽样、cosine1NN/Top1%/pixel评价全部保持。
- 只在原fit-only池中重新选择，不生成支持、不重拟合权重。固定Top1%、CVaR25%、one-SE系数1、chunk256，允许明确回退。异常标签和mask只在选择结束后用于评价，不按test调参数/输入/分支。
- BMAD全六×独立来源fit/memory seeds42/123/3407，18个配对区组；物理GPU仅0–3，使用前检查占用和共同调度锁，不抢占其他进程。

## 配对延续与完整旧控制

来源为已完成、冻结且完整审计的v22三个fit。它们彼此独立，但本轮是**同三个来源fit的配对延续**，不是额外三次新拟合，也不是新的患者样本或未接触测试。

每个区组重新评价六个预先指定表示：nuisance_mean、nuisance_tail、nuisance_tail_uniform、beta_zero_mean、beta_zero_tail、beta_zero_tail_uniform。共108个新评价行；其中两个mean是重复的一致性控制，只有四个新增表示。新身份`normal-tail-evaluation-v1`，不混入v22原版结果。

全部352个旧方法/区组的指标、预测和正常来源单独复制到`controls/`，共6336行保持原版本、原成本与原指纹，不冒充本轮重拟合结果。其余完整拟合产物由项目内原始批次及完整归档保留，不删除不利控制。来源文件在使用前后逐SHA验证。

正常输入按原种子和抽样实现重新提取并完整保存，所有候选的原正常验证损失必须严格复现。允许复用已经核验的准备缓存，但不扩大正常图像数。选择器直接复用上次代码，不更改尾部规则。

每个区组重建相同正常memory；原mean与β0的**逐图分数必须完全相等**，图像/像素点估计绝对差≤1e-12，图像/标签/mask清单及memory抽样、容量、reservoir seed必须一致。任一失败则标记工程失败、停止分派后续任务，保留已运行任务和日志，不能放宽门槛来制造收益。未通过这两项重复控制，不允许用旧352控制作性能比较。

## 入口与隔离

```bash
python -m scripts.run_normal_tail_validation \
  --source-batch results/remote-experiments/v22-four-gpu-20261013T073952Z \
  --run-root results/remote-experiments/normal-tail-evaluation-<UTC> \
  --expected-head <已提交的评价源码SHA> \
  --control-dir /data/ftcl/guo/uvarfs-automation
```

协调者不训练旧求解器，不修改nohup.sh、原数据或历史结果，不安装依赖。源码、选择配置、worker/协调者/分析器SHA固定；改动即失败。stdout、exit、缓存、权重支持、来源、预测和报告全部实际位于项目内，Git忽略；仅SSH认证与必要GPU锁/调度控制外置。见[存储限制](EXPERIMENT_STORAGE.md)。

先完成本地完整回归与服务器CUDA/真实Liver正常准备验证，再四卡启动固定18区组；不能根据Liver异常成绩决定其他区组的配置。每个正式worker只写自身目录，complete标记和逐文件SHA齐全才算完成，不因旧PID缺失误报失败，不重复启动。

## 预设分析与边界

全部18严格完成后，`scripts.analyze_normal_tail_validation`独立复算图像点估计，报告tail−同池mean、同层Raw/PCA/五Random、tail β0、同支持uniform、v21/v20等完整对照。五Random先在fit内平均，再按三个来源fit计算均值/样本SD；图像全六等权，定位仅Brain/Liver/RESC等权，完整报告逐集方向和成本。图像成对分层bootstrap固定2000次，条件于单个fit，不是患者/跨fit区间，不虚构像素CI。历史test多次开发的限制保持。

正常16采样patch的Top1%仍是最大值；64图留出曾用于ASLS，one-SE仍是选择启发式，原池可能没有更好候选。本轮只是检验已经提出的具体候选，不能预告收益，压缩/正常CVaR下降不能补偿检测或定位下降。

完整回传包括新批次、缓存、控制副本、来源和日志，终态封闭、整包及每文件SHA后发布本机项目副本；分析/索引镜像服务器项目。完成本批后停止监控，长期研究目标按真实性能判断。

## 第19节检查

研究问题、ASLS、已批准数学目标、异常标签约束、detector和预算保持；上次独立选择候选的真实评价现已获用户明确授权。新增独立评价身份/目录和来源配对语义，默认Main/EXPERIMENT_VERSION不变；不把历史352方法标为本轮新训练，不将旧三fit算作额外重复。代码、配置说明和测试按授权提交同步，实验产物不进Git。
