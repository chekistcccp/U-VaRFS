# 纯正常停滞端点诊断

用户2026-10-18批准按上一精度研究建议修改代码并在服务器继续研究。本轮是固定诊断实验 `normal-endpoint-diagnostics-v1`，不是再次求解、重新选择支持或启动条件检测。注册Main仍第48节v22，第46节真实normal1NN/完整邻居KL/.002 nuisance目标及ASLS/K256/floor/cap/memory保持。

## 问题与固定对象

上一固定精度批次18组252项正常拟合中22项过1e-5，主稀疏18/18线搜索停滞。当前分支投影残差不能证明非光滑驻点；不能把停滞全归于max切换、错误梯度、迭代不足或代理目标无效。本诊断区分这些可能机制，未事先认定答案。

小规模固定对象：全部六集、来源seed42，每集五个已冻结精度端点：主稀疏β>0、主稀疏β0、Random支持seed42的β>0、全维β>0、全维β0，共30个。全部从`precision-validation-20261018T020339Z`原端点只读加载，不挑选最难/最好test集，不将Random子集seed当来源fit重复。仅seed42，不是全六三fit或新患者确认实验。

主稀疏β>0与β0沿用各自父支持，可能不同；不能把两者差异解释为固定支持下nuisance的纯因果效应。

恢复上一正常144fit、96 nuisance/16patch、正常温度及ASLS层；48selection和64audit特征值不索引。β0不读取扰动值。原始normal缓存/父支持/权重/精度终点/config均先SHA并在结束后复核；原端点、支持和权重逐项保持。没有异常label/mask、test表、detector或新的正常审计。

## 方向、步长与记录

每端点预设五个可行线段方向：`Proj(p-gradient_p)-p`、当前分支梯度最陡可行单坐标对质量转移、回到父800步端点、回到同支持uniform、按原坐标顺序固定的可行坐标对。坐标对转移量取原floor/cap内最大可行量。后四者是解释控制，不参与权重选择；零方向保留记录但不重复算常量点。

沿每段固定h=`[1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12]`，实际点`p+h*d`；不做新投影/优化/线搜索接受。每点保留原实际目标、相对端点损失差、单侧差商、解析分支斜率、实际p/mass位移、可行性、原Armijo条件是否满足、浮点舍入分辨标记以及逐query最近邻ID和第一/第二间隔。即使某探针降低目标，也不采用该权重、不选最好的方向/步长。

独立按cosine的方向导数公式复核`gradient_p dot d`。对浮点计算中观测到的**精确**并列近邻，另外计算所有精确并列分支cosine方向导数的max，再经原平方分数误差求方向导数。`64*eps*(1+|cos|)`仅标记舍入尺度近似并列，不把其加入精确并列导数。近邻线性化首次交叉步长只是局部诊断，不保证真实交叉点。原normal nuisance导数独立按实际配对cosine复算。

舍入标记`|Δloss|<=64*eps*(1+|loss(p)|+|loss(probe)|)`为启发式数量级，不是误差界或停止标准。大步有限差商还含曲率，改变近邻ID不直接证明其导致停滞；单个方向通过也不证明Clarke/全局最优。解析梯度与差商不一致须同时看步长、分支变化和舍入，不依据一个数自动修solver。

每个非零方向7个探针；每端点原实际objective调用为`2+7*非零方向数`（首末重复复核），另计`1+探针数`完整近邻几何计算及每方向独立解析导数。最多1110次实际objective调用、1080次几何计算和150次方向导数，不新增DINO forward。不会用较低loss、近邻间隔或工程通过宣布检测改善。

## 执行、审计与结束

先真实CUDA/协议检查，正式Liver seed42诊断作为六组中的工程门控，成功后物理GPU0–3执行其余五组；GPU占用/全局锁检查，不抢占、不安装依赖。冻结新提交、代码与worker/scheduler/analyzer/config/SHA，原nohup.sh、未提交文件和历史结果保持。`scripts.run_normal_endpoint_diagnostics`为独立显式入口，不改默认Main入口和EXPERIMENT_VERSION。

结果保存在两端项目内独立`results/remote-experiments/endpoint-diagnostics-*`，日志在`logs/remote-experiments`、报告在`reports/`，Git忽略；仅认证与必要调度控制外置。六组严格完成、退出且coordinator exit0后，完整归档包含正常探针/权重来源引用、逐query ID/间隔数组、源码/config、stdout/exit、工程/CUDA夹具；主动完整回传、整包及逐文件SHA、安全解包、独立审计、双端报告/索引镜像后暂停本批。复用正常缓存两端已完整SHA保存，不重复复制，但项目真实路径/大小/SHA须一致。

独立审计复算残差、方向可行性、g·d、实际差商、query ID变化数量、精确/近似并列计数与固定步长覆盖；原生表全量对照。不把差商探针当独立样本，不做患者CI/显著性检验。正常/历史test研发暴露均保留，不能从本轮得到检测/定位结论。结束后提出具体求解器建议，不自动开启改目标、放宽上一门槛、再次支持搜索或检测批次。

第19节检查：研究问题/ASLS/U-VaRFS损失/标签约束/detector/预算不变；新增诊断不与旧检测结果混为同一评价，使用独立身份，不更名Main。用户“按建议”已授权本诊断代码、GitHub同步和服务器执行。

设计审查使用experimental-design v1.5与scientific-critical-thinking v1.5。程序参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 （2026-10-18已核对当前arXiv v2；不作方法有效性证据）。
