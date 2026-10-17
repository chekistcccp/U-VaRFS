# 固定机制实验：支持选择、权重与正常审计隔离

用户2026-10-17明确批准依据研究审查修改代码、同步仓库并在服务器实验。本分支`mechanism-factorial-audit-v1`保持第46节正常1-NN/完整KL/.002 nuisance目标、Frozen DINOv3、L2、原始latent坐标、ASLS 2–6层、主支持K256、floor/cap、20000 memory及简单cosine detector。注册Main仍v22，不通过本轮test自动更名。此前所有原结果保留。

## 数据角色与已知限制

复用已逐SHA回传的全层正常backbone缓存，正常总预算仍256图×16patch，扰动仍96图。原192图中，前96个已有扰动图像全部进入拟合；其余96图用固定seed+27017随机分为48拟合、48选择。得到144拟合、48选择、64审计，三者按图像身份不交叉。64审计为原正常holdout，**过去已被历史研发使用**；本次从ASLS到特征选择完全重新拟合，不输入旧ASLS赢家/旧支持池/旧权重，当前管线不使用审计值决定表示。不能声称它是历史上未接触数据、患者隔离或新增患者验证。

ASLS只比较144拟合reference和48选择query，并仅用拟合nuisance；支持/权重/温度只用144拟合，有限支持选择只用48选择。表示先冻结哈希，之后才读取64审计特征；正常审计结果不反馈调参/选择。BMAD test及异常mask仍只用于最终固定评价，历史测试集继续标为研发验证。

## 支持与权重分离

每个β分支用原六起点/150步refit/两轮四交换建立新的fit-only池，不对每个旧支持做SLSQP。取训练目标最低的两个不同支持，沿原匹配梯度联合投影最多四轮、每轮四提案/150步原refit；只按真实训练下降继续。新池中的两个训练最优不同支持进入最后阶段。这里改变有限候选生成/选择流程，采用独立版本，不冒充原pursuit算法的等价复现。

对这些学习支持、五个预设Random256支持及全部M维支持，统一从uniform开始，用相同原bounded-simplex BB/Armijo权重refit，最多800步/tol1e-5，保留训练目标最佳点。均沿用原floor_mass/K与cap_factor/K；稠密baseline令K=M，仍只有主方法固定256维。相同算法、初始化规则与最大步数不保证相同收敛精度，逐项记录实际迭代/残差/时间，不以solver标志作为最优证书。

学习支持在两个最终weighted候选中，按48图matching+nuisance最小选择，再输出其同支持uniform。Random不在五种seed间择优，全部报告mean/SD；β0全部特征阶段不使用nuisance，ASLS稳定性项保持，故β0仍是特征阶段条件消融。

## 固定35表示和比较

学习支持β>0/β0各weighted与uniform（4）；五Random的uniform/β>0 weighted/β0 weighted（15）；同ASLS Raw、同目标dense weighted及β0、PCA256（4）；Fixed4/All/Last Raw（3）；五Random-K层Raw（5）；v22与pursuit各β>0/β0重复锚点（4）。总计35/区组，BMAD全六×来源seed42/123/3407共18区组630行。旧6336、尾部108、岭180及pursuit180按不可变项目引用保留、身份/成本单独审计。

主对象预设`mechanism_sparse`。优先比较新同输入Raw、五Random weighted/uniform、dense weighted、新β0/同支持uniform；层对照单列。新ASLS/144+48协议改变，所以历史同名Raw不是新主对象的同输入控制，不将跨协议差异解释为单一目标因果效应。

四个重复锚点逐图分数必须完全一致，全部适用指标差<=1e-12、memory抽样一致，才能引用历史评价。任何失败只定位工程，不能放宽容差/换方法。

## 正常审计与真实性能

第一层正常审计用64图缓存query对144拟合reference，报告固定表示的1-NN/邻居KL及采样图像分数误差，明确16点Top1%为max。

第二层正常审计用64张完整正常图像，另建共享20000行/1024图预算的audit memory，**排除全部64审计图像路径**，按真实全网格Top1%评分，报告相对同ASLS Raw的逐图分数/MSE与实际开销；该memory仅用于诊断。最终异常评价按原官方normal train共享memory协议另建原memory，四锚点验证原协议一致。两种memory不混合，审计不选参数。

预设成功方向是全六Image AUROC及三mask Pixel AUPRC相对新同输入Raw平均均为正，并完整披露逐集、Image AP/Pixel AUROC/AUPRO、来源fit样本SD和条件图像bootstrap；这只是固定研发判据，不构成未接触患者确认性检验。不得事后设置非劣界限、以稀疏/低正常loss替代性能，或按最好的解释控制切换主对象。若未达方向，不自动启动同目标新调参。正式确认需新患者/站点或预注册外层验证。

第19节：研究问题、数学损失、异常标签/mask限制、detector和主K256保持；正常数据角色、ASLS实际拟合数据与有限候选流程明确改变，已获本轮“按建议”授权，独立版本/新输出且保留全部历史身份。先CPU/CUDA协议与Liver normal工程检查，提交推送后冻结，物理GPU0–3占用检查/共同锁后执行；不装依赖，不抢占，不覆盖nohup/未提交文件。所有产物实际在两端项目results/logs/reports并主动完整SHA回传，见EXPERIMENT_STORAGE.md；完整审计/镜像后停止本批监控，不将长期目标标成功。

执行门控：CUDA/协议检查后，先在GPU0执行固定18组中的Liver seed42，包含normal fitting、两级正常审计、35表示评价和四锚点核验；它严格完成才在GPU0–3分派剩余17组，不重复Liver。不依据Liver测试指标修改设置/选择方法；失败仅定位工程。该门控是实际正式任务的工程验收，不冒充已经四卡同时运行。

审查程序参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 （当前记录v2，已在本轮前一审查核对）。该流程引用不作为方法有效性证据。
