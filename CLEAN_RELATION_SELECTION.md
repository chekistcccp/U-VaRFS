# Clean normal relation sparse selection：独立研究设计

用户于2026-10-19明确要求按研究复核建议修改代码、同步并在服务器再次实验。本设计实现已建议的“跨图可重复正常关系 + 实际nuisance”候选；是**新数学目标**，不是第46节目标的等价提速。注册Main仍第48节v22，唯一新研究主比较为`relation_sparse`，不按test选择分支或自动升级Main。

## 问题与约束

研究核心仍是开发改善异常检测/定位的原始坐标稀疏选择。冻结DINOv3 ViT-S+、原逐层L2、ASLS输入、K256、simplex/floor=1e-4/cap=4、normal-reference、memory20000、cosine1NN、Top1%、官方BMAD全六与评价指标不变。没有异常合成、疾病标签、mask、encoder训练、旋转投影、decoder或复杂detector。仅降低正常loss或更强压缩不构成成功。

为先隔离U-VaRFS的作用，复用已封闭mechanism批次的ASLS和144fit/48selection/64audit角色、96扰动图像×16patch及来源fit/memory seeds42/123/3407。支持与新目标权重重新拟合；旧ASLS/PCA/缓存及其成本保留。这是18个配对延续区组，不是新增三次独立正常抽样或新患者。完整固定4/全层/末层/Random-K的Raw控制继续评价；本轮不同时重设计ASLS。

## 固定clean图

只用fit clean特征：每个query从其他图像正常patch取全维cosine前8邻居。用`seed+61019`固定的3个50%坐标无放回子集重复搜索；原前8中的邻居至少在2/3子集中出现才为可靠正关系，权重为票数/3。禁止同图边。记录子集、有效query、边数和图权重SHA；没有可靠正关系的query不参与关系loss，全部无效则明确失败，不事后放宽阈值。坐标重采样只是图可靠性启发，不是患者bootstrap。

β>0/β0使用完全相同的clean图、温度和起点。图构造不读任何扰动。温度保持fit-only的原median近邻间隔规则，最小.01、rank8。48图在全部训练候选及权重冻结后，才构造对fit reference的同规则clean关系用于正常选择；64图不用于ASLS、支持、权重或参数选择。历史正常图和test曾参与研发，流程隔离不能恢复其未接触性。

## 新目标

`z_i(p)=normalize(x_i[S] * sqrt(p[S]))`。`A_i`为所有其他图像reference，`P_i⊆A_i`为固定可靠正邻居，`b_ij=a_ij`当j为正关系，否则`b_ij=1`。对有正关系的query等权平均：

```
L_relation = mean_i [log Σ_{j∈A_i} b_ij exp(cos(z_i,z_j)/tau)
                   −log Σ_{j∈P_i} a_ij exp(cos(z_i,z_j)/tau)]
min_{S,p} L_relation + .002 * L_actual_nuisance
|S|=256; sum(p)=1; 1e-4/K <= p_j <= min(1,4/K)
```

正关系在分子分母权重一致且非负，因此比值不超过1。真实nuisance项保留paired cosine误差及dense标定，仅β>0读取；β0删除此项，也不将增强正对藏在基础目标内。零dense/sparse行、无正关系、重复/越界支持或非有限值明确拒绝。分母其他正常关系不是疾病/异常负标签。

图冻结后该目标没有训练hardmax切换；最终推理仍是真实cosine1NN。float64 blocked解析梯度由独立autograd/有限差分复核，支持外坐标梯度用于联合投影。固定支持投影残差仅为此新光滑目标的一阶诊断，不是稀疏全局最优或检测证书。旧精度批次仍inconclusive，不能追认通过。

## 支持、统一拟合和选择

三个fit-only起点：经典Laplacian Score前K；clean split方差repeatability前K；固定rank relevance减0.5×已选坐标最大平方相关的互补支持。repeatability是两组正常图像的方差一致性代理，不使用扰动或疾病信号，不能称为病变SNR。

起点共同uniform初始化、同BB/Armijo最多800步。训练目标最低的两个不同支持进入最多4轮、每轮至多4个固定crossing-quantile投影提案、150步原refit；只有真实训练目标严格下降才继续。最终两个训练最优不同支持再从uniform共同800步，所有池冻结后以48图关系loss加训练nuisance选唯一赢家。拒绝提案、初始与最终拟合都计入成本；不无限扩预算或因Liver test改图/温度/目标。

五Random、Laplacian和repeatability支持及全维支持各自用同目标、同uniform初始化、同800步权重拟合；另保留uniform。两β主候选各自搜索，并增加固定β>0赢家支持、只用clean关系重新拟合β0的控制，区分支持变化与权重变化。dense cap使用4/M，与sparse可行域不嵌套，差异不可解释为单纯删除的唯一因果效应。

## 固定41表示与研究验收

每区组41表示：新目标主分支/β0及各uniform、同支持β0、dense两β、五Random weighted/uniform/β0、Laplacian和repeatability各uniform/weighted/β0；继承10个层/Raw/PCA表示及3个旧机制权重锚点。28项为新拟合或对应uniform，13项继承。18组共738评价行，含重复控制，不能全部称为独立新方法。8个锚点逐图预测精确、所有适用指标差<=1e-12、memory抽样一致后才能引用机制630行及其历史控制。

主要对照预定为同输入Raw和同目标Random weighted五seed均值；共同主要终点为全六Image AUROC、三mask Pixel AUPRC。预先采用两类对照的两终点均值均至少+0.5pp作为**探索性研发筛选**，不是临床阈值/功效保证/显著性标准。其余指标与逐集退化完整报告，不允许单模态收益掩盖其他数据集实质下降。三fit样本SD与五Random within-fit SD分开；2000次条件图像bootstrap只作未校正描述，不作多比较显著性/SOTA声明。无像素地图bootstrap，不声称患者CI。

外部完整方法竞争和新的独立患者/数据来源确认是后续科学验收阶段；本轮只判断候选选择准则能否先通过受控组件检验，不宣称全面超过现有方法。近期比较背景见ReConPatch、Laplacian Score、RealNet等已记录审查文献。新图仍可能编码设备、身份或背景，正常关系假设需由真实性能挑战；本轮不预先保证收益。

## 执行、封闭与来源

先正式Liver42完整工程门控（属于18组，不丢弃、不重复），其成功仅依据协议与工程，不依据test分数。随后物理GPU0–3执行剩余17组。固定入口/源码/config/source/cache/history状态SHA全部冻结；运行时不pull或安装依赖、不改原nohup.sh、不抢占其他任务。所有产物实际位于两端当前项目results/logs/reports且Git忽略。全18严格成功、exit0、所有原身份退出后，封闭归档、主动整包及逐文件SHA回传、独立审计、报告与索引双端镜像后暂停该批监控。

第19节检查：研究核心不变；ASLS定义不变但本轮输入冻结；U-VaRFS数学目标明确改变且已获本次用户授权；异常label/mask不进入拟合；detector不变；同输入控制数据预算一致、搜索成本完整记录；旧版本只能配对比较不可混合；独立evaluation_version升级，默认EXPERIMENT_VERSION/Main保留。

技能程序参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 。本次使用experimental-design v1.5和scientific-critical-thinking v1.5；程序指导不是方法有效性证据。
