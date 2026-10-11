# v21：正常匹配分数与完整邻居分布的稀疏保持

用户于2026-10-11批准新目标、代码同步、独立全六三种子验证，并允许物理GPU0–3并行。研究目标仍是开发改善异常检测和定位的稀疏特征选择方法。正常代理改善与实现检查不证明异常检测收益；真实结果和分析只保存于仓库之外。

## 固定框架与数据

BMAD全部六集、Frozen DINOv3 ViT-S+全部层候选、ASLS选层、U-VaRFS选潜在维数、normal memory cosine 1-NN、Top-1%图像评分与原定位协议保持。ASLS沿用[v20正常局部选层](NORMAL_LOCAL_SELECTION.md)，原2–6层全部组合；主输入仍预设逐层L2，raw完整配对消融。未使用测试成绩选择输入或参数。

原256张官方normal train图像、每图16patch，192拟合与64正常留出；96张拟合图像的原四种nuisance扰动，同位置patch缓存。全部方法共享图像、划分、memory图像/patch及20000行容量。异常标签、疾病标签和mask只用于评价。拟合排除同图全部reference；留出只查拟合集。不声称患者隔离。最终memory沿原官方normal train共享抽样，不将选择阶段holdout称为最终独立测试。

## 精确目标

在ASLS选中层的声明dense输入上，用所有允许的正常拟合reference构建teacher。令`t_i=max_r cosine(x_i,x_r)`；学生每次计算自己的最近reference，`c_i(p)=max_r cosine(x_i diag(sqrt(p)),x_r diag(sqrt(p)))`。不把teacher最近边固定为student最近边。

仅用拟合集确定`tau=max(.01, median_i(t_i-dense_rank8_cosine_i))`。正常reference少于8时取该query最后可用排序，至少需要两个跨图patch；真实默认参考预算不触发此边界。tau在整个有限池及留出保持固定，留出和测试不选温度。

令`T_i=softmax(dense_cosine_i/tau)`、`S_i(p)=softmax(student_cosine_i/tau)`，概率涵盖全部允许reference，不做top-k截断。同图reference概率为0。query权重`a_i=clip(1+(1-t_i)/max(mean(1-t),1e-4),1,4)`，再归一均值为1。

```text
L_score = mean_i a_i*(c_i(p)-t_i)^2
          / max(mean_i a_i*(1-t_i)^2,1e-4)
L_neighbors = mean_i a_i*KL(T_i || S_i(p))
              / max(mean_i a_i*KL(T_i || Uniform_allowed_i),1e-6)
L_var = mean_clean,perturb(1-cosine(weighted_clean,weighted_perturb))
        / max(mean_dense(1-cosine(clean,perturb)),1e-4)

min_support,p L_score + L_neighbors + .002 L_var
|support|=K=min(256,M), sum(p)=1
1e-4/K <= p_j <= min(1,4/K) on support; p_j=0 elsewhere
```

保留256预算，不扫Main lambda，不优先压缩。fit与holdout分别使用各自dense teacher的score/distribution归一尺度；nuisance仅属于fit。零dense行不能定义teacher，报错；零稀疏行支持无效并记录/拒绝，不以epsilon产生伪正常邻居。无有效候选时明确失败，不偷偷取消预算。

## 求解与选择

三起点为拟合局部坐标贡献、正常能量、稳定局部贡献。每个保留uniform与固定支持重拟合。float64解析梯度、query分块256、bounded-simplex projected BB与Armijo下降，固定支持默认最多150步；最多两轮支持交换，每轮四个全坐标梯度提案。donor采用当前质量`p_j*g_j`，质量转给新维度；每个提案保留uniform并重拟合，只有拟合目标严格下降才继续该路径。有限池不声明穷尽支持。

`generate_fit_only_pool`没有holdout参数。完整拟合池及继续搜索轨迹先生成并哈希，再创建正常留出validator；留出按`heldout L_score + heldout L_neighbors + .002 training L_var`选择有限池最小点。留出无效零行支持不参与选择。存储池、温度、尺度、单调拟合历史、交换接受、支持、权重、真实成本及残差，允许uniform胜出。

max在最近reference切换/并列处非光滑，只报告固定支持当前最近分支的投影残差；不借用convex/global Gram证书，不宣称全局最优、完全收敛或未知异常分离保证。KL邻居保持本身不作为独有新颖性或性能证明。

## 完整控制与验收

`main`为新目标。新目标在fixed4/all/random层、历史ASLS选择等输入上也评价。主方法同输入Raw、PCA256、五Random256、同支持uniform、β=0、无score项、无distribution项完整保留。β=0删除特征阶段所有nuisance访问，包括起点；ASLS相同，用于检验U-VaRFS variability贡献。

完整v20 local分支保留为`local_*`，raw下为`raw_input_local_*`，包括β=0/去rank、完整历史Main、quadratic支持解与cosine/cardinality/v17控制。`gram_*`和`raw_input_gram_*`继续完整原quadratic分支。原默认172个控制全部存在，新增90个结果，共262方法/数据集、1572行/全六/独立fit种子。`configs/v20_control.yaml`固定历史配置用于回归，不能作为新Main或混入新版本输出。

锁定新配置与代码后，独立fit/memory种子42、123、3407，全六统一评价。Random baseline五种子只报告其均值/样本SD，不能当Main重复。独立fit种子报告mean/样本SD、同输入Raw/PCA/Random/β0、v20与层控制，图像成对bootstrap与定位差异。部分收益、部分下降完整披露，不以低loss、稀疏或提速补偿下降；历史test已多轮用于开发，不声称未接触的最终测试。

## 并行与审计

版本`gpu-eval-v21-normal-matching-distribution`，新结果目录和指纹；不覆盖/混合v20。每个dataset worker限定一个数据集，仅写自己的目录；同fit seed使用完全一致配置及results_dir，不发布可能竞争的global summary。各seed六集全部严格完成后，由唯一协调者调用`python -m scripts.merge_dataset_workers --seed SEED`检查版本、配置、代码、完整方法/指标、六集文件，再发布汇总。

```bash
python -m scripts.run_all --seed 42 --dataset liver --dataset-worker
python -m scripts.merge_dataset_workers --seed 42
python -m scripts.analyze_local_results --seed-results SEED42_DIR SEED123_DIR SEED3407_DIR --out OUTSIDE_REPO
```

GPU调度、锁、连接、控制、stdout、exit与收回助手在仓库外；只允许物理0–3并先核对占用。保留用户nohup.sh、未提交文件和原始数据。单Liver工程验证后按数据集分配18个任务，完成的同版本Liver允许严格resume。估计以完整旧实验实测及新增成本为基准，Liver实测后更新；启动身份核验一次，随后到预计完成窗口检查，不频繁轮询。工程失败定位具体阶段，等价修复可以继续；研究目标变更需新的具体提案。

## 第19节变更检查

研究问题不变；ASLS定义不变；U-VaRFS目标和有限支持生成明确改变，用户已批准；无异常标签/mask进入拟合；detector不变；方法共享原数据/memory/维度预算；新结果不能当历史数值复现；版本、目录与指纹升级。实现与实际性能分别验证，性能尚待完整独立实验。
