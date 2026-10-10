# v20：以检测性能为目标的正常局部稀疏选择

用户于 2026-10-10 授权依据研究复核建议修改方法和研究主线。本轮研究问题是：**仅使用正常参考数据，能否学习一种改善异常检测与定位表现的稀疏特征选择方法？** 压缩、低正常训练损失或工程加速均不能替代检测收益。性能尚待真实 BMAD 全六数据集验证。

固定框架：BMAD 全六 → Frozen DINOv3 ViT-S+ 全层候选 → ASLS → U-VaRFS → normal memory → cosine 1-NN / Top-1% image score。异常标签、疾病标签和病灶 mask 只参与评价。正常邻居之间的排序约束是表征代理任务，不是异常类别监督。

## 1. 修正研究假设

正常全局 Gram 保持与未知异常分离之间没有充分保证。全局平均也可能掩盖局部邻居错误；原 ASLS 的最小可行层数优先与检测性能优先不一致。新方法将**局部匹配质量、正常近邻排序与实际扰动角度稳定性**作为可以检验的代理，不能将这些代理直接称为异常可分性。

主输入预先固定为逐层 L2。对于非零 patch，拼接后的余弦恰好等于所选层余弦的均值。拟合、扰动、memory 和 query 全部使用同一变换。这是本次明确授权的几何定义修正，不是按某个历史 test 分数择优。raw 输入保留完整配对消融；零行支持显式拒绝。

## 2. 数据与留出协议

仍只提取原 256 张正常训练图像、每图 16 个 patch；96 张图像进行原四种轻度 nuisance 扰动。采样图像先以 `seed+1907` 打乱，再保留末尾 25% 为正常留出集。默认 192 张用于拟合、64 张仅用于候选比较；96 张扰动图像完全属于拟合集。所有输入、目标分支和 PCA 使用同一拟合集。预算不足则报错，不静默取消留出。

patch 的 image group 随 manifest 保存。拟合近邻排除同图全部 patch；留出 query 只匹配拟合 reference。留出数据不进入权重梯度、P 统计或支持提案。没有可信患者 ID 时，只声称 image-disjoint，不能声称 patient-disjoint。该留出仅用于选择阶段；选择结束后，最终 memory 仍按原统一协议从官方正常 train 抽样，可能包含这些正常图像，所有方法一致。

四种扰动沿用 noise / blur / gamma / resolution；缓存既有 forward 的同位置 patch，不增加样本或 backbone forward，不制造病灶。扰动不会改变医学语义是设计意图，仍需医学/数据层面核验。

## 3. 正常局部目标

对声明的 dense 输入构建 teacher。每个正常 query 的跨图最近 reference 是正边，排序第 8 个邻居是较远边；参考不足时取可用的最远排序，至少需要两个跨图 patch。两者都正常，不是正常/异常正负样本。

设 teacher 两边余弦为 `t+`、`t-`，稀疏表示余弦为 `c+(p)`、`c-(p)`。每个 query 权重 `a=clip(1+(1-t+)/mean(1-t+),1,4)`，再归一至均值 1，使较难匹配的正常 patch 不被容易的正常 patch 完全淹没。定义 `s=max(mean((1-t)^2),1e-4)`，其中 t 包含两类边：

```text
L_local = mean_edges(a * (c(p)-t)^2) / s
L_rank  = mean_queries(a * relu((t+ - t-) - (c+(p)-c-(p)))^2) / s

L_var   = mean_clean/perturb(1-cosine(X_clean diag(sqrt(p)),
                                        X_perturb diag(sqrt(p))))
          / max(mean_dense(1-cosine(X_clean,X_perturb)), 1e-4)

min_support,p  L_local + alpha L_rank + beta L_var
|support|=K=min(256,M), sum(p)=1,
1e-4/K <= p_j <= min(1,4/K) on support; elsewhere p_j=0
```

固定 `alpha=1`、`beta=.002`、邻居排序 8，不使用 test 调整。原 P 及 quadratic/cosine 的定义保留在旧目标控制；新 Main 的 variability 明确改为实际 cosine nuisance，不再把旧二次 P 项称为相同损失。

支持从局部边坐标贡献、正常能量、稳定局部贡献三个启发式起点产生，各保留 uniform 和 capped projected-gradient refit。每起点至多两轮、每轮至多四个全梯度交换提案；提案只来自拟合集，按拟合目标接受继续搜索。候选池中，选择 `heldout L_local + alpha heldout L_rank + beta training L_var` 最小者。留出仅做有限池比较，不生成支持或梯度。未精修 uniform 候选可胜出，此时仍重新计算一阶残差。

只声称固定支持 capped simplex 一阶残差；不声称稀疏全局最优、未知异常分离保证或所有 fit 收敛。局部 RMSE 与全局 cosine Gram error 分开命名；旧 0.05 全局 tolerance 仅作用于旧目标控制，新 Main 不套用该可行性证书。全局 Gram / 1-NN agreement 继续作为诊断。

## 4. ASLS

全部 DINO 层仍为候选。在原 2–6 层范围内枚举全部组合，以正常留出 `L_local + L_rank` 加拟合 nuisance 平均 cosine distance（权重 .15）选最小分数组合。固定候选范围，而非先挑最小可行 K；不加入以更少层为优先的损失。层组合的实际拼接余弦从各层 edge 内积/行范数精确计算，raw 消融也用实际 raw 拼接几何。

当前 Main 是离散局部候选选择，不是旧 sigmoid/Adam gate，也不是 Hard-Concrete。旧 sigmoid/Adam + geometry-search / gate-prefix 与 pooled 消融完整保留，保存 `asls_previous.json`。新的 `previous_main` 使用旧 ASLS + 完整旧 fixed-budget cosine solver，在同轮同输入和同数据划分下运行；`raw_input_previous_main` 是原 raw 输入设计的配对控制，但因 v20 数据划分改变，不等于历史 v18/v19 数值复现。

## 5. 对照与判据

- Main 对相同层输入 Raw、PCA256、五个 Random256 seeds；保留同支持 uniform，区分支持与相对权重贡献。
- `asls_no_variability_uvarfs` 删除特征阶段所有 nuisance 使用（β=0，支持起点亦不使用扰动）；ASLS 相同，因此隔离 U-VaRFS variability。
- `asls_no_rank_uvarfs` 删除特征排序项，其他设置一致。以上两项仅在 local 分支定义，不伪造 Gram 分支的同名控制。
- 固定四层、全层、同 K 随机层的 raw/U-VaRFS，保留原 quadratic 全分支、旧 cosine/cardinality 和旧 fixed-budget 控制。不能只保留有利 baseline。
- `--seed` 为独立 Main/fit/memory 种子写入独立目录；随机 baseline 的五 seeds 不是 Main 独立重复。

结果需比较六数据集 Image AUROC/AUPRC，三 mask 数据集 Pixel AUROC/AUPRC/AUPRO，并报告成对差异、随机 mean±SD、独立 Main seeds 与实际开销。预设判据：Main 首先应改善同输入 Raw 的检测/定位并优于同预算随机/PCA；跨数据集下降和置信区间须如实报告。更低维度、更低正常 loss、更快程序均不能补偿性能下降。若代理改善而 AD 未改善，应判定该代理假设未获支持。

历史 test 已多轮用于开发复核，因此不能把再次测试称为完全未接触的无偏最终证据。可在正常数据上锁定此设计/配置，再报告预设全六结果；独立最终评估或额外未参与开发数据能加强结论，不能静默按 test 挑版本、层、β 或输入。

## 6. 第 19 节变更检查

1. 研究问题保持稀疏选择改善医学异常检测；删除压缩足以代表成功的叙事。
2. ASLS 定义改变：局部正常留出候选比较，已由本轮用户授权。
3. U-VaRFS 目标改变：局部/排序/实际角度 variability，已由本轮授权，明确方法升级。
4. 不引入异常标签或 mask 拟合。
5. 主 detector 不变。
6. 正常 fit/holdout 划分改变，原总提取预算不变且所有方法共享；memory 预算不变。
7. 新旧结果不可直接作为同协议重复；保留新轮配对旧方法。
8. bump 为 `gpu-eval-v20-normal-local-selection`，新目录/指纹，不覆盖混合旧结果。

## 7. 运行与证据边界

```bash
# 首先验证 Liver 的数据/数值/流程，不能据其 test 调整全六 Main
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
# 锁定配置后全六
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
# 独立重复；seed 参数会隔离目录
python -m scripts.run_all --seed 123
python -m scripts.run_all --seed 3407
python -m scripts.analyze_local_results --results results/gpu-eval-v20-normal-local-selection --out reports/v20
python -m scripts.analyze_local_results --seed-results results/gpu-eval-v20-normal-local-selection results/gpu-eval-v20-normal-local-selection/seed123 results/gpu-eval-v20-normal-local-selection/seed3407 --out reports/v20-seeds
```

本地数值与 mock 流程检查不构成医学检测实验。本机无真实 BMAD/权重/CUDA，真实收益尚未验证。代码、配置、测试、开发设计文档默认提交推送；回传结果、日志、图表和分析报告仅留本地。

本轮完整检查222项通过，10项CUDA检查跳过。覆盖独立解析梯度oracle、固定K/cap与候选比较、跨图近邻、留出隔离、β=0全部特征nuisance删除、172方法mock拟合/匹配/定位、断点与篡改拒绝、只读部分报告及独立Main种子汇总；Python/shell语法和diff检查通过。

## 8. 文献边界

[VaRFS 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12852897/)使用标签判别项，其监督表现不能转移为本项目无异常标签保证。[AnomalyDINO](https://arxiv.org/html/2405.14529v3)已有 frozen DINO + normal cosine matching；[DINOv3](https://arxiv.org/html/2508.10104v1)的 Gram anchoring 是与其他训练目标协同使用，不能证明 normal Gram 单独保证异常检测。[UniNet](https://openaccess.thecvf.com/content/CVPR2025/html/Wei_UniNet_A_Contrastive_Learning-guided_Unified_Framework_with_Feature_Selection_for_CVPR_2025_paper.html)已有异常检测特征选择；[HiMatchAD 预印本](https://arxiv.org/html/2606.22556v1)涉及 frozen DINOv3 医学层级匹配。因此创新应限定为可检验的正常局部匹配与 nuisance 正则联合稀疏选择，不宣称首次 DINO 特征选择/多层异常检测。
