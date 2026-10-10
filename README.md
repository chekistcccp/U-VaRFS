> **当前 v20：正常局部稀疏选择**。研究核心是开发改善检测与定位性能的稀疏特征选择方法。Main 预设逐层 L2，以正常局部匹配、近邻排序和实际扰动稳定性选层/选维，保留原256维上限；旧目标、raw、PCA与五seed随机控制完整保留。精确设计、变更授权与证据边界见 [NORMAL_LOCAL_SELECTION.md](NORMAL_LOCAL_SELECTION.md)。

> **Codex 接手先读 [AGENTS.md](AGENTS.md)**。固定 BMAD 全六、Frozen DINOv3 ViT-S+、ASLS + U-VaRFS、normal-only 和简单 cosine 1-NN。性能收益尚待真实六数据集验证，压缩、低正常 loss 与工程提速不能代替检测提升。代码/配置/测试/开发文档默认检查后提交推送；实验结果、日志、图表及分析报告仅留本地。

旧设计说明作为历史控制保留：[quadratic 与支持求解](OBJECTIVE_SPARSITY_UPGRADE.md)、[cosine/cardinality](COSINE_UVARFS_UPGRADE.md)、[fixed-budget cosine](PERFORMANCE_FIRST.md)、[v18支持交换](REFITTED_SUPPORT_EXCHANGE.md)、[v19工程缓存](EXACT_COSINE_CACHE.md)。当前主输入与目标明确变更，不按历史 test 自动挑选输入或超参。

> **已兼容 BMAD 官方 6 个整理后的 AD 压缩包**：包括 `Liver_AD.zip` 的 `Liver/Train/hist_DIY` 特殊 img/label 目录，以及 Chest/OCT2017/RESC 的 `val` 命名。无需重新下载原始 BTCV/LiTS。  
> **数据入口已固定为 `data/archives/`**  
> 你只需把自行下载的 BMAD 原始压缩包全部放入该目录，不需要手工解压、改名或整理内部目录。  
> `bash run.sh` 会自动解压到 `data/_extracted/`、识别并预处理到 `data/processed/BMAD/`，然后运行 U-VaRFS 全部实验。详见 [data/README.md](data/README.md)。

# U-VaRFS：DINOv3 自适应稀疏层选择与无监督可变性正则特征选择用于医学图像异常检测

> **当前阶段：方法实现与 BMAD 验证**
> 目标：在 BMAD 全部 6 个医学异常检测数据集上，开发并检验 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS + Normal Memory Matching** 的性能导向、无异常标签稀疏特征选择方法。

---

## 1. 研究目标

本项目研究一个简单但具有明确方法学问题的医学图像异常检测框架：

> **DINOv3 的所有 Transformer 层和所有潜在特征维度是否都对医学异常检测同样有用？能否仅依靠训练集本身的表征结构与扰动稳定性，自适应地选择少量层和少量潜在特征，在不使用异常标签的情况下达到或超过完整 DINOv3 表征？**

最终方法暂定名：

**AdaVaR-DINO / U-VaRFS**

核心框架：

```text
Medical image
    ↓
Frozen DINOv3 ViT-S+/16
    ↓
All 12 Transformer blocks
    ↓
Adaptive Sparse Layer Selection (ASLS)
    ↓
Selected multi-layer patch features
    ↓
Unsupervised Variability-Regularized Feature Selection (U-VaRFS)
    ↓
Sparse stable patch representation
    ↓
Normal memory bank + cosine 1-NN
    ↓
Image-level anomaly score + pixel-level anomaly map
```

本项目不微调 DINOv3，不训练大型分类器或分割网络。主要可学习/优化对象仅为：

1. 12 个 DINOv3 层的稀疏选择变量；
2. U-VaRFS 的潜在特征权重。

---

## 2. 研究定位

### 2.1 任务定义

采用 BMAD 的标准 one-class / unsupervised medical anomaly detection protocol：

- 训练阶段使用 BMAD 官方训练集作为正常参考库；
- **ASLS 与 U-VaRFS 不使用 good/Ungood、疾病类别或分割标签作为优化目标**；
- 异常标签与像素级 mask 仅用于最终评价；
- 新设计与超参固定后统一测试；历史test多轮用于开发，须披露自适应复用。

因此，本项目中的“无监督/无标签”特指：

> 在给定 BMAD 官方 normal-reference training split 的前提下，特征选择和异常检测过程不使用异常类别标签或病灶 mask。

它不等价于“含未知污染样本的完全无监督混合数据异常检测”。

### 2.2 核心假设

H1. DINOv3 多层表征可能存在可利用冗余；这是需要验证的前提。

H2. 保持正常局部匹配和近邻排序可能更适合cosine 1-NN检测，但不保证未知异常分离。

H3. 正常nuisance稳定性与稀疏支持联合优化可能改善AD；必须由β=0、PCA和Random对照验证。

H4. 固定原预算的**ASLS + U-VaRFS**能否改善六数据集检测/定位，是当前核心问题；压缩和速度不是收益替代品。

---

## 3. 数据集：BMAD 全部 6 个 benchmark

BMAD（Benchmarks for Medical Anomaly Detection）包含来自 5 个医学影像域的 6 个标准化异常检测数据集。

| Benchmark | 原始来源 | 模态 | Total | Train | Test | Val | 原始尺寸 | 像素级 mask |
|---|---|---|---:|---:|---:|---:|---|---|
| Brain MRI | BraTS2021 | MRI | 11,298 | 7,500 | 3,715 | 83 | 240×240 | ✓ |
| Liver CT | BTCV + LiTS | CT | 3,201 | 1,542 | 1,493 | 166 | 512×512 | ✓ |
| RESC | RESC | Retinal OCT | 6,217 | 4,297 | 1,805 | 115 | 512×1024 | ✓ |
| OCT2017 | OCT2017 | Retinal OCT | 27,315 | 26,315 | 968 | 32 | 512×496 | × |
| Chest X-ray | RSNA | X-ray | 26,684 | 8,000 | 17,194 | 1,490 | 1024×1024 | × |
| Camelyon16 | Camelyon16 | Histopathology | 7,321 | 5,088 | 1,997 | 236 | 256×256 | × |

主评价：

- 所有 6 个数据集：**Image-level AUROC**
- Brain MRI / Liver CT / RESC：**Pixel-level AUROC + PRO/AUPRO**

其中 **Liver CT** 作为 CT 主分析数据集；全部 BMAD 用于验证跨器官、跨模态泛化。

---

## 4. 数据预处理

### 4.1 原则

尽可能遵循 BMAD 已整理好的图像，不重新构建原始数据集，不额外依赖人工 ROI。

统一流程：

1. 读取 BMAD reorganized image；
2. 保持原始纵横比；
3. resize + padding 到方形；
4. 灰度图复制为 3 通道；
5. 按 DINOv3 官方 normalization 处理；
6. 默认输入分辨率 **448×448**；
7. 使用 224×224 作为效率/分辨率消融。

DINOv3 patch size 为 16，因此：

- 224×224 → 14×14 patch grid；
- 448×448 → 28×28 patch grid。

主实验优先使用 448，以改善像素级异常定位分辨率。

### 4.2 数据增强/扰动

扰动仅用于估计 feature variability，不用于生成异常：

- mild Gaussian noise；
- mild Gaussian blur；
- gamma / contrast perturbation；
- resolution degradation 后恢复原尺寸。

所有扰动必须满足：**不改变医学语义、不人为制造明显病灶**。

第一版不使用大角度旋转、强 crop 或强形变，以避免 patch correspondence 被破坏。

---

## 5. Frozen DINOv3 Backbone

主干：

**DINOv3 ViT-S+/16**

主要属性：

- 约 29M 参数；
- patch size = 16；
- embedding dimension = 384；
- 12 Transformer blocks；
- backbone 全程冻结。

候选层：

[
mathcal{L} = \{1,2,ldots,12\}
]

对每张图像提取：

[
F^{(l)} \in \mathbb{R}^{P\times384}
]

其中 (P) 是 patch token 数。

不使用人工固定的 ({3,6,9,12}) 作为主方法，仅作为 baseline。

---

## 6. 模块一：Adaptive Sparse Layer Selection（ASLS）

所有 DINO 层作为候选，在原2–6层预算内比较全部组合。目标为正常留出局部余弦误差 + 正常近邻排序误差 + 拟合图像 nuisance cosine distance（权重 .15）；选择分数最小组合，不优先减少层数。

每层 patch 先 L2，再拼接。此时实际拼接余弦等于所选层余弦均值，ASLS、U-VaRFS、memory 和 query 几何一致。raw 完整配对消融使用实际 raw 拼接余弦，不混用参考定义。

原256正常图像随机排列，192用于拟合、64用于候选比较；96扰动图像属于拟合，所有方法共享。拟合近邻排除同图全部patch，留出只匹配拟合reference，不参与权重梯度、支持提案或P统计。只声称图像隔离，不声称患者隔离。最终memory在选择结束后仍按共享官方normal train抽样。

当前 Main 是离散局部候选选择，不是 sigmoid/Adam 或 Hard-Concrete。旧 sigmoid/Adam、geometry-search、gate-prefix 和 pooled 控制保留；`asls_previous.json`记录旧ASLS。完整定义见 [v20设计](NORMAL_LOCAL_SELECTION.md)。

## 7. 模块二：U-VaRFS

正常全局 Gram 保持不能保证未知异常分离。新方法尝试让稀疏特征保持检测器依赖的局部近邻关系，并抑制实际 nuisance 敏感性；这是待检验的代理假设，不称为异常标签判别学习。

用全部候选输入建立 dense teacher，每个query的跨图最近正常reference和第8个正常邻居构成两类局部边。难匹配正常query权重有界至4，避免全局平均过度偏重容易正常patch。

```text
min_support,p L_local_cosine + L_normal_neighbor_rank + beta L_actual_nuisance_cosine
K = min(256, candidate dimension), sum(p)=1
1e-4/K <= p_j <= min(1,4/K) on support; others zero
beta=.002, rank weight=1
Y = X diag(sqrt(p))
```

三个正常数据支持起点保留 uniform 和 projected-gradient refit；有限交换只用拟合集。候选由正常留出 local/rank 加拟合 variability 分数选出，不扫 Main lambda，不挑更小维数。解析梯度只计算局部边；仅报告固定支持 capped simplex 一阶残差，不声称全局最优或异常性能保证。

新局部误差与全局 Gram 分开记录，旧0.05 global tolerance只作用于旧控制。原 quadratic/P、旧cosine/cardinality、完整旧fixed-budget目标都保留；它们与新目标不属于等价数值优化。`previous_main`保留旧ASLS+旧fixed-budget solver，但同轮共享v20数据划分，不能当历史版本数值复现。

`asls_no_variability_uvarfs`去除特征阶段所有nuisance使用，`asls_no_rank_uvarfs`去掉排序项；同支持uniform、Raw、同维PCA、五seedRandom保留。通过这些AD对照判断支持、权重与variability是否有效，不能只比较正常训练目标。公式、选择尺度、零行规则、预算与变更检查见 [NORMAL_LOCAL_SELECTION.md](NORMAL_LOCAL_SELECTION.md)。

---

## 8. 正常 Memory Bank 与异常评分

### 8.1 Patch memory

训练集正常图像经过：

[
\text{DINOv3}
\rightarrow
\text{ASLS}
\rightarrow
\text{U-VaRFS}
]

得到 patch representation：

[
z_p\in\mathbb{R}^{D'}
]

构建 normal memory：

[
\mathcal{M}=\{m_1,\ldots,m_K\}
]

为控制内存，默认：

- 每数据集采样最多 1024 张正常 train 图像、每图 32 个 patches；
- 相同 reservoir seed 将正常 memory bank 限制到 20000 rows；
- 检索使用 Torch CUDA FP16 exact cosine 1-NN，FAISS 仅为 fallback。

所有 feature-selection 方法使用相同 memory protocol，保证公平。

### 8.2 Pixel anomaly score

测试 patch (q)：

[
A(q)=
\min_{m\in\mathcal{M}}
\left[1-\cos(q,m)\right]
]

恢复 patch grid 后双线性插值得到：

[
A_{map}\in\mathbb{R}^{H\times W}
]

### 8.3 Image anomaly score

避免单个噪声 patch 主导结果，使用：

[
A_{img}=
\operatorname{Mean}
\left(
\operatorname{TopK}(A_{patch})
\right)
]

主设置：Top 1% patch；Top 5 patches / max score 作为消融。

---

## 9. 核心 Baselines

### 9.1 DINOv3 层选择对照

| 方法 | 层策略 | U-VaRFS |
|---|---|---|
| Last Layer | L12 | × |
| Fixed-4 | L3/L6/L9/L12 | × |
| All Layers | L1–L12 | × |
| Random-K | 随机 K 层 | × |
| ASLS | 自适应稀疏层 | × |
| Fixed-4 + U-VaRFS | 固定 4 层 | ✓ |
| All + U-VaRFS | 全 12 层 | ✓ |
| Random-K + U-VaRFS | 随机 K 层 | ✓ |
| **ASLS + U-VaRFS** | **自适应稀疏层** | **✓** |

Random-K：

- K = 3 / 4 / 5；
- 每个设置至少 5 个随机种子；
- 报告 mean ± SD。

### 9.2 特征选择对照

在完全相同 DINOv3 layers 和 anomaly detector 下比较：

- Raw feature；
- Random feature subset；
- Variance filtering；
- PCA；
- Sparse PCA；
- U-VaRFS。

### 9.3 BMAD 标准异常检测方法

至少比较 BMAD 已支持/已报告的典型方法：

- PaDiM；
- PatchCore；
- RD4AD；
- CFLOW；
- STFPM。

如复现成本允许，再加入近期 DINO-based medical anomaly detection 方法。

---

## 10. Ablation Study

### A. ASLS 模块

1. 固定最后层；
2. 固定 4 层；
3. 全 12 层；
4. Random-K；
5. 仅 geometry preservation；
6. geometry + variability；
7. **geometry + variability + continuous gate sparsity**。

### B. U-VaRFS

1. Representation preservation only；
2. + variability；
3. + (L_1) sparsity；
4. **完整 U-VaRFS**。

### C. 特征预算

[
D' \in \{32,64,128,256\}
]

以及 label-free automatic sparsity。

### D. 输入分辨率

- 224×224；
- 448×448。

### E. Memory size

- 5k；
- 10k；
- 50k；
- 100k patches。

### F. Dataset-specific vs Universal Layer Selector

**Dataset-specific ASLS：**

[
g^{Brain}\neq g^{Liver}\neq g^{OCT}\ldots
]

**Universal ASLS：**

在全部 BMAD normal training representation 上学习同一组：

[
g^{global}
]

比较两者可回答：

> 不同医学模态是否依赖不同 DINOv3 hierarchy？

---

## 11. 评价指标

### 11.1 检测性能

全部 6 个 BMAD：

- Image AUROC；
- Image AUPRC（补充，尤其用于不平衡数据）。

### 11.2 定位性能

Brain MRI / Liver CT / RESC：

- Pixel AUROC；
- PRO / AUPRO。

### 11.3 轻量化指标

必须同时报告：

- selected layers / 12；
- retained dimensions / candidate dimensions；
- feature-retention ratio；
- memory-bank size；
- feature storage MB；
- peak GPU VRAM；
- feature extraction time；
- image inference latency；
- ASLS optimization time；
- U-VaRFS optimization time。

推荐核心压缩指标：

[
Compression =
1-
\frac{D'\times |L^*|}
{384\times12}
]

注意：若 (D') 已定义为跨选中层拼接后的最终维度，应直接使用：

[
Compression =
1-
\frac{D'}
{384\times12}
]

---

## 12. 统计与可重复性

- 所有 random feature / random layer baseline：至少 5 seeds；
- 主要结果报告 mean ± SD；
- Image AUROC 建议使用 patient/image-level bootstrap 95% CI；
- 对支持 patient ID 的数据必须避免同一患者跨 split；
- test set 仅运行最终锁定配置；
- 保存：
  - selected layer gates；
  - selected feature indices；
  - U-VaRFS weights；
  - variability vectors；
  - memory-bank seed / indices；
  - 完整 config；
  - 软件版本与随机种子。

默认随机种子：

```text
42, 123, 3407, 2026, 2027
```

---

## 13. 主要论文结果图表

### Figure 1
完整方法框架图：

DINOv3 → ASLS → U-VaRFS → Memory Matching → Detection/Localization。

### Figure 2
**BMAD × DINOv3 Layer Selection Heatmap**

纵轴：

- Brain MRI
- Liver CT
- RESC
- OCT2017
- Chest X-ray
- Camelyon16

横轴：Layer 1–12。

显示每层 gate probability / selected state。

### Figure 3
Feature sparsity vs AUROC：

[
32\rightarrow64\rightarrow128\rightarrow256\rightarrow Full
]

### Figure 4
异常热图：

- Brain MRI；
- Liver CT；
- RESC。

显示 Input / GT / Full-DINO anomaly map / ASLS+U-VaRFS map。

### Table 1
BMAD 六数据集主结果。

### Table 2
层选择与 U-VaRFS ablation。

### Table 3
PCA / SparsePCA / Random / Variance / U-VaRFS 对比。

### Table 4
性能—资源开销对比。

---

## 14. 成功判据

首要目标是稀疏选择改善检测和定位表现。在相同层输入和统一memory预算下，比较Main与Raw、PCA、Random；以β=0和去rank对照检验代理模块贡献，以相同支持uniform区分选维与加权贡献。

全部六数据集Image AUROC/AUPRC与三个mask数据集Pixel AUROC/AUPRC/AUPRO都必须报告，包含下降项、成对差异及不确定性。随机基线报告全部五seed mean±SD；Main需独立fit/memory seeds，不能将随机baseline seeds算成Main重复。

更强压缩、较低正常loss或工程加速不能补偿检测下降。若局部代理改善而AD未改善，当前代理假设仍未成立。历史test已多轮用于开发，不能把再次评价称为完全未接触的最终测试；锁定设计后统一全六评价，并争取独立验证。

---

## 15. 两周实验计划

### Day 1–2：数据与 DINOv3
- 下载并检查 BMAD 全部数据；
- 建立统一 dataloader；
- 接入 DINOv3 ViT-S+/16；
- 完成 12 层 patch feature cache。

### Day 3–4：基础异常检测
- normal memory；
- exact cosine 1-NN（CUDA Torch；FAISS fallback）；
- image-level AUROC；
- pixel anomaly map；
- Last / Fixed-4 / All-layer baselines。

### Day 5–6：ASLS
- layer geometry；
- perturbation variability；
- sigmoid continuous layer gates + geometry-constrained discrete selection；
- Random-K 对照。

### Day 7–8：U-VaRFS
- variability matrix；
- efficient (H=(X^TX)\odot(X^TX))；
- FISTA；
- automatic geometry-retention sparsity selection。

### Day 9–10：完整 BMAD
- 六数据集批量实验；
- PCA / SparsePCA / variance / random feature baselines。

### Day 11–12：消融与效率
- 224 vs 448；
- feature budgets；
- memory budgets；
- universal vs dataset-specific selector；
- latency / VRAM / memory。

### Day 13–14：结果整理
- heatmap；
- anomaly visualization；
- bootstrap CI；
- 主表和消融表；
- 论文方法与实验初稿。

---

## 16. 推荐项目结构

后续实现建议：

```text
U-VaRFS/
├── README.md
├── configs/
│   ├── bmad_brain.yaml
│   ├── bmad_liver.yaml
│   ├── bmad_resc.yaml
│   ├── bmad_oct2017.yaml
│   ├── bmad_xray.yaml
│   └── bmad_camelyon.yaml
├── data/
│   └── README.md
├── models/
│   ├── dinov3_backbone.py
│   ├── adaptive_layer_selector.py
│   └── u_varfs.py
├── datasets/
│   └── bmad.py
├── anomaly/
│   ├── memory_bank.py
│   └── scoring.py
├── utils/
│   ├── perturbations.py
│   ├── metrics.py
│   └── visualization.py
├── scripts/
│   ├── extract_features.py
│   ├── fit_asls.py
│   ├── fit_u_varfs.py
│   ├── build_memory.py
│   └── evaluate.py
├── results/
└── run.sh
```

最终目标是：

```bash
bash run.sh
```

可以完成主要实验。

---

## 17. 参考资料

1. **BMAD: Benchmarks for Medical Anomaly Detection**  
   GitHub: https://github.com/DorisBao/BMAD  
   Paper: https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.html

2. **DINOv3**  
   GitHub: https://github.com/facebookresearch/dinov3

3. **Variability Regularized Feature Selection (VaRFS)**  
   Sadri AR et al. *npj Imaging*. 2026;4:5.  
   DOI: https://doi.org/10.1038/s44303-025-00136-5

---

## 18. 当前锁定的主方案

[
\boxed{
\text{BMAD all six datasets}
+
\text{Frozen DINOv3 ViT-S+/16}
+
\text{Adaptive Sparse Layer Selection}
+
\text{U-VaRFS}
+
\text{Cosine Normal Memory}
}
]

原则：

- 不训练 DINOv3；
- 不使用异常标签或 lesion mask 做特征选择；
- 不人工固定 DINO 层；
- layer sparsity 自适应学习；
- feature sparsity 由 U-VaRFS 学习；
- detector 保持简单，避免性能提升来自复杂下游网络；
- 所有主要创新通过统一 BMAD protocol 和严格 ablation 验证。

当前版本为 `gpu-eval-v20-normal-local-selection`，默认172方法、全六1032行。先固定设计验证Liver运行，再统一全六；不要据Liver test更改其余数据集Main。独立重复和只读审计：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
python -m scripts.run_all --seed 123
python -m scripts.run_all --seed 3407
python -m scripts.analyze_local_results --results results/gpu-eval-v20-normal-local-selection --out reports/v20
```

独立seed汇总使用分析器`--seed-results`，要求每个seed全六完整且设计/代码一致；部分回传使用`--completed-only`，不称为全六宏平均。本地实现检查通过不能代替真实检测实验。
