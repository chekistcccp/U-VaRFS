> **Codex 接手请先读：[AGENTS.md](AGENTS.md)**。该文件锁定研究主线、实验协议、当前实现状态与禁止偏移项；后续修 bug、提速和补实验均应以其为最高优先级项目说明。  
> **当前实现版本：`gpu-eval-v7-geometry-audit`**。主 ASLS 保持已批准的 normal patch geometry 和原 gate-prefix 规则；增加正常几何组合搜索消融、U-VaRFS 预算后证书与固定支持集下界。原 loss、无异常标签约束、统一预算与所有原 baseline 保持。当前共 32 方法；步骤和验证边界见 [GEOMETRY_AUDIT.md](GEOMETRY_AUDIT.md)，patch 主输入批准记录见 [PATCH_ASLS_UPGRADE.md](PATCH_ASLS_UPGRADE.md)。新版真实性能仍需服务器验证。

已归档的历史回传结果见 [v4 六数据集分析](reports/2026-10-06-v4-analysis/analysis.md) 与 [v3 历史分析](reports/2026-10-06-v3-analysis/analysis.md)。按用户最新默认设置，每轮改进检查通过后只提交、推送代码、配置、测试和开发文档；实验结果、日志、图表及结果分析报告仅保留本地，具体约定见 `AGENTS.md` 第 28 节。
> **已兼容 BMAD 官方 6 个整理后的 AD 压缩包**：包括 `Liver_AD.zip` 的 `Liver/Train/hist_DIY` 特殊 img/label 目录，以及 Chest/OCT2017/RESC 的 `val` 命名。无需重新下载原始 BTCV/LiTS。  
> **数据入口已固定为 `data/archives/`**  
> 你只需把自行下载的 BMAD 原始压缩包全部放入该目录，不需要手工解压、改名或整理内部目录。  
> `bash run.sh` 会自动解压到 `data/_extracted/`、识别并预处理到 `data/processed/BMAD/`，然后运行 U-VaRFS 全部实验。详见 [data/README.md](data/README.md)。

# U-VaRFS：DINOv3 自适应稀疏层选择与无监督可变性正则特征选择用于医学图像异常检测

> **当前阶段：实验设计说明（v0.1）**  
> 目标：在 BMAD 全部 6 个医学异常检测数据集上，验证 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS + Normal Memory Matching** 的轻量、无标签异常检测框架。

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
Normal memory bank + cosine kNN
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
- test set 在所有方法和超参数固定后才进行最终测试。

因此，本项目中的“无监督/无标签”特指：

> 在给定 BMAD 官方 normal-reference training split 的前提下，特征选择和异常检测过程不使用异常类别标签或病灶 mask。

它不等价于“含未知污染样本的完全无监督混合数据异常检测”。

### 2.2 核心假设

H1. DINOv3 多层表征存在明显冗余，不同医学模态所需要的层深度不同。

H2. 对轻微成像扰动稳定、同时能够保持正常数据几何结构的 DINOv3 层更适合医学异常检测。

H3. 在选中的层内，仅有一小部分 latent dimensions 对稳定正常表征有贡献。

H4. 与 PCA 或固定层组合相比，**ASLS + U-VaRFS** 能以更少的层、特征维度和 memory-bank 开销达到相当或更高的异常检测性能。

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

### 6.1 目标

从全部 12 个 DINOv3 block 中自动选择少量层，而非人工指定。

希望同时满足：

1. 选中层能够保持完整 DINOv3 hierarchy 的正常表征结构；
2. 对无意义成像扰动具有较高稳定性；
3. 层数尽可能少。

### 6.2 每层的正常表征几何

对训练集正常图像/采样 patch，得到第 (l) 层特征 (F_l)。

归一化后构造：

[
K_l = F_lF_l^T
]

建立全层 consensus geometry：

[
K_C = \frac{1}{L}\sum_{l=1}^{L} K_l
]

当前主方法对采样 normal-reference patches 构造每层 cosine Gram，并使用全层平均 consensus。默认复用 256 张正常 fit 图像、每图 16 patches，即 4096 个对应 patch rows。

v6 按用户批准将 patch geometry 作为 Main 输入，保留 `asls_pooled_raw`/`asls_pooled_uvarfs` 消融。通过 layer-Gram 内积等价计算控制内存；原 sigmoid/Adam/loss/离散规则和数据预算保持。不以测试性能择优选择版本。

### 6.3 Layer variability

对于正常图像 (x) 和轻微扰动 (T_v(x))：

[
u_v^j = \frac{\mathbb{E}_i[(f_j(x_i)-f_j(T_v(x_i)))^2]}{\operatorname{Var}_i(f_j(x_i))+\epsilon},
\qquad V_l = \operatorname{Mean}_{v,j\in l}(u_v^j)
]

当前实现使用全体采样正常 patch 的总体方差与扰动平方差均值，包含 batch 之间的方差；不对 batch 内的比值取平均。

低 (V_l) 表示该层对 nuisance perturbation 更稳定。

### 6.4 稀疏层选择

每层设置 continuous gate probability：

[
p_l = \operatorname{sigmoid}(a_l)
]

当前代码使用 sigmoid continuous layer gates 与 Adam。它不是 Hard-Concrete 或严格的 L0 relaxation。最终按概率排序，再在预设层数预算内选取满足 geometry tolerance 的最小前缀集合，检测阶段仅使用选中的层。

目标函数：

[
\mathcal{L}_{layer} =
\frac{\|L^{-1}\sum_l p_l K_l-K_C\|_F}{\|K_C\|_F}
+\gamma\frac{\sum_l p_l\widetilde V_l}{\sum_l p_l}
+\rho L^{-1}\sum_l p_l
]

其中 \(\widetilde V_l\) 是 normal-only variability 的 min-max 归一化值：

- 第一项：保持 DINOv3 hierarchy 的几何结构；
- 第二项：抑制扰动敏感层；
- 第三项：continuous gate sparsity。

### 6.5 无标签选择策略

为了避免通过 validation anomaly labels 调节 layer sparsity，采用无标签停止标准：

> 选择能够达到预设 geometry-preservation 水平的最小层集合。

默认：

[
\frac{\|K_C-K_g\|_F}{\|K_C\|_F} \le 0.05
]

该条件表示相对 Frobenius 几何误差不超过 0.05，不等同于解释方差或“保留 95% 信息”。若层数预算内无可行前缀，当前实现返回预算上限前缀并明确记录 `feasible=false`。

默认允许 2–6 层，实际层数由正常数据决定。记录 gate 范围、梯度和选层几何误差，以诊断饱和与几何退化。

---

## 7. 模块二：U-VaRFS

### 7.1 原始 VaRFS 的修改原则

原 VaRFS 的核心思想为：

[
\text{Discriminability}
+
\text{Variability Regularization}
+
\text{Sparsity}
]

其中监督 discriminability 项依赖标签 (y)。

U-VaRFS 只替换这一监督项：

[
\boxed{
\text{Representation Preservation}
+
\text{Variability Regularization}
+
\text{Sparsity}
}
]

从而实现无异常标签特征选择。

### 7.2 输入

ASLS 选出 (L^*) 个层后，将对应 latent channels 拼接：

[
X \in \mathbb{R}^{N\times M}
]

例如选择 4 层时：

[
M = 4\times384=1536
]

### 7.3 表征几何保持

完整特征：

[
K=XX^T
]

为每个 feature/channel (j) 设置非负权重 (w_j)。

加权特征几何：

[
K_w =
X\operatorname{diag}(w)X^T
]

无监督表征保持项：

[
\mathcal{L}_{rep}
=
\frac12
\left\|
K -
X\operatorname{diag}(w)X^T
\right\|_F^2
]

含义：

> 用尽可能少的稳定 latent dimensions 保留完整 DINOv3 正常特征空间的几何结构。

### 7.4 Variability regularization

对每种扰动 (v)，第 (j) 个特征定义：

[
u_v^j =
\frac{
\mathbb{E}_i
\left[
(f_j(x_i)-f_j(T_v(x_i)))^2
\right]
}{
\operatorname{Var}_i(f_j(x_i))+\epsilon
}
]

分母用于防止选择“几乎恒定但没有信息”的 latent channel。

将多种 variability source 组成：

[
P=[u_1,u_2,\ldots,u_V]
]

并继承 VaRFS 的形式：

[
R=PP^T
]

[
\mathcal{L}_{var}=w^TRw
]

### 7.5 最终 U-VaRFS

[
\boxed{
\min_{w\ge0}
\frac12
\left\|
K-X\operatorname{diag}(w)X^T
\right\|_F^2
+
\beta w^TRw
+
\lambda\|w\|_1
}
]

可选约束：

[
0\le w_j\le1
]

用于防止某些 channel 被过度放大。

### 7.6 计算优化

不能直接对大量 patch 构造 (N\times N) Gram matrix。

令：

[
H=
(X^TX)\odot(X^TX)
]

则：

[
\left\|
XX^T-X\operatorname{diag}(w)X^T
\right\|_F^2
=
(\mathbf{1}-w)^T
H
(\mathbf{1}-w)
]

因此可以直接在 feature 维度上优化：

[
\boxed{
\min_w
\frac12(\mathbf{1}-w)^TH(\mathbf{1}-w)
+
\beta w^TRw
+
\lambda\|w\|_1
}
]

当 (M\approx384\sim2000) 时，该问题规模很小。

优化器：

- FISTA / Accelerated Proximal Gradient；
- proximal soft-threshold；
- 可选 ([0,1]) clipping。

### 7.7 无标签 sparsity 选择

主实验不使用 anomaly labels 调 (lambda)。

沿 (lambda) path 求解，并选择：

> 在几何保持误差不超过 5% 的前提下，维度最小的解。

额外报告固定预算：

- 64 dims；
- 128 dims；
- 256 dims。

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

本项目不要求所有数据集均刷新 SOTA；核心目标是证明：

### 目标 1：高稀疏率

理想结果：

[
12\rightarrow3\sim5\;layers
]

且：

[
4608\rightarrow64\sim256\;effective\ dimensions
]

### 目标 2：性能不下降或提升

期望：

[
AUROC_{sparse}
\ge
AUROC_{full}
]

至少达到 non-inferior，同时显著降低 memory / computation。

### 目标 3：自适应优于随机

[
ASLS+U\text{-}VaRFS
>
RandomLayer+RandomFeature
]

并显著高于其多随机种子的均值。

### 目标 4：跨模态规律

观察不同医学模态是否形成不同的 DINOv3 layer-selection pattern。

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
