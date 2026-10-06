# U-VaRFS Project Agent Guide

> **用途：Codex 项目迁移与长期开发约束。**
>
> 本文件是本仓库的**最高优先级研究主线说明与工程交接文件**。后续 Codex 在修复错误、优化性能、增加实验、重构代码或解释结果之前，应先读取本文件。
>
> **核心原则：不要因为局部实现问题、性能问题、某个数据集报错或某次问答而改变研究问题。**
> 工程实现可以调整，baseline 可以补充，求解器可以加速，但论文主线、数据协议和无标签约束不得在没有用户明确指令的情况下漂移。

---

# 0. 一句话锁定研究主线

本项目研究：

> **在 BMAD 全部 6 个医学异常检测 benchmark 上，使用 Frozen DINOv3 的全部层作为候选，通过 Adaptive Sparse Layer Selection（ASLS）自动选择少量 DINOv3 层，再使用无需异常标签的 U-VaRFS 对选中层的 latent dimensions 做可变性正则稀疏选择，最后使用简单的 normal memory cosine matching 完成 image-level anomaly detection 与 pixel-level anomaly localization。**

最终主方法固定为：

```text
BMAD all 6 datasets
        ↓
Frozen DINOv3 ViT-S+
        ↓
All DINOv3 layers as candidates
        ↓
ASLS: adaptive sparse layer selection
        ↓
Selected multi-layer patch features
        ↓
U-VaRFS: unsupervised variability-regularized feature selection
        ↓
Sparse stable latent representation
        ↓
Normal memory bank
        ↓
Cosine 1-NN anomaly matching
        ↓
Image AUROC/AUPRC + Pixel AUROC/AUPRC/AUPRO
```

**论文核心不是“更复杂的异常检测器”，而是验证：DINOv3 的层和 latent dimensions 存在可利用的冗余，可仅利用 normal-reference 数据的结构与扰动稳定性完成自适应稀疏选择。**

---

# 1. 不允许发生的研究方向漂移

除非用户明确提出重新设计研究，否则后续 Codex **不得**自动把项目改成以下方向：

- 不改成普通 supervised classification。
- 不改成 segmentation network 训练。
- 不改成 diffusion model。
- 不改成 Qwen/VLM/LLM 项目。
- 不改成 private colorectal CT 项目。
- 不改成只做 Liver CT 的单数据集论文。
- 不把 DINOv3 fine-tuning 作为主方法。
- 不加入异常标签、疾病类别标签或 lesion mask 来训练 ASLS/U-VaRFS。
- 不把复杂 decoder / segmentation head / classifier 作为主 detector。
- 不把 anomaly-aware supervised feature selection 引入主方法。
- 不因为某个 baseline 更强就删除 ASLS 或 U-VaRFS 主线。
- 不把随机层/随机特征选择改成主方法；它们只用于 baseline。
- 不把 PCA 改成主方法；PCA 是 feature-selection baseline。
- 不把工程加速写成论文核心创新。

如果用户只要求“修 bug / 提速 / 继续实验”，默认意味着：

> **保持上述研究设计完全不变，仅修改工程实现。**

---

# 2. 数据集与实验协议锁定

## 2.1 主 benchmark

必须保留 **BMAD 全部 6 个 benchmark**：

| Internal name | BMAD/source | Modality | Train | Valid | Test | Pixel mask |
|---|---|---|---:|---:|---:|---|
| Brain | BraTS2021 slice | MRI | 7500 | 83 | 3715 | Yes |
| liver | BTCV + LiTS / released Liver AD | CT | 1542 | 166 | 1493 | Yes |
| RESC | RESC | Retinal OCT | 4297 | 115 | 1805 | Yes |
| OCT2017 | OCT2017 | Retinal OCT | 26315 | 32 | 968 | No |
| xray | RSNA | Chest X-ray | 8000 | 1490 | 17194 | No |
| camelyon16 | Camelyon16 | Histopathology | 5088 | 236 | 1997 | No |

**Liver CT 是 CT 重点分析数据集，但不是唯一实验数据。**
全部 6 个数据集用于证明方法跨模态泛化。

## 2.2 当前数据入口

用户自行下载 BMAD 压缩包，固定放在：

```text
data/archives/
```

当前代码优先支持 BMAD 已整理好的 6 个 AD 压缩包，例如：

```text
Brain_AD.zip
Chest-AD.zip
Histopathology_AD.zip
Liver_AD.zip
Retina_OCT2017_AD.zip
Retina_RESC_AD.zip
```

代码自动：

```text
data/archives/
    ↓
data/_extracted/
    ↓
normalize / preprocess
    ↓
data/processed/BMAD/
```

原始压缩包不得被修改或删除。

Liver released archive 的特殊结构已经兼容：

```text
Liver/Train/hist_DIY/
├── train/good/
├── valid/img/good/
├── valid/img/Ungood/
├── valid/label/Ungood/
├── test/img/good/
├── test/img/Ungood/
└── test/label/Ungood/
```

不要重新要求用户下载 BTCV/LiTS，除非 released Liver archive 本身缺失或损坏。

## 2.3 无标签协议

训练/拟合阶段的定义必须保持：

- BMAD 官方 normal-reference train split 用于建立正常表征。
- ASLS 不使用 anomaly labels。
- U-VaRFS 不使用 anomaly labels。
- ASLS/U-VaRFS 不使用 lesion masks。
- lesion masks 只用于最终 pixel-level evaluation。
- abnormal test labels 只用于最终 AUROC/AUPRC 评价。
- 不允许通过 test labels 选择 layer、lambda、feature budget 或 memory hyperparameters。

这里的“无监督/无标签”指：

> **在已知 normal-reference training split 的 one-class anomaly detection protocol 中，特征选择不使用异常类别或病灶标注。**

不要把它误写成“训练集本身完全不知道哪些样本正常”的 contamination setting。

---

# 3. Backbone 锁定

主 backbone：

> **Frozen DINOv3 ViT-S+**

当前由 ModelScope 下载至：

```text
models/dinov3_vitsplus/
```

默认 ModelScope ID：

```text
keras/dinov3_vit_small_plus_lvd1689m
```

代码应动态读取：

- number of layers；
- hidden dimension；
- patch size；
- number of register tokens。

当前主实验使用所有 DINOv3 Transformer 层作为 layer-selection 候选。

**主方法不允许预先固定 L3/L6/L9/L12。**

固定 `[3,6,9,12]` 只允许作为 baseline / ablation。

DINOv3 backbone 全程 frozen：

- no supervised fine-tuning；
- no LoRA；
- no anomaly-label adaptation；
- no segmentation fine-tuning。

---

# 4. 核心创新模块一：ASLS

全称：

> **Adaptive Sparse Layer Selection**

研究问题：

> DINOv3 的所有层是否都需要？不同医学模态是否依赖不同的 DINO hierarchy？

输入：

```text
DINO layer 1 ... layer L
```

使用 normal training images 的 pooled/patch representation 与轻微 perturbation variability，评价每层：

1. 是否保持 full hierarchy 的 normal representation geometry；
2. 是否对 nuisance perturbation 稳定；
3. 是否可以用更少的层完成表示。

目标概念：

```text
representation geometry preservation
+ perturbation stability
+ layer sparsity
```

## 4.1 当前代码实现必须准确描述

当前 `uvarfs/asls.py` 使用：

- 每层 cosine Gram；
- 所有层平均得到 consensus geometry；
- 每层 variability；
- sigmoid continuous layer probability；
- Adam 优化；
- 最后按 gate probability 排序；
- 选择满足 geometry tolerance 的最小层集合。

**注意：当前代码不是严格的 Hard-Concrete L0 implementation。**

因此：

- 不要在论文/README/回复中声称“当前实现已经是 Hard-Concrete”，除非代码以后真的实现它；
- 可以描述为 continuous sparse layer gates / adaptive layer weights + geometry-constrained discrete selection；
- 若后续实现 Hard-Concrete，必须作为明确的方法升级并重新做 ablation，不能静默替换。

当前无标签选择规则：

> 在 geometry error 不超过预设 tolerance 的条件下选择尽量少的层。

默认：

```yaml
geometry_tolerance: 0.05
min_layers: 2
max_layers: 6
```

---

# 5. 核心创新模块二：U-VaRFS

全称：

> **Unsupervised Variability-Regularized Feature Selection**

这是本项目第二个核心方法模块，不能被 PCA/Random feature replacement 取代。

## 5.1 从 VaRFS 到 U-VaRFS 的核心修改

原监督 VaRFS 思路：

```text
discriminability
+ variability regularization
+ sparsity
```

本项目移除需要标签的 discriminability term，改为：

```text
normal representation preservation
+ variability regularization
+ sparsity
```

因此 U-VaRFS 不需要异常标签。

## 5.2 数学主线

ASLS 选择层后，将对应 latent features 拼接：

[
X in mathbb{R}^{N	imes M}.
]

为每个 latent dimension 设置：

[
0le w_jle1.
]

完整 normal representation：

[
K = XX^T.
]

加权 representation：

[
K_w=Xoperatorname{diag}(w)X^T.
]

无监督 representation-preservation term：

[
mathcal{L}_{rep}
=
rac12
left|
XX^T-
Xoperatorname{diag}(w)X^T
ight|_F^2.
]

对每种轻微扰动构建 feature variability：

[
u_v^j=
rac{
mathbb{E}_i[(f_j(x_i)-f_j(T_v(x_i)))^2]
}{
operatorname{Var}_i(f_j(x_i))+epsilon
}.
]

组成：

[
P=[u_1,ldots,u_V],
qquad
R=PP^T.
]

最终目标保持为：

[
oxed{
min_{0le wle1}
rac12
left|
XX^T-Xoperatorname{diag}(w)X^T
ight|_F^2
+
eta w^TRw
+
lambda|w|_1
}
]

等价的 feature-space 形式：

[
H=(X^TX)odot(X^TX),
]

[
oxed{
min_w
rac12(mathbf1-w)^TH(mathbf1-w)
+
eta w^TRw
+
lambda|w|_1
}
]

这条数学定义属于**研究主线，不要为了提速而改变目标函数**。

## 5.3 当前 solver

当前 `uvarfs/u_varfs.py` 已改为：

> **GPU batched FISTA**

所有 lambda 同时优化，而不是旧版本逐 lambda FISTA。

当前仍使用同一目标函数；主要工程等价变换：

[
Rw=P(P^Tw)
]

因此无需显式构造大的 (R=PP^T)。

当前默认：

```yaml
beta: 0.002
lambda_grid:
  [0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
   0.0005, 0.0002, 0.0001, 0.00005,
   0.00002, 0.00001, 0.000005, 0.000002, 0.000001]
geometry_tolerance: 0.05
max_iter: 2000
tol: 1e-5
min_features: 32
max_features: 256
```

lambda selection 必须保持 label-free：

> 从 lambda path 中选择满足 geometry tolerance 和最小 feature 数约束的最稀疏解；若不存在 feasible solution，则选 geometry error 最小者。

禁止使用 anomaly AUROC 来挑 lambda。

---

# 6. Perturbation 的角色

扰动的唯一研究作用是：

> **估计正常特征对 nuisance variation 的敏感性。**

它不是 synthetic anomaly generation。

允许的轻度扰动：

- Gaussian noise；
- Gaussian blur；
- mild gamma / contrast；
- mild resolution degradation / restoration。

原则：

- 不改变医学语义；
- 不主动制造病灶；
- 不使用异常 mask；
- 不把 perturbation classifier 变成新任务。

---

# 7. Detector 必须保持简单

论文需要证明性能来自：

> **ASLS + U-VaRFS 的 representation selection**

而不是复杂 detector。

因此主 detector 固定为：

```text
normal memory bank
+ cosine 1-NN
```

Patch anomaly score：

[
A(q)=1-max_{minmathcal M}cos(q,m).
]

Image score：

[
A_{img}=operatorname{Mean}(operatorname{TopK}(A_{patch})).
]

默认 Top 1%。

Pixel anomaly map：

- patch anomaly score；
- reshape patch grid；
- bilinear interpolation 到图像尺寸。

不要把 segmentation decoder、classifier head 或 trainable reconstruction network 加入主方法。

---

# 8. 当前 Memory Bank 设计

为了控制 BMAD 全六数据集运行成本，当前默认：

```yaml
memory_size: 20000
memory_images: 1024
memory_patches_per_image: 32
```

这属于**统一的计算预算**，所有方法必须公平使用相同 protocol。

当前 anomaly NN backend：

> `TorchCosineIndex`

使用 GPU FP16 exact cosine 1-NN。

旧版 faiss-cpu 是性能瓶颈；FAISS 现在仅作为 fallback。

GPU NN、memory sampling、query chunk 都属于**工程优化，不是论文创新**。

---

# 9. 主方法与 baseline 的身份必须严格区分

## 9.1 唯一主方法

```text
ASLS + U-VaRFS
```

代码 method name：

```text
main
```

## 9.2 Layer baselines

必须保留：

- `last_raw`
- `fixed4_raw`
- `all_raw`
- `asls_raw`
- random layer baseline

它们回答：

> 自适应 layer sparsity 是否真的优于人工层、全层或随机层？

## 9.3 Feature baselines

必须保留/逐步完善：

- Raw；
- PCA；
- Random feature subset；
- U-VaRFS。

当前代码包括：

- `asls_pca`
- `asls_random_seed*`
- `fixed4_uvarfs`
- `all_uvarfs`
- `random4_uvarfs_seed*`

随机 baseline 默认 seeds：

```text
42
123
3407
2026
2027
```

Random baseline 必须报告 mean ± SD，不能挑最好 seed 与 main 比。

## 9.4 外部 benchmark baselines

论文阶段可加入：

- PaDiM；
- PatchCore；
- RD4AD；
- CFLOW；
- STFPM；
- 合理的近期 DINO-based AD methods。

这些是**外部性能比较**，不得改变主方法定义。

---

# 10. 评价指标锁定

## 10.1 全部 6 个数据集

必须至少报告：

- Image AUROC；
- Image AUPRC。

## 10.2 有 mask 的数据集

Brain / Liver / RESC：

- Pixel AUROC；
- Pixel AUPRC；
- AUPRO。

## 10.3 必须同时报告稀疏/效率指标

- selected DINO layers；
- selected layer count；
- final feature dimension；
- candidate dimension；
- compression ratio；
- memory-bank size；
- peak GPU memory；
- fit time；
- memory-build time；
- evaluation time。

核心论文结论不能只写“AUROC 更高”。

必须回答：

> 用多少层、多少 latent dimensions、多少 memory/compute 换来了什么性能？

---

# 11. 预期论文核心问题

所有实验和代码修改应服务于以下 4 个核心问题：

## Q1. DINOv3 是否存在可显著裁剪的层冗余？

比较：

```text
Last
Fixed-4
All
Random-K
ASLS
```

## Q2. ASLS 是否优于人工/随机 layer selection？

必须用多随机 seed 证明。

## Q3. U-VaRFS 是否优于普通降维/随机 feature selection？

比较：

```text
Raw
PCA
Random
U-VaRFS
```

在相同 detector、相同 layer input、相同 memory protocol 下进行。

## Q4. 稀疏表示是否能维持/改善性能并显著降低开销？

理想叙事：

```text
12 DINO layers
    ↓
~3–5 selected layers

large concatenated representation
    ↓
~32–256 retained dimensions

AUROC/AUPRO non-inferior or improved
+ lower memory
+ lower matching cost
```

不要把“刷新所有 BMAD SOTA”设为必要成功条件。

---

# 12. 论文创新的正确表述

可以主张/验证的创新方向：

> **normal-only nuisance variability estimation + adaptive DINO layer sparsification + unsupervised variability-regularized latent feature selection + simple normal-memory anomaly matching**

更具体地说：

1. 从 DINOv3 全层中进行 normal-only adaptive sparse layer selection；
2. 将 VaRFS 从 supervised discriminability 改为 normal representation preservation，使其不依赖 anomaly labels；
3. 同时在 layer 维度和 latent-feature 维度进行可解释稀疏化；
4. 在 BMAD 六数据集上研究不同医学模态的 DINO hierarchy selection pattern；
5. 同时评价 anomaly performance 与 representation compression。

禁止宣称：

- first DINOv3 anomaly detection；
- first multi-layer DINO anomaly detection；
- first DINO feature selection；
- first adaptive DINO weighting；
- first sparsity in anomaly detection。

这些表述已有较高 novelty 风险。

---

# 13. 工程优化与方法创新必须分开

以下内容**不是论文方法创新**，只是为了把实验跑完：

- CUDA exact cosine NN；
- batched GPU FISTA；
- low-rank (P(P^TW)) 实现；
- CUDA PCA baseline；
- TF32；
- mixed precision；
- memory-bank sampling；
- progress heartbeat；
- resume/checkpoint；
- archive extraction cache；
- multiprocessing preprocessing；
- 双 GPU dataset-level parallel execution（若以后加入）。

Codex 后续不得把这些包装成核心科学贡献。

---

# 14. 当前运行入口

完整运行：

```bash
bash run.sh
```

数据/模型已经准备好时：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

只跑 Liver 调试：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

只预处理：

```bash
PREPROCESS_ONLY=1 bash run.sh
```

强制重跑实验：

```bash
FORCE_EXPERIMENT=1 SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

注意：

- 用户自行创建/维护 conda 环境；
- `run.sh` 不允许自动 pip/conda install；
- 模型权重可从 ModelScope 自动下载；
- BMAD 数据由用户手动下载压缩包。

---

# 15. 当前实验代码关键文件

```text
run.sh
│
├── scripts/prepare_bmad.py
├── scripts/download_model.py
└── scripts/run_all.py
      │
      ├── uvarfs/data.py
      ├── uvarfs/dinov3.py
      ├── uvarfs/pipeline_fit.py
      │     ├── uvarfs/asls.py
      │     ├── uvarfs/u_varfs.py
      │     └── uvarfs/transforms.py
      │
      └── uvarfs/pipeline_eval.py
            ├── uvarfs/memory.py
            └── uvarfs/metrics.py
```

各文件职责：

- `preprocess.py`：原始/released BMAD archive 标准化；
- `dinov3.py`：冻结 DINOv3 全层 patch feature；
- `asls.py`：自适应层选择；
- `u_varfs.py`：无标签 variability-regularized feature selection；
- `pipeline_fit.py`：normal-only fit 与 baseline specs；
- `memory.py`：normal memory 与 cosine NN；
- `pipeline_eval.py`：image/pixel anomaly evaluation；
- `metrics.py`：AUROC/AUPRC/AUPRO；
- `run_all.py`：六数据集总调度、checkpoint 与汇总。

---

# 16. 当前加速版状态

当前 experiment version：

```text
gpu-eval-v5-normal-audit
```

原因：

旧版曾出现：

- DINO forward 完成；
- GPU memory 仍占用；
- `nvidia-smi` utilization 长期 0%；
- 日志在 `fit-features 100%` 后长时间不更新。

已定位主要工程瓶颈包括：

1. faiss-cpu 最近邻；
2. CPU PCA transform；
3. Pixel AUPRO 阈值矩阵；
4. Python per-patch reservoir；
5. U-VaRFS 每个 lambda 串行 FISTA；
6. 每次 FISTA iteration CPU/GPU synchronization；
7. PCA fit 使用 sklearn CPU。

当前已改为：

- Torch CUDA cosine NN；
- GPU PCA transform；
- CUDA `torch.pca_lowrank` PCA fit；
- vectorized AUPRO；
- sampled memory construction；
- batched all-lambda GPU FISTA；
- reduced synchronization；
- heartbeat；
- dataset checkpoint/resume。

**这些优化必须保持数学协议一致。**

如果为了提速需要改变 sampling budget、feature fit sample 数、memory size 等实验参数：

- 必须写入 config；
- 所有比较方法统一；
- 不能只给 main 方法更多资源；
- 必须记录到结果文件。

---

# 17. 当前运行/调试状态交接

最近日志证明：

- BMAD 六个 released AD archives 均可成功识别；
- Brain 完整运行成功；
- Camelyon16 完整运行成功；
- GPU NN 后 Brain test 已从旧版约 1 小时级缩短到约分钟级；
- 最近一次旧 solver 运行在 Liver `fit-features 100%` 后长时间无日志；
- 已将 U-VaRFS 更新为 batched GPU FISTA，并给 ASLS/U-VaRFS/PCA 全部加入显式进度日志。

因此 Codex 接手后：

**不要重新猜测“是不是数据集损坏或 DINOv3 卡死”。**
优先从当前 `main` 最新代码继续验证：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

观察日志应依次看到：

```text
[liver] stage 1/3: fit ASLS/U-VaRFS
fit-features ...
[fit] ASLS start ...
[fit] ASLS done ...
[fit] U-VaRFS main ...
[uvarfs:main] iter=...
...
[pca] start ...
[pca] done ...
[liver] fit complete ...
[liver] stage 2/3 ...
[liver] stage 3/3 ...
```

如果再次长时间停住：

1. 记录最后一条 heartbeat；
2. 检查 CPU/RAM/GPU；
3. 定位具体阶段；
4. 优化该阶段；
5. **不要改变研究方法来规避计算问题。**

---

# 18. 结果文件与断点策略

每个数据集：

```text
results/<dataset>/
├── asls.json
├── uvarfs_*.json
├── memory_progress.json
├── eval_progress.json
└── metrics.csv
```

全局：

```text
results/
├── all_metrics.partial.csv
├── all_metrics.csv
├── summary_metrics.csv
└── layer_selection.csv
```

只有包含当前：

```text
experiment_version = gpu-eval-v5-normal-audit
```

的完整 dataset result 才允许 resume。

不同 experiment version 的旧结果不能静默混合。

---

# 19. Codex 修改代码前的强制检查清单

任何实质性代码修改前先回答：

1. 这个修改是否改变研究问题？
2. 是否改变 ASLS 定义？
3. 是否改变 U-VaRFS 目标函数？
4. 是否引入 anomaly labels / masks 到训练？
5. 是否改变主 detector？
6. 是否对 main 与 baseline 使用不同数据预算？
7. 是否会使旧结果与新结果不可直接比较？
8. 是否需要 bump `EXPERIMENT_VERSION`？

如果 2–7 任一答案为“是”，除非用户明确要求改变实验设计，否则不要直接实施。

---

# 20. Codex 修 bug / 提速时的默认行为

默认允许：

- vectorization；
- batched GEMM；
- mixed precision；
- GPU implementation；
- caching；
- dataloader tuning；
- multi-process preprocessing；
- dataset-level multi-GPU；
- checkpoint；
- resume；
- deterministic manifest；
- log/heartbeat；
- equivalent numerical reformulation。

默认不允许：

- 删掉主实验模块；
- 换研究任务；
- 用 test AUROC 自动调参；
- 引入监督标签；
- 修改数学目标后仍称为同一实验；
- 因速度慢就把 all-6 benchmark 缩成单数据集主论文；
- 只保留对 main 有利的 baseline。

---

# 21. 双 RTX 3090 的后续合理优化方向

当前主要代码是单 GPU。

若继续加速，优先方案：

> **dataset-level two-process parallelism**

例如：

```text
GPU0:
Brain
liver
OCT2017

GPU1:
camelyon16
RESC
xray
```

最后合并结果。

不要优先做 DINO tensor-parallel / model-parallel，因为：

- DINOv3 ViT-S+ 很小；
- 单卡显存足够；
- 跨卡通信不值得；
- benchmark 之间天然独立。

多 GPU 只是工程加速，不改变实验协议。

---

# 22. 最终论文需要的主图/主表

## Figure 1
方法框架：

```text
Frozen DINOv3
→ ASLS
→ U-VaRFS
→ normal memory
→ anomaly detection/localization
```

## Figure 2
BMAD × DINOv3 layer-selection heatmap。

横轴：

```text
Layer 1 ... Layer 12
```

纵轴：

```text
Brain
Liver
RESC
OCT2017
X-ray
Camelyon16
```

## Figure 3
feature sparsity vs AUROC / AUPRO。

## Figure 4
Brain / Liver / RESC anomaly maps。

## Table 1
BMAD 全六数据集主结果。

## Table 2
ASLS ablation。

## Table 3
U-VaRFS vs PCA / Random / Raw。

## Table 4
性能—压缩—计算开销。

---

# 23. 项目成功判据

本项目成功不等于必须所有 benchmark SOTA。

优先级依次是：

1. **自适应稀疏有效**；
2. **ASLS 优于随机/固定层**；
3. **U-VaRFS 优于 Random/PCA 或至少在更低维度下 non-inferior**；
4. **显著压缩 feature dimension / memory matching cost**；
5. **在六 BMAD 数据集上具有跨模态一致性**；
6. Liver CT 等带 mask 数据有合理 localization；
7. 如部分数据集性能提升、部分持平，也可以通过“redundancy/compression + robustness”形成论文故事。

如果性能失败，应优先诊断：

- ASLS layer selection；
- variability definition；
- geometry preservation；
- lambda path；
- feature scaling；
- memory protocol；
- image anomaly aggregation；

而不是立即改变研究任务。

---

# 24. 与用户沟通时必须保持的表述

后续回复用户时，应始终把项目称为：

> **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究。**

如果只在 Liver 调试，也应说明：

> Liver 是当前调试数据集 / CT 主分析数据集，最终实验仍是 BMAD 全 6 数据集。

如果讨论速度：

> 说明是工程瓶颈，不是改变研究方法的理由。

如果讨论新模块：

> 先判断它是 baseline、ablation、工程优化还是主方法变更。

不要在没有用户明确确认的情况下创建新的“主线”。

---

# 25. 当前最优先下一步

Codex 接手后优先级：

1. 拉取最新 `main`；
2. 不改研究设计；
3. 先单独验证 Liver 新版 batched U-VaRFS：
   ```bash
   SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
   ```
   v4 六数据集已完成，mask 归一化完整；已有数据/模型时，新版可跳过准备阶段。
4. 确认：
   - ASLS 有日志；
   - batched FISTA 有日志；
   - PCA GPU fit 有日志；
   - memory/test 正常；
5. Liver 完成后再全 6 benchmark；
6. 再考虑双 3090 dataset-level parallel runner；
7. 结果稳定后补缺失 baseline 与论文图表。

**不要在第 3 步尚未验证前重新大规模重构项目。**

---

# 26. 最终不可漂移的研究定义

[
oxed{
	ext{BMAD all six}
+
	ext{Frozen DINOv3}
+
	ext{Adaptive Sparse Layer Selection}
+
	ext{Unsupervised VaRFS}
+
	ext{Simple Normal Memory Matching}
}
]

其中：

[
oxed{
	ext{ASLS selects layers}
}
]

而：

[
oxed{
	ext{U-VaRFS selects latent dimensions without anomaly labels}
}
]

这是本仓库所有后续开发、实验、论文与问答必须围绕的唯一主线。

---

# 27. 2026-10-06 原始协议实现修正交接

本节保留 v4 当时的历史交接：版本为 `gpu-eval-v4-protocol-fixes`，默认结果目录为 `results/gpu-eval-v4-protocol-fixes/`。当前 v4 已在服务器验证全六数据集，最新状态见第 29 节；第 17 节属于更早运行历史。

本次按原主线修正：总体 normal variability 统计、每个 lambda 的预算后几何检查、FISTA 投影残差与迭代上限、mask 配对和 pixel metric 数学、同 K 随机层 baseline、统一 memory sampling 与严格 checkpoint 指纹。ASLS sigmoid/Adam 目标与 U-VaRFS 目标保持原样。ASLS 饱和只增加诊断，没有切换 Hard-Concrete 或改目标函数。

当前默认方法共 28 个，新增 `randomk_raw_seed*` / `randomk_uvarfs_seed*`，保留原 18 个方法及全部 5 个随机 seeds。FISTA 最大迭代数由 400 提高到 2000，仅为数值求解精度；不能宣称每个 fit 均收敛。所有数据与表示预算不变。无预算内可行解时，必须报告 `feasible=false`，不能将 fallback 描述为满足 geometry tolerance。

首次验证 Liver 应保留 preprocessing，以更新 `NORMALIZE_VERSION=3`：

```bash
SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

或先 `PREPROCESS_ONLY=1 DATASETS=liver bash run.sh`，再 `python scripts/run_all.py --dataset liver --check-data-only`；预检不加载 DINOv3。Liver 成功后仍须运行六数据集，第一次全量也保留 preprocessing。

本地只能验证数学、I/O 与 CPU fixture 流程；当前工作机缺少真实 BMAD 数据、模型与 CUDA PyTorch，尚未得到 v4 真实性能结果。具体修改与统计/效率口径见 [PROTOCOL_FIXES.md](PROTOCOL_FIXES.md)。

---

# 28. 仓库同步默认设置（用户于 2026-10-06 明确授权）

用户要求：**每次改进完成后，将改进结果同步到仓库，作为本项目默认行为。**

- 每轮改进完成且必要检查通过后，将相关代码、配置、测试、交接文档和结果分析提交到 Git，并推送到配置的远程仓库；沿用当前工作分支，除非用户另有指定。
- 该授权持续有效，后续完成改进时直接执行提交与推送，无需再次确认。
- 可共享的实验摘要、分析报告与图表放入可跟踪的 `reports/`，明确实验版本与验证边界；实验运行目录继续遵循 `.gitignore` 和既有原始数据/模型管理协议。
- 提交包含本轮工作相关文件；已有用户文件按其原有用途保留。同步前核对远程分支状态，使用正常提交和推送流程。
- 完成后告知提交号与同步状态；网络、认证或分支保护导致推送失败时保留本地提交，明确报告尚未同步成功。

---

# 29. 2026-10-06 v4 回传与 v5 normal-only 诊断交接

v4 全六数据集 × 28 方法完整；Brain/Liver/RESC 异常 mask 全覆盖，全部 image/pixel 指标、逐图预测和来源信息已复核。分析见 [v4 报告](reports/2026-10-06-v4-analysis/analysis.md)。旧版本产物不可改写/混合。

Main Macro Image AUROC=74.26%，低于 Fixed-4 Raw=78.12% 与同 K Random Raw mean=79.47%；Main 仅在 Liver/OCT2017 高于同 K Random baseline。Liver AUPRO=95.10% 是正向证据，但 Brain 明显较差。主 U-VaRFS 六个最终表示均不可行（geometry error≈0.052–0.071）；投影残差较小，不能将迭代变化未收敛与几何不可行混为一谈。

当前版本为 `gpu-eval-v5-normal-audit`，独立输出 `results/gpu-eval-v5-normal-audit/`。具体说明见 [RESULT_DRIVEN_FIXES.md](RESULT_DRIVEN_FIXES.md)：

- ASLS/U-VaRFS 目标拟合使用 highest float32 matmul precision，退出恢复原精度；原 frozen backbone、detector、loss 与所有实验预算保持。
- Batched FISTA 按 lambda 重启动量，记录原目标分项、box optimality gap 与独立数值诊断；beta、lambda path、几何容差、维度预算和原选择规则不变。
- `lambda=0` 只用于不可行原因的 normal-only 诊断，明确 `selection_candidate=false`，不参与 main 选择。
- **Main 默认 ASLS 仍是 pooled geometry**。增加 `asls_patch_raw`/`asls_patch_uvarfs` 两项消融，使用已有 fit patches、原 loss、sigmoid/Adam/离散规则和等价 Gram 内积，共 30 方法；原 28 方法与全部五 seeds 保留。
- 主 ASLS geometry 输入切换按第 19 节等待用户明确选择。不得据 test AUROC 选择 pooled/patch 版本，不得静默替换为 Hard-Concrete 或改目标函数。
- Mask 连通域共享与 pixel histogram/PRO 排序复用只影响最终评价开销。与 v4 的独立 CPU fixture 对照计数及指标完全一致，不代表已验证服务器总加速。
- 不向含其他版本 CSV 的目录写入；指纹统一 POSIX 路径和 LF。

本地工作机缺少真实数据、模型和 CUDA PyTorch，尚未取得 v5 真实性能。服务器先执行 `SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh`，检查正常拟合与 pixel 指标，再运行全六。不得用本轮评价标签调 beta、lambda、层数或其他训练参数。
