# U-VaRFS 实现与运行说明

## 现在的数据与环境策略

当前版本不再自动安装 Python 包，也不再自动下载 BMAD 数据。

你负责：

1. 创建并激活实验环境；
2. 按 `requirements.txt` 安装依赖；
3. 将 BMAD 对应的**原始数据或原始压缩包**放到仓库根目录 `data/`。

代码负责：

1. 检查环境；
2. 从 ModelScope 下载 DINOv3 权重到 `models/dinov3_vitsplus/`；
3. 自动解压和识别 `data/` 中的原始 BMAD 来源数据；
4. 将它们预处理到 `data/processed/BMAD/`；
5. 运行 DINOv3 + ASLS + U-VaRFS 及全部 baseline；
6. 计算 image/pixel 指标、AUPRO、压缩率、显存和耗时并汇总 CSV。

原始数据的详细期望结构见 [`data/README.md`](data/README.md)。

## 一键运行

激活你已经创建好的环境后：

```bash
bash run.sh
```

`run.sh` **不会执行 pip install/conda install**。

## 首次建议：只检查数据处理

先不下载模型、不跑实验：

```bash
SKIP_MODEL_DOWNLOAD=1 PREPROCESS_ONLY=1 bash run.sh
```

确认 `data/processed/BMAD/` 和 `data/processed/BMAD/preprocess_summary.json` 正常后，再运行完整实验：

```bash
bash run.sh
```

## 数据处理输出

```text
data/
├── <你下载的原始文件/压缩包>
├── _extracted/                 # 自动解压缓存
└── processed/
    └── BMAD/
        ├── Brain/
        ├── liver/
        ├── RESC/
        ├── OCT2017/
        ├── xray/
        ├── camelyon16/
        └── preprocess_summary.json
```

每个数据集都会生成 `preprocess_manifest.json`，记录来源、确定性划分规则和关键数量。

## 预处理原则

- **BraTS2021**：FLAIR + segmentation，轴向 slice 60–99；训练只保留无病灶 slice；异常评估输出 mask。
- **BTCV/ATLAS + LiTS**：BTCV liver label=6 构建正常训练集；LiTS label>0 为肝区、label=2 为肿瘤；使用 BMAD 风格 liver mask、翻转和直方图均衡。
- **RESC**：P-Net 原始 train/images 作为正常训练；test normal/abnormal + lesion mask 确定性拆成 validation/test。
- **OCT2017**：NORMAL 训练；标准 test 中 8 NORMAL + 每个异常类别各 8 张用于 validation，其余 test。
- **RSNA**：DICOM 转 PNG；前 8000 个确定性排序的 Normal 作为训练；剩余样本确定性划 validation/test。
- **Camelyon16**：使用 BMAD 官方公开 coordinate lists，在 level 0 裁剪 256×256 patch。

BMAD 官方部分原始预处理脚本使用了 `set()` / 未排序 `os.listdir()`，因此无法保证跨机器得到同一顺序。本项目将这类步骤改为排序后的确定性规则，并写入 manifest。这样结果可重复，但从原始来源重建的文件不承诺与 BMAD Google Drive 已整理版本逐字节一致。

## 常用控制

只处理部分数据：

```bash
DATASETS=brain,liver PREPROCESS_ONLY=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

强制重新生成预处理数据：

```bash
FORCE_PREPROCESS=1 PREPROCESS_ONLY=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

已有预处理数据时跳过：

```bash
SKIP_PREPROCESS=1 bash run.sh
```

模型已经手工准备好时：

```bash
SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

指定 Python：

```bash
PYTHON=/path/to/conda/env/bin/python bash run.sh
```

## DINOv3

默认 ModelScope ID：

```text
keras/dinov3_vit_small_plus_lvd1689m
```

保存到：

```text
models/dinov3_vitsplus/
```

KerasHub Torch backend 直接读取本地 preset 的 `stage1...stage12`，DINOv3 全程冻结。

## 加速设计

- `torch.inference_mode()`；
- CUDA bf16（支持时）；
- TF32 matmul/cudnn；
- 一个 batch 一次提取所有 DINOv3 层，多 baseline 共享 forward；
- ASLS/U-VaRFS 仅在采样的正常图像和 patch 上拟合；
- U-VaRFS 使用 `H=(X^T X)⊙(X^T X)`，不构造大型 `N×N` Gram；
- FISTA + power iteration；
- reservoir memory bank；
- FAISS cosine 1-NN；
- 预处理的 RSNA DICOM 和 Camelyon WSI 支持多进程。

## 主要结果

```text
results/<dataset>/metrics.csv
results/all_metrics.csv
results/summary_metrics.csv
results/layer_selection.csv
```
