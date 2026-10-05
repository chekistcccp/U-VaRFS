# BMAD 原始压缩包固定放置位置

请把你自行下载的 **全部 BMAD 压缩包** 直接放到：

```text
U-VaRFS/
└── data/
    └── archives/
        ├── <BraTS2021 压缩包>
        ├── <BTCV/ATLAS 压缩包>
        ├── <LiTS 压缩包>
        ├── <RESC 压缩包>
        ├── <OCT2017 压缩包>
        ├── <RSNA 压缩包>
        └── <Camelyon16 压缩包>
```

不需要手工解压，不要求改名，也不要求事先整理内部目录。**优先支持 BMAD 官方已经整理好的 `Brain_AD.zip`、`Chest-AD.zip`、`Histopathology_AD.zip`、`Liver_AD.zip`、`Retina_OCT2017_AD.zip`、`Retina_RESC_AD.zip` 这类 AD 数据包；同时保留对 BraTS/BTCV/LiTS/RSNA 等原始源数据包的重建支持。**

支持的压缩格式：

```text
.zip
.tar
.tar.gz
.tgz
.tar.xz
.txz
.7z
```

> 注意：`.nii.gz` 是医学影像文件，不会被当成普通压缩包继续解压。

## 自动处理流程

运行：

```bash
bash run.sh
```

代码会自动执行：

```text
data/archives/
      ↓
扫描全部压缩包
      ↓
自动解压
      ↓
若压缩包内部仍有压缩包，则继续递归解压
      ↓
data/_extracted/
      ↓
递归识别 BraTS / BTCV / LiTS / RESC / OCT2017 / RSNA / Camelyon16
      ↓
BMAD 风格预处理
      ↓
data/processed/BMAD/
```

原始压缩包不会被修改或删除。

## 解压缓存

解压结果保存到：

```text
data/_extracted/
```

每个压缩包会根据：

- 原始路径
- 文件大小
- 修改时间

生成缓存指纹。压缩包没有变化时，重复运行不会重新解压。

如果你替换了压缩包，新文件会得到新的缓存目录；成功解压后，代码会自动清理不再对应当前 `data/archives/` 内容的旧缓存，避免重复或旧版本样本混入。

## 预处理输出

统一输出：

```text
data/processed/BMAD/
├── Brain/
├── liver/
├── RESC/
├── OCT2017/
├── xray/
├── camelyon16/
└── preprocess_summary.json
```

每个数据集还会生成：

```text
preprocess_manifest.json
```

记录识别到的原始来源、划分规则和样本数量。

## 六类原始数据的自动识别规则

### 1. Brain MRI — BraTS2021

代码递归寻找：

```text
*_flair.nii.gz
*_seg.nii.gz
```

自动执行：

- FLAIR 提取；
- 轴向 slice 60–99；
- 正常训练 slice 仅保留 segmentation 为空者；
- 异常 slice 生成像素级 mask；
- 使用仓库内 `metadata/brats_splits.json` 的 BMAD 划分信息。

### 2. Liver CT — BTCV/ATLAS + LiTS

BTCV/ATLAS 自动寻找：

```text
imgXXXX.nii(.gz)
labelXXXX.nii(.gz)
```

LiTS 自动寻找：

```text
volume-*.nii(.gz)
segmentation-*.nii(.gz)
```

自动执行肝区 masking、翻转、直方图均衡、肿瘤 mask 生成和确定性 validation/test 划分。

### 3. RESC

代码递归寻找原始 P-Net 风格目录：

```text
train/images/
test/images/
test/lesion_mask/
test/normal_images/
```

自动重建 BMAD 的 train/valid/test。

### 4. OCT2017

代码递归寻找：

```text
train/NORMAL/
train/CNV/
train/DME/
train/DRUSEN/
test/NORMAL/
test/CNV/
test/DME/
test/DRUSEN/
```

仅 NORMAL 用于 one-class training。

### 5. RSNA Chest X-ray

代码递归寻找：

```text
stage_2_train_images/*.dcm
stage_2_detailed_class_info.csv
```

自动将 DICOM 转为 PNG，并构造 BMAD one-class 训练和测试划分。

### 6. Camelyon16

代码递归寻找所有：

```text
*.tif
*.tiff
```

并使用仓库：

```text
metadata/camelyon16/*.txt
```

中的 BMAD 官方 patch 坐标，自动裁取 256×256 patch。

Camelyon16 需要：

- Python 包 `openslide-python`
- 系统 OpenSlide 动态库

## 首次建议：只跑预处理

```bash
PREPROCESS_ONLY=1 bash run.sh
```

如果成功，检查：

```text
data/processed/BMAD/preprocess_summary.json
```

再运行全部实验：

```bash
bash run.sh
```

## 强制重新预处理

```bash
FORCE_PREPROCESS=1 PREPROCESS_ONLY=1 bash run.sh
```

## 只处理部分数据

```bash
DATASETS=brain,liver PREPROCESS_ONLY=1 bash run.sh
```

如果压缩包不存在、损坏，或无法从解压结果中识别所需原始结构，预处理阶段会直接报错并停止，不会继续运行后续实验。


## 已整理 BMAD AD 数据包的特殊兼容

BMAD 六个整理包的内部目录并不完全统一。当前代码会自动标准化：

- Brain：`train/valid/test -> good/Ungood -> img/anomaly_mask`
- Chest/OCT2017/RESC：自动把官方常见的 `val` / `Val/val` 统一为内部 `valid`
- Camelyon16：直接图片目录自动转为统一的 `img/`
- Liver：专门兼容官方 `Liver/Train/hist_DIY`：
  - `train/good`
  - `valid/img/good`
  - `valid/img/Ungood`
  - `valid/label/Ungood`
  - `test/img/good`
  - `test/img/Ungood`
  - `test/label/Ungood`

最终全部转换成项目统一结构，不再要求 Liver_AD 包中存在 BTCV/LiTS NIfTI。
