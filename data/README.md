# `data/` 原始 BMAD 数据放置说明

本目录只放你自行下载的**原始数据/原始压缩包**。`run.sh` 不下载 BMAD 数据，也不会修改原文件。
预处理输出统一写到 `data/processed/BMAD/`；压缩包临时解压到 `data/_extracted/`。

文件夹名不要求完全一致，预处理器会递归识别。推荐布局如下。

## 1. Brain MRI — BraTS2021

```text
data/brats2021/
  BraTS2021_00000/
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_seg.nii.gz
  ...
```

只需要 FLAIR 与 segmentation。代码复现 BMAD 的 60–99 轴向切片规则；正常训练/正常评估切片要求 segmentation 为空，异常评估按 BMAD 步长抽样并生成像素 mask。仓库内 `metadata/brats_splits.json` 保存 BMAD 官方脚本公开的 train/valid-normal/test-normal ID；异常 ID 用剩余患者排序后确定性划分，避免原官方脚本 `set()` 导致的不可复现顺序。

## 2. Liver CT — BTCV/ATLAS + LiTS

BTCV/ATLAS：

```text
data/atlas/Training/img/img0001.nii.gz
 data/atlas/Training/label/label0001.nii.gz
```

LiTS：

```text
data/lits/volume-0.nii(.gz)
data/lits/segmentation-0.nii(.gz)
```

BTCV 的 liver label=6 用于生成 normal training slices；LiTS 中 label>0 为 liver、label=2 为 tumor。代码按 BMAD 原处理方式进行肝区 mask、翻转、直方图均衡；LiTS 确定性排序后取 93 normal + 73 abnormal 为 validation，其余为 test。

## 3. RESC OCT

支持 P-Net 原始发布结构：

```text
data/RESC/
  train/images/<case>/*.png
  test/images/<case>/*.png           # abnormal
  test/lesion_mask/<case>/*.png      # abnormal masks
  test/normal_images/<case>/*.png    # normal
```

全部 train/images 作为正常训练库。原 test 集确定性抽取 65 normal + 50 abnormal（共 115）作为 validation，其余作为 test，因此完整原始 RESC 时得到 BMAD 规模 4297/115/1805。

## 4. OCT2017 / Kermany

```text
data/OCT2017/
  train/{NORMAL,CNV,DME,DRUSEN}/*
  test/{NORMAL,CNV,DME,DRUSEN}/*
```

只用 NORMAL 训练。标准 1000 张 test 中，validation 取 8 NORMAL + 每个异常类别各 8 张；剩余 242 NORMAL + 726 abnormal 为 test，对应 BMAD 的 32/968。

## 5. Chest X-ray — RSNA Pneumonia Detection Challenge

```text
data/rsna/
  stage_2_train_images/*.dcm
  stage_2_detailed_class_info.csv
```

DICOM 自动转 8-bit PNG。Normal 排序后前 8000 张作为 one-class training；剩余样本按类别比例确定性分出 1490 张 validation，其余 test。该数据集没有像素级异常 mask。

## 6. Histopathology — Camelyon16

把官方 `.tif/.tiff` WSI 放在 `data/` 下任意子目录即可，例如：

```text
data/camelyon16/training/normal/*.tif
data/camelyon16/training/tumor/*.tif
data/camelyon16/testing/images/*.tif
```

仓库 `metadata/camelyon16/*.txt` 保存 BMAD 官方公开的 patch 坐标。预处理器按这些坐标以 level 0 裁 256×256 patch，生成 train/valid/test good/Ungood。需要 Python `openslide-python` 以及系统 OpenSlide 动态库。

## 压缩包

`data/` 中的 `.zip/.tar/.tar.gz/.tgz/.7z` 会自动解压到 `data/_extracted/`，不会覆盖或删除原压缩包。若已经自行解压，直接放目录即可。

## 只测试预处理

```bash
PREPROCESS_ONLY=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

强制重新预处理：

```bash
FORCE_PREPROCESS=1 PREPROCESS_ONLY=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

只处理部分数据：

```bash
DATASETS=brain,liver PREPROCESS_ONLY=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```
