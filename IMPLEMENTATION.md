# 实现说明

## 一键运行

```bash
bash run.sh
```

默认行为：

1. 安装 Python 依赖；
2. 使用 ModelScope 将 DINOv3 ViT-S+/16 权重下载到 `experiment/models/dinov3_vitsplus/`；
3. 在 ModelScope 搜索 BMAD 镜像并下载到 `experiment/data/BMAD/`；
4. 对发现的 BMAD 六个数据集依次运行 ASLS、U-VaRFS、memory bank 与全部 baseline；
5. 输出每数据集 `results/<dataset>/metrics.csv`，以及 `results/all_metrics.csv`、`results/summary_metrics.csv` 和 `results/layer_selection.csv`。

## BMAD ModelScope 镜像

BMAD 官方发布页目前仍指向 Google Drive，因此代码不会伪装存在官方 ModelScope 镜像。下载器会：

1. 优先读取 `BMAD_MODELSCOPE_ID`；
2. 否则调用 ModelScope OpenAPI 搜索关键词 `BMAD`；
3. 验证下载内容必须至少包含 6 个具有 `train/` 与 `test/` 的 BMAD 数据目录；
4. 若没有符合条件的镜像则停止，并要求设置一个真实的 ModelScope mirror ID。

示例：

```bash
BMAD_MODELSCOPE_ID=your_namespace/BMAD bash run.sh
```

如果数据已经位于 `experiment/data/BMAD/` 且布局合法，下载步骤自动跳过。

## DINOv3 模型

默认 ModelScope ID：

```text
keras/dinov3_vit_small_plus_lvd1689m
```

KerasHub 使用 Torch backend 加载本地 ModelScope preset，并直接读取 `pyramid_outputs` 的 `stage1...stage12`，无需修改 DINOv3 源码。

## 加速设计

- DINOv3 全冻结，`torch.inference_mode()`；
- CUDA 支持时优先 bf16；
- TF32 matmul/cudnn 打开；
- 一个 batch 一次提取 12 层，多个 baseline 共享同一次前向；
- ASLS/U-VaRFS 仅在采样的正常训练图像/patch 上拟合；
- U-VaRFS 使用 `H=(X^T X)⊙(X^T X)`，避免构造巨大的 `N×N` Gram matrix；
- FISTA + power iteration 估计 Lipschitz 常数；
- memory bank 使用 reservoir sampling，避免保存所有训练 patch；
- FAISS cosine 1-NN；若安装 GPU FAISS 自动搬到 GPU；
- 所有方法共享 DINO 前向，避免 baseline 重复提取特征。

## GPU FAISS

默认 requirements 使用 `faiss-cpu` 以保证兼容性。`run.sh` 默认在检测到 CUDA 12 时自动尝试安装 GPU FAISS；失败会自动恢复 CPU FAISS。需要强制启用可使用：

```bash
INSTALL_FAISS_GPU=1 bash run.sh
```

禁用自动尝试：

```bash
INSTALL_FAISS_GPU=0 bash run.sh
```

## 常用控制

不重复安装依赖：

```bash
SKIP_INSTALL=1 bash run.sh
```

指定 Python：

```bash
PYTHON=/path/to/conda/env/bin/python bash run.sh
```

修改实验规模：编辑 `configs/default.yaml`。
