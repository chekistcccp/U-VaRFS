# v10：层输入归一化的完整同轮消融

本项目仍是 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究**，最终使用 BMAD 六数据集与 normal memory exact cosine 1-NN。实验结果、诊断和分析留在本地忽略目录，本文件只记录实现与运行协议。

## 当前默认与授权边界

版本：`gpu-eval-v10-layer-alignment`。当前输出目录：`results/gpu-eval-v10-layer-normalization-ablation/`。

```yaml
representation:
  layer_normalization: none
  ablation_layer_normalization: l2
```

Main 暂时保持原始层特征输入；新增 `layer_l2_*` 完整归一化消融。ASLS patch 主输入与稀疏支持集求解升级已有用户授权，本次不重新询问这些授权。将层 L2 输入用于 Main 涉及原实验输入协议与跨版本可比性，按 AGENTS 第 19 节须等待明确批准。本轮完成可复核实现后再请求该批准；不能按测试 AUROC 从两个分支择优定义 Main。

如获批准，将固定 `layer_normalization: l2`、`ablation_layer_normalization: none`，独立写入 `results/gpu-eval-v10-layer-alignment/`。此时 Main 与其全部 baseline 统一使用层 L2 输入，原始输入全部方法另以 `raw_input_*` 保留。两个配置均有完整 CPU 流程验证；批准前不得静默改变默认 Main。

## 原因与数学范围

ASLS 的 patch 几何使用每层单位向量的 cosine Gram，并等权平均；原下游直接拼接各层原始特征，然后对拼接行归一化。各层数值尺度不同会导致这两个几何定义不一致。这一实施动机来自正常训练表示的结构，无需异常标签或 mask。

对非零 patch 行，令 `z_l = f_l / ||f_l||_2`。选中 K 层后拼接 `z_l` 并对整行归一化，其 cosine Gram 精确等于 `K^-1 sum_l z_l z_l^T`，与 ASLS 离散子集几何一致。实现使用 float32 `F.normalize`；零行保持有限并由 normal audit 记录，不能对零行宣称上述等式。实际 FP16 matching 的舍入误差仍需服务器验证。

归一化是**输入协议消融**，不是等价求解器加速，也不是新的异常检测器。U-VaRFS 仍对指定输入 X 优化原来的 representation preservation + variability + L1 稀疏目标；box `[0,1]`、beta、lambda path、5% 容差、32–256 维预算不变。它保持的是未重新行归一化的加权 Gram，detector 使用再归一化 cosine；两者的误差仍须分别报告。该修改不能保证 U-VaRFS 几何可行、异常性能提升或全局支持集最优。

## 两个分支如何保持公平

- 正常 fit 图像、patch IDs、variability 图像、四种扰动及显式 noise RNG 完全共享。DINO 每次只 forward 一份 normal/perturbed 输入，两个分支复用同一输出。
- `none` 分支保持原有值；`l2` 分支在每层 patch 上归一化。normal 与每种 perturbation 都在对应分支归一化后计算 variability，不能把 raw variability 用于归一化 X。
- ASLS、U-VaRFS、PCA、Random、Raw、固定层/随机层和旧算法对照，在各自分支统一使用该输入。所有原 37 方法及五 seeds 保留，完整另一分支再增加 37 方法，默认共 **74 方法**，全六共 444 行。
- 每个分支的 Random-K 跟随该分支 Main 的实际 K，PCA/Random feature 使用该分支 Main 的维度。两分支可选出不同层/维度，但相同预算上限和无标签规则保持。
- memory 与 query 在 latent selection/PCA 之前采用相同的每层归一化。concat cache 同时以模式和有序层列表索引，防止跨分支复用错误输入。
- memory 图像、patch IDs、每方法 memory_size、Top-1% 与 pixel 评价定义保持。74 方法的总体 fit/memory/eval 时间和峰值显存是共享运行开销，不能当作单一 Main 的开销或声称新版已加速。

## 清单、结果审计与断点

每数据集根目录保存 primary 分支的 ASLS/U-VaRFS/fit/normal audit；另一分支保存于 `layer_l2/` 或 `raw_input/`。新增 `representation_manifest.json` 记录 primary/ablation 角色、全方法输入映射及共享 forward/抽样；CSV 与 layer selection 表保存 `layer_normalization`。

两个 fit manifest 记录各自 variability 空间，以及共享原始 backbone patch norm 统计、正常图像/patch/noise 清单。normal audit 的 full/selected input 指**该分支裁剪前的输入**。历史 `raw` 字样字段作为兼容别名保留，不表示所有分支都使用原始 backbone 数值。

配置、代码、数据、版本指纹和 74 方法完整性继续限制 resume。新版本不能覆盖 v9 目录；切换 primary 输入须使用不同输出目录，不能续接前一配置。`scripts/analyze_results.py` 独立核验两个分支的输入清单与共享预算、U-VaRFS path/旧候选保护、逐图指标及所有随机 seeds，结果分析不参与训练选择。

## 第 19 节强制检查

| 检查 | 本轮结论 |
| --- | --- |
| 改变研究问题 | 否，仍保留六 BMAD、Frozen backbone、ASLS、U-VaRFS、简单 matching |
| 改变 ASLS 定义 | loss/gates/离散规则不变；L2 分支 variability 输入随表示改变，Main 切换待批准 |
| 改变 U-VaRFS 目标函数 | 公式不变；L2 分支的 X/P 输入改变，必须明确记录协议消融 |
| 引入异常 labels/masks 拟合 | 否 |
| 改变 detector | 否，exact cosine 1-NN；输入协议在分支内统一 |
| main/baseline 使用不同数据预算 | 否 |
| 旧新结果不可直接归因比较 | 是，需要独立版本与同轮完整输入对照；Main 输入升级待批准 |
| bump EXPERIMENT_VERSION | 是，v10 |

## 本地验证与服务器运行

已验证单位层拼接与 ASLS 子集 Gram 等价、正层尺度不变性、raw 数值路径保持、混合缓存分离、各自 variability 的独立重算，以及两个 primary 配置下的 74 方法 fit → memory → image/pixel evaluation。对照 manifest 被篡改时审计应拒绝。完整本地回归 81 passed、2 CUDA skipped；本机无真实 BMAD/模型/CUDA，不能宣称真实检测收益。

服务器先验证 Liver（CT 调试数据集），最终仍运行 BMAD 全六：

```bash
git pull --ff-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

核对两套 fit/variability 空间、74 方法、相同 normal/memory 清单、normal geometry 与 pixel 指标，尤其不要将 `feasible=false` fallback 描述为满足容差。未经明确批准，保持当前 raw Main + L2 消融配置。
