# v13：评价统计收尾与部分回传诊断

项目保持 Frozen DINOv3 + Adaptive Sparse Layer Selection + 已批准的 cosine/simplex U-VaRFS + normal memory cosine 1-NN，最终仍为 BMAD 六数据集。当前版本 `gpu-eval-v13-evaluation-finalization`，独立输出 `results/gpu-eval-v13-evaluation-finalization/`。本轮为工程修正，没有新的损失、特征选择规则或训练超参数变化。

## 问题与边界

旧实现的 test 进度只覆盖 DINO/NN 前向。前向后按每个方法独立计算 1000 次 image AUROC bootstrap，再计算最终 pixel 指标；期间没有阶段进度，逐图预测在全部统计后才保存。152 方法会延长这一收尾阶段。看到 `test=100%` 且没有最终 CSV 时，只能确认前向结束；没有退出日志不能断定崩溃，也不能据拟合文件声称检测性能提高。

本轮不使用缺失的 test 指标调整 ASLS、beta、lambda、特征维数、memory 或输入模式。正常训练几何可行、固定支持求解收敛与异常检测性能分别报告。全局 cosine Gram 保持更好也不保证正常 patch 最近邻一致率更高。

## 等价 bootstrap

`metrics.bootstrap_auc` 保持原 `default_rng(seed).integers(0,N,N)` 的非分层图像有放回抽样。连续的 draw 按预设 `bootstrap_batch_size=32` 分块，整数流与逐次生成相同；仍跳过单类别 draw，仍使用原 2.5/97.5 percentile。没有改为分层或 multinomial resampling，没有减少 1000 次统计预算。

对固定 score 仅排序一次。每个 draw 统计图像重复次数，按相同 score 分组，以 weighted rank-sum 计算 AUROC；同分 positive/negative 配对计 0.5。与独立 sklearn 对每个有放回样本重新排序的结果一致至浮点尾数。内存只随批次×图像数增长，不保存所有 bootstrap draw 的方阵。

该 CI 与分析脚本的按类别分层、同图像配对差值 bootstrap 用途和抽样定义不同；两者继续各自保留。该工程加速不是论文方法贡献，也不代表已验证服务器整体提速。

## 落盘与进度

前向完成后，原完整 `image_predictions.csv` 先原子保存。再计算 image/pixel 点估计并保存 `evaluation_metrics.partial.csv`，然后逐方法补充原 CI。每次 CI 开始前输出 `metrics` 阶段、当前 method、已完成数与 ETA；每个完成后更新 partial CSV。全部统计结束后才标记评价完成，runner 继续写含稀疏/效率/来源信息的最终 `metrics.csv`。

partial 文件仅是统计中间产物，不是完整实验 checkpoint；不能用它跳过拟合/评价，不能改名当成 `metrics.csv`。现有严格完整 dataset resume 规则保持。若在统计阶段中断，已经生成的逐图预测和点估计可用于查因，CI 或最终来源字段缺失仍显式不完整。

版本保护增加 `run_metadata.json` 检查：即使历史目录没有任何 CSV，其已记录版本仍禁止新版覆盖。损坏或无法确定版本的 metadata 也要求使用新目录。v12 原回传不改写、不自动接入 v13。

## 部分回传分析

```bash
python scripts/analyze_results.py \
  --results results/gpu-eval-v12-cosine-simplex \
  --out reports/v12-fit-audit --fit-only
```

`--fit-only` 独立核查已回传的数据集：保存的配置/方法清单、四分支正常样本/扰动/ASLS 共用、全部已回传 U-VaRFS 支持/simplex/候选/lambda 选择与证书作用域、若存在的共享正常 memory，以及可读取的记录 Git source。输出 scope 固定 `normal_fit_only`、`anomaly_performance_assessed=false`；不伪造缺失的 AUROC/AUPRC/AUPRO，也不将部分数据集报告当成全六结果。原完整结果分析入口保持。

诊断不证明所有拟合 baseline 的实际 transform 维数，也不能重新核算本机没有的原始 BMAD 文件 data fingerprint；完整检查仍需最终 CSV、预测和完整数据协议。源文件 SHA256 在分析前后复核，报告写在原结果目录之外，历史跟踪报告受覆盖保护。

## 第 19 节检查

- 研究问题、ASLS、U-VaRFS 目标/求解器/选择规则：不变。
- 无异常标签与 lesion masks 拟合：保持；本轮只读取拟合诊断与最终评价产物。
- Frozen backbone、normal memory cosine 1-NN、Top-1%、pixel 定义：不变。
- 正常 fit/variability/memory/feature/layer 预算、beta/lambda：不变。
- raw Main、L2 消融、原 Gram 完整控制、152 方法与五 seeds：保持。
- CI 抽样/次数/seed/percentile：不变；独立 oracle 检查数值等价。
- 版本与结果：bump v13，独立目录，旧完整或部分回传不续接/覆盖。

## 服务器验证

129 项本地回归检查通过，4 项 CUDA 检查跳过；新增检查包括独立 sklearn CI oracle、批次不变性、同分处理、统计中断保留预测/点估计、阶段进度、部分历史版本保护及 fit-only 审计/预算/源文件保护。训练源码逐文件对照未改变；shell/diff 与 Python 解析检查通过。CPU 统计等价不代表已获得真实检测收益。

```bash
git pull --ff-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

先 Liver 验证 `test` → 预测落盘 → `metrics current=...` → 最终 dataset checkpoint；再固定设计运行全六。不得根据 Liver test 指标调参。实验 CSV/JSON、图表、日志与报告继续仅留本地，代码/配置/测试和本开发文档按用户持续授权提交推送。
