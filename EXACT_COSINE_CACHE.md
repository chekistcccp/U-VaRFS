# v19：精确余弦状态缓存与部分回传审计

历史版本 `gpu-eval-v19-exact-cosine-state-cache`，独立目录 `results/gpu-eval-v19-exact-cosine-state-cache/`。这是工程修正，该历史Main使用第5.5/43节的固定预算capped cosine/simplex。当前v20将此实现保留在旧方法控制，新Main见[NORMAL_LOCAL_SELECTION.md](NORMAL_LOCAL_SELECTION.md)。性能优先，不用更强压缩、低训练目标或正常近邻诊断代替异常检测收益。回传成绩、分析和运行日志只留本地。

## 保持当前研究与算法

BMAD 全六、Frozen DINOv3 全层候选、patch ASLS、raw Main 与 L2 消融、normal-only nuisance variability、beta、K=min(256,M)、floor/cap、正常几何容差和有限池选择保持。保留完整 v17 候选和 `asls_fixed_budget_uvarfs`，v18 的交换提案仍全部先重拟合再比较，严格下降及父状态正常几何可行性保护保持。原压缩 cosine 和 original Gram 分支、五随机 seeds、同维 PCA/Random、同支持去权重、正常 fit/variability/memory 预算、cosine 1-NN、Top-1% 和 pixel 定义保持。默认 164 方法，完整六数据集应有 984 行。

## 复用完全相同的余弦状态

投影线搜索先计算 trial 的损失，再为被接受的同一 trial 求梯度；原实现再次构建同一个 N×N cosine matrix。`CachedCosineObjective` 保留最后一个固定支持/权重状态的 actual cosine blocks、行归一化量和已计算梯度，解析损失、梯度及行块求和顺序沿用原实现。

只有有序 active 和 p 的 dtype/device/全部值完全相同才命中。一个可表示的权重变化或支持顺序变化都会失效。没有舍入 key、近似权重匹配、少量 patches 估计或跨输入复用。返回值与输入 key 隔离，调用方修改不会污染缓存。正常 x/P 在该 engine 生命周期中只读。

`performance_objective_cache_mib: 128` 限制持久缓存张量的字节数；预留两种梯度空间，超预算则走原实现，0 明确关闭。默认 N=4096/K=256 时 actual blocks 为 64 MiB，还需支持和归一化张量；这是 solver 临时工作空间，不是 normal memory bank 扩容。所有数据/方法的同类求解共用这一规则。

原 `CosineObjective`、`refit_simplex`、v17 reference solver、原 cosine/Gram 控制不改。全部拒绝 refit/SLSQP 的迭代与成本仍纳入原统计。`objective_evaluations` 继续指有效 evaluate 请求；新增 `objective_cache` 区分请求数、实际 geometry 构建、复用、完整返回复用以及缓存峰值，审计核验计数恒等式和字节上限。

CPU 数学一致性和减少矩阵构建不证明服务器实际加速。CUDA 等值与端到端耗时须实测，保留原 peak GPU memory 记录。

## 部分回传只分析完整数据集

```bash
python scripts/analyze_results.py \
  --results results/gpu-eval-v18-refitted-support-exchange \
  --out reports/partial-return-analysis \
  --completed-only --bootstrap 2000
```

当前 cosine/simplex 主目标支持该选项。读取最终 `all_metrics.csv` 或 `all_metrics.partial.csv`；汇总只能包含同时有 `metrics.csv` 和 `eval_progress.stage=complete` 的数据集，不接受方法缺失、最终指标/进度不一致、重复记录、混版本或混来源。未完成目录的 config/code/Git/version 也核验，不能静默忽略冲突来源。

完整数据集继续核验全部逐图 AUROC/AP、mask coverage、五 seed mean/sample SD、训练/memory 正常清单、四个表示分支、有限池和求解保护。报告列出全部六数据集状态、已完成范围及缺失范围；单数据集不写成六数据集宏平均。未全部完成时输出 `completed_metrics.csv`，原始产物只读并记录前后 SHA256。默认无该选项仍要求全六；`--fit-only` 保持只讨论正常拟合，不编造检测指标。

## 第 19 节检查与验证边界

1. 研究问题：不变。
2. ASLS 定义：不变。
3. U-VaRFS 目标与候选/选择规则：不变。
4. anomaly labels/masks：仅用于结果评价，不进入拟合。
5. detector：不变。
6. main/baseline 数据预算：不变；缓存是显式有上限的工程工作空间。
7. 旧/新统计：输出增加工程成本字段；数值等值由独立原实现对照验证，真实性能不另作保证。
8. version：升为 v19，新代码/配置指纹与目录防止旧结果续接混用。

本机缺少真实 BMAD/模型和 CUDA runtime。完成本地回归后，服务器先验证 Liver 的数值、缓存内存/命中统计、完整控制和运行成本，再以同一配置运行全六。不得根据单一数据集的 test 指标切换 Main 输入、改变目标或增减预算。
