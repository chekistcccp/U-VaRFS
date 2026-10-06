# v7 正常几何与预算诊断

版本：`gpu-eval-v7-geometry-audit`。默认输出 `results/gpu-eval-v7-geometry-audit/`，与旧实验目录隔离。实验分析和图表只在本地 `reports/` 生成，本文只记录开发与复跑协议。

## 主方法与修改检查

Main 继续使用 Frozen DINOv3 全层候选、已获批准的 normal patch geometry ASLS、原 U-VaRFS、normal memory exact cosine 1-NN。ASLS 的 sigmoid、Adam、loss、2–6 层预算、0.05 容差和 gate-prefix 离散规则保持。U-VaRFS 的 beta、lambda grid、目标函数、FISTA、32–256 维预算和选解规则保持。

第 19 节检查：研究问题、主 ASLS 定义、U-VaRFS 目标、无异常标签约束、detector、统一 fit/variability/memory 数据预算均不改变。增加两项明确标记的消融与诊断；旧主方法表示可作同协议比较，但整套方法耗时因增加消融不能直接归因于单方法提速。版本升为 v7，旧目录不可覆盖或 resume。

既有批准只覆盖 patch 主输入。组合搜索当前是 **ablation**，没有升级为主离散规则；若要将它设为 Main，应明确批准新的离散规则，并保留 `asls_gate_prefix_raw` / `asls_gate_prefix_uvarfs` 对照。不得根据 test AUROC 选择输入、规则或超参数，也不声称已实现 Hard-Concrete。

## ASLS 组合搜索消融

主配置显式为 `discrete_selection: gate_prefix`。新增 `asls_geometry_search_raw` 与 `asls_geometry_search_uvarfs`，原 30 方法与五个随机 seeds 均保留，共 32 方法。

保存正常 layer-Gram 内积矩阵 `Q[l,m]=<K_l,K_m>`；仅需 L×L 元素，不保存 N×N patch Gram。组合搜索复用同一正常采样、Q 和已学 gate probability，没有额外 DINO forward 或额外正常训练数据。

1. 对 k=2,…,6 枚举全部 k 层组合，使用原等权 Gram 平均与原 full-hierarchy consensus 计算几何误差。
2. 在第一个有可行组合的 k 内，选择 gate probability 总和最大者；误差与层编号提供确定性 tie-break。
3. 若全部组合不可行，取预算内几何误差最小者，明确 `feasible=false`，不放宽容差或增加层数。
4. 输出层按已有 gate 排序拼接，记录搜索路径和实际检查组合数。12 层且无可行解时共检查 2497 个组合。

这用正常几何区分“gate 前缀没有找到可行组合”和“枚举范围内不存在可行组合”。它不保证异常检测性能，不能替换 test-free 拟合协议。Main 的原设备计算与最大前缀 fallback 保持，避免诊断静默改变主结果。

## U-VaRFS 证书的作用范围

原 `objective`、`box_optimality_gap`、`objective_converged` 都描述截断前的完整连续权重，新增 `objective_certificate_scope=continuous_full_weights` 明确这一点。

新增 `budgeted_objective`、预算后各目标分项、`budgeted_box_optimality_gap`、`budgeted_objective_converged`，直接在 detector 实际使用的零填充权重上计算同一个原目标。box gap 相对原完整 box 问题，不能将固定稀疏支持集下的解说成全局连续最优。

对每个 lambda 的固定保留支持集 S，额外计算 `fixed_support_geometry_floor`：将 S 内所有权重设为 1，其他设为 0，评价原几何误差。由于 H 元素非负，这给出固定 S 在 0≤w≤1 下的最小几何误差；若它大于容差，当前支持集无法仅靠调整权重达到容差。它不是所有 max_features 组合的下界，也不包含 variability/L1 的最优性结论。

上述数值全部为诊断，不重新拟合权重、不参与 lambda 选解，不改变 scales、active dimensions 或 detector。`lambda=0` 仍只作为诊断，`selection_candidate=false`。

## 本地审计与测试

`scripts/analyze_results.py` 支持完整单版本回传；v4 历史模板保留，新版本报告由实际数据生成。独立重算每个方法的逐图 AUROC/AP、全部 seed 的 mean/sample SD，核对 config/code Git 来源、fit/memory normal manifest、统一预算、mask 数量和汇总一致性。

paired bootstrap 比较同一图像的 Main 与固定方法；随机对照在每次 draw 内平均各 seed 的 AUC，不构造预测 ensemble。区间只用于评价，缺少患者/slide 分组时仅报告 image-level 区间。

审计核对原文件前后 SHA256，拒绝往原结果目录、其父目录或已跟踪的历史报告目录写入。所选 lambda 的数值诊断取对应 path point，修复了原脚本总取最后一个 lambda 的相邻变化值的问题。源码工具可以同步，输出报告与图表不能默认同步。

50 项本地检查通过，包括独立直接 Gram 对照、前缀遗漏可行组合、全部预算组合的不可行 fallback、连续/预算后证书区分、固定支持集下界、正常训练约束、32 方法 CPU fit→memory→pixel evaluation，以及旧版本覆盖保护。`bash -n run.sh` 与 diff 格式检查通过。

本地没有真实 BMAD 数据、权重或 CUDA PyTorch；这些检查不能证明服务器上的新版精度或耗时。

## 服务器验证

先验证 Liver（CT 调试/主分析数据集，最终仍跑 BMAD 全六）：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
```

检查 Main 为 patch/gate_prefix、两个 geometry_search 消融、normal-only manifest、Q、预算后证书和完整 image/pixel 指标。确认输出进入 v7 新目录后运行全部六数据集：

```bash
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

用户自行维护环境；run.sh 不安装依赖。回传后先读版本、配置与来源再分析，结果保留本地。
