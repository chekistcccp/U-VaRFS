# v14：保持原目标的固定支持数值精修

项目继续为 **Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究**。Main 仍用第 5.4/37 节已批准的 cosine/simplex/cardinality 目标及 raw 输入；L2 为完整消融。实验版本为 `gpu-eval-v14-fixed-support-polish`，独立目录为 `results/gpu-eval-v14-fixed-support-polish/`。

本轮回传结果及科学结论仅存本地 `reports/`；本文件仅记录工程实现与运行协议。数值目标下降不等于异常检测指标提高，也不等于全局最优。

## 修改与保护

保留原 sample-space 精确损失、解析梯度、normal-reference Gram、variability 归一化、simplex floor 和 projected BB/Armijo 求解。固定支持投影求解后，如果真实一阶残差仍超过原 `tol=1e-5`，使用 SLSQP 精修同一固定支持：

```text
min  representation_term(p) + beta * relative_variability(p)
s.t. sum(p)=1; p_j >= floor_mass/K
```

支持不变时 `lambda*K` 为常数，仍在最终候选池比较完整目标。没有增加近邻损失、异常任务、列标准化、L2 Main 或新的统计采样。原 projected refit 最多 2000 次；精修最多 200 次，`ftol=1e-12` 只控制数值停止，写入 config 与 solver manifest，不能解释为科学效果阈值。将 `cosine_polish_max_iter` 明确设为 0 可关闭精修，用于数值诊断。

SLSQP 只在 CPU 上控制至多 256 维权重向量；同一 Torch 分块损失/梯度继续在输入的原 device/dtype 上计算。使用已有 SciPy 依赖，不自动安装环境。内部试探点可能不满足等式或逐步下降；只有优化后的端点经同一 simplex/floor 投影、实际损失重新计算且 **严格小于输入端点损失** 才被接受。端点无效或没有下降则保留已有投影解。优化器即使达到迭代上限，端点仍须通过该实际下降检查；即使返回 success，也不能跳过检查。

精修之后重新计算 `||p-project_simplex(p-gradient)||`。`fixed_support_converged` 只取决于此真实残差，独立保存 optimizer success/status/message。无法达到阈值时仍报告 nonstationary，不降低原容差，不以几何 feasible 代替收敛，不借用原 quadratic 的凸/全局证书。SLSQP 的浮点计算与非凸性仍可能造成停滞。

每个初始化候选和接受的支持交换均执行相同精修。完整记录 initial/refit 与 exchange/refit 的精修前后目标、真实残差、接受与否、迭代/目标评价次数/耗时。所有精修时间计入原 fit time；增加 CPU 控制和 CPU/GPU 向量同步，服务器实际成本需验证，不宣称整体训练更快。日志包括精修 start/done 与长过程的迭代进度。

## 选择、控制与结果审计

候选支持来源、K 列表、normal energy/原 Gram 初始化、每次交换的 proposal 数、交换预算、beta、lambda grid、geometry tolerance 均保持。有限生成池选择完整新目标最小的 lambda 点，再选择几何容差内最稀疏点，或显式 infeasible fallback；不使用 test 标签调参或选择权重。

原 quadratic/FISTA/v11 forward solver 代码与完整 `gram_*` 控制保持。Raw/L2 × cosine/Gram 的 152 方法、五 seeds、按各支实际 Main K 匹配的 PCA/Random、共享正常图像/patch/扰动/backbone/ASLS/memory 抽样保持。冻结 backbone、全部层候选、ASLS 定义、BMAD 六数据集、正常 fit/variability/memory 和 latent dimension 预算、exact cosine 1-NN/Top-1%/bilinear pixel protocol 均保持。

分析器验证精修 manifest、严格端点下降、最终残差作用域、每个候选与交换的成本总计及最终有限池选择；仍能审计旧 v12 无精修字段的完整回传。带精修配置而缺少 manifest 的结果拒绝。更新数值求解会改变拟合权重和候选轨迹，不能覆盖、续接或混合旧 v12/v13 结果，必须使用新版本与严格代码/配置指纹。

## 第 19 节检查

| 检查项 | 本轮结论 |
| --- | --- |
| 研究问题、ASLS、U-VaRFS 精确目标 | 保持，精修是同一固定支持目标的数值求解升级 |
| anomaly labels/masks 训练 | 无，回传标签仅用于最终性能分析 |
| 主输入、detector、六数据集 | raw Main、原 detector、全部六数据集保持 |
| main/baseline 正常数据和维度预算 | 保持；所有 cosine U-VaRFS 同样执行精修，数值成本明确记录 |
| 原 quadratic/消融/五 seeds | 保持完整，未根据 test 指标删除基线 |
| 新旧结果与版本 | 数值权重可能改变，bump v14 并独立目录，不作位级等价声明 |

## 验证与服务器运行

独立正常合成特征探针核验：投影求解停滞后仍可下降同一目标；sample-space oracle 核验实际端点损失；无效/更差端点不能通过 success 覆盖已有解；iteration limit 不伪造收敛；真实一阶残差和候选/交换全部成本可回读验证。完整 CPU 回归覆盖两个输入 primary 的 152 方法、同数据独立原求解器控制、memory、image/pixel 评价、resume 与报告审计。CUDA 与真实 BMAD 数据/模型不在本工作机，不能据 fixture 宣称新的 AUROC/AUPRO 提升。

先在服务器 Liver 验证精修日志、实际残差/耗时、四支拟合文件和 152 方法完成；随后以同一配置运行全部六数据集。Liver 是调试与 CT 重点分析，最终研究仍为 BMAD 全六；不用 Liver test 指标改变 beta/lambda/K/input。

```bash
git pull --ff-only
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 DATASETS=liver bash run.sh
SKIP_PREPROCESS=1 SKIP_MODEL_DOWNLOAD=1 bash run.sh
```

旧结果保持原样；完整分析仍用 `scripts/analyze_results.py --results <独立版本目录> --out reports/<新分析目录>`，正常拟合单独审计可用 `--fit-only`。结果/报告/日志继续只存本地，代码、配置、测试与本开发交接按持续授权提交和推送。

本轮完整回归：140 passed，5 skipped（本机无 CUDA）；两个输入 primary 的 152 方法、独立原 quadratic 控制、数学/端点/残差/成本保护、评价与 resume 通过。旧 v12 完整回传的 24 个主分支审计重新通过，748 个原文件 SHA256 保持；loss/投影/coverage AST、ASLS/原 solver/fit/eval/backbone/detector 代码及除新增数值设置/版本目录外全部配置对照保持。shell/diff 检查通过；真实服务器的 v14 异常检测性能尚未验证。
