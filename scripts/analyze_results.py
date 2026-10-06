#!/usr/bin/env python3
"""Read-only audit/report of a complete returned BMAD run; labels are evaluation only.

No layer, lambda, representation budget or detector setting is selected here.
The bootstrap compares the mean AUC of random seeds, not an ensemble score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ['brain','liver','resc','oct2017','xray','camelyon16']
PIXEL_DATASETS = {'brain','liver','resc'}


def source_hashes(folder):
    return {p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()}


def validate_output(results, output):
    if output.resolve().is_relative_to(results.resolve()) or results.resolve().is_relative_to(output.resolve()):
        raise ValueError('analysis output must be outside the original results directory')
    repo=Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo):
        tracked=subprocess.check_output(['git','ls-files','--',output.resolve().relative_to(repo).as_posix()],
                                        cwd=repo,text=True)
        if tracked.strip():
            raise ValueError('analysis output contains tracked historical artifacts; choose a new local reports directory')


class WeightedRanking:
    """Exact weighted AUROC/AP for fixed scores, including score ties."""
    def __init__(self, labels, scores):
        self.labels = np.asarray(labels,dtype=np.int64)
        scores = np.asarray(scores,dtype=np.float64)
        if len(scores)!=len(self.labels) or not np.isfinite(scores).all():
            raise ValueError('predictions must be finite and aligned with labels')
        self.order = np.argsort(scores,kind='stable')
        sorted_scores = scores[self.order]
        self.starts = np.r_[0,np.flatnonzero(np.diff(sorted_scores))+1]
        self.positive = self.labels[self.order]==1

    def metrics(self, weights):
        weights = np.atleast_2d(np.asarray(weights, dtype=np.float64))[:,self.order]
        positive = np.add.reduceat(weights*self.positive,self.starts,axis=1)
        negative = np.add.reduceat(weights*~self.positive,self.starts,axis=1)
        total_positive = positive.sum(1)
        total_negative = negative.sum(1)
        negatives_below = np.cumsum(negative,axis=1)-negative
        auc = (positive*(negatives_below+.5*negative)).sum(1)/(total_positive*total_negative)
        tp = np.cumsum(positive[:,::-1],axis=1)
        observations = np.cumsum((positive+negative)[:,::-1],axis=1)
        precision = np.divide(tp,observations,out=np.zeros_like(tp),where=observations>0)
        ap = (positive[:,::-1]*precision).sum(1)/total_positive
        return auc,ap


def paired_bootstrap(frame, comparisons, draws=2000, seed=42, batch_size=32):
    y=frame['label'].to_numpy(dtype=np.int64)
    normal=np.flatnonzero(y==0); abnormal=np.flatnonzero(y==1)
    if not len(normal) or not len(abnormal):
        raise ValueError('paired AUROC bootstrap needs both classes')
    names=['main']+sorted({name for group in comparisons.values() for name in group})
    rankings={name:WeightedRanking(y,frame[name]) for name in names}
    rng=np.random.default_rng(seed)
    differences={family:[] for family in comparisons}
    for start in range(0,draws,batch_size):
        size=min(batch_size,draws-start)
        weights=np.empty((size,len(y)),dtype=np.float64)
        weights[:,normal]=rng.multinomial(len(normal),np.full(len(normal),1/len(normal)),size=size)
        weights[:,abnormal]=rng.multinomial(len(abnormal),np.full(len(abnormal),1/len(abnormal)),size=size)
        aucs={name:ranking.metrics(weights)[0] for name,ranking in rankings.items()}
        for family,methods in comparisons.items():
            baseline=np.mean([aucs[name] for name in methods],axis=0)
            differences[family].extend((aucs['main']-baseline).tolist())
    return {family:np.percentile(values,[2.5,97.5]).tolist() for family,values in differences.items()}


def markdown_table(frame, columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(str(row[column]) for column in columns)+' |')
    return '\n'.join(lines)


def audit(results: Path, output: Path, draws: int):
    validate_output(results,output)
    hashes=source_hashes(results)
    df=pd.read_csv(results/'all_metrics.csv')
    if sorted(df.dataset.unique())!=sorted(DATASETS) or df.duplicated(['dataset','method']).any():
        raise ValueError('run must contain exactly six datasets without duplicate methods')
    versions=df.experiment_version.unique()
    if len(versions)!=1:
        raise ValueError('mixed experiment versions are not comparable')
    output.mkdir(parents=True,exist_ok=True)
    summary_rows=[]; diagnostics=[]; paired_rows=[]; prevalence_rows=[]; lambda_rows=[]; zero_rows=[]
    config_hashes=set(); code_hashes=set(); revisions=set()
    for dataset in DATASETS:
        folder=results/dataset
        source=pd.read_csv(folder/'metrics.csv')
        expected=df[df.dataset==dataset].sort_values('method').reset_index(drop=True)
        if not (set(expected.columns)-set(source.columns)) <= {'pixel_auroc','pixel_auprc','aupro'}:
            raise ValueError(f'{dataset}: unexpected missing per-dataset metric columns')
        source=source.reindex(columns=expected.columns)
        pd.testing.assert_frame_equal(source.sort_values('method').reset_index(drop=True),expected,
                                      check_dtype=False,check_like=True)
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        config_hash=hashlib.sha256(json.dumps(meta['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if config_hash!=meta['config_fingerprint']:
            raise ValueError(f'{dataset}: config fingerprint does not match the saved config')
        if sorted(source.method)!=sorted(meta['expected_methods']):
            raise ValueError(f'{dataset}: incomplete configured method set')
        progress=json.loads((folder/'eval_progress.json').read_text())
        if progress['stage']!='complete' or progress['methods']!=len(source):
            raise ValueError(f'{dataset}: incomplete evaluation')
        config_hashes.add(meta['config_fingerprint']); code_hashes.add(meta['code_fingerprint']); revisions.add(meta['git_head'])
        cov=json.loads((folder/'mask_coverage.json').read_text())
        if dataset in PIXEL_DATASETS:
            if cov['missing_abnormal_masks'] or source[['pixel_auroc','pixel_auprc','aupro']].isna().any().any():
                raise ValueError(f'{dataset}: missing pixel evaluation')
        predictions=pd.read_csv(folder/'image_predictions.csv')
        if len(predictions)!=progress['samples'] or predictions.image_path.duplicated().any():
            raise ValueError(f'{dataset}: prediction coverage mismatch')
        y=predictions.label.to_numpy(dtype=np.int64)
        if set(np.unique(y))!={0,1} or cov['samples']!=len(y) or cov['abnormal']!=int((y==1).sum()):
            raise ValueError(f'{dataset}: labels and mask coverage counts disagree')
        if dataset in PIXEL_DATASETS and cov['matched_abnormal_masks']!=int((y==1).sum()):
            raise ValueError(f'{dataset}: incomplete abnormal mask coverage')
        for _,row in source.iterrows():
            auroc,ap=WeightedRanking(y,predictions[row['method']]).metrics(np.ones(len(y)))
            np.testing.assert_allclose([auroc[0],ap[0]],[row.image_auroc,row.image_auprc],rtol=1e-10,atol=1e-10)
        fit=json.loads((folder/'fit_manifest.json').read_text())
        memory=json.loads((folder/'memory_manifest.json').read_text())
        if not memory['shared_sampling_across_methods'] or len(set(memory['memory_rows'].values()))!=1:
            raise ValueError(f'{dataset}: unequal memory protocol')
        if any('/good/' not in path.replace('\\','/').lower() for path in fit['normal_train_paths']):
            raise ValueError(f'{dataset}: fit manifest contains a non-normal path')
        cfg=meta['config']; seeds=cfg['random_baseline_seeds']
        for manifest in [fit,memory]:
            paths=manifest['normal_train_paths']; ids=manifest['normal_train_indices']
            if len(paths)!=len(ids) or len(set(ids))!=len(ids) or any(
                '/train/good/' not in '/'+path.replace('\\','/').lower() for path in paths):
                raise ValueError(f'{dataset}: normal training manifest is inconsistent')
        if (fit['fit_images']!=len(fit['normal_train_paths']) or fit['fit_images']>cfg['data']['fit_images'] or
            fit['variability_images']!=min(cfg['data']['variability_images'],fit['fit_images']) or
            fit['patches_per_image']!=cfg['data']['patches_per_image'] or
            len(memory['normal_train_paths'])>cfg['data']['memory_images'] or
            memory['memory_capacity']!=cfg['data']['memory_size'] or
            set(memory['memory_rows'])!=set(source.method) or
            not (source.memory_size==next(iter(memory['memory_rows'].values()))).all()):
            raise ValueError(f'{dataset}: recorded normal data or memory budget differs from config')
        comparisons={'fixed4_raw':['fixed4_raw'],'asls_raw':['asls_raw'],
                     'randomk_raw':[f'randomk_raw_seed{s}' for s in seeds],
                     'randomk_uvarfs':[f'randomk_uvarfs_seed{s}' for s in seeds],
                     'asls_random':[f'asls_random_seed{s}' for s in seeds]}
        for method in ['asls_pooled_uvarfs','asls_pca','asls_geometry_search_uvarfs','asls_gate_prefix_uvarfs',
                       'asls_top_weights_uvarfs','legacy_main']:
            if method in source.method.values:
                comparisons[method]=[method]
        intervals=paired_bootstrap(predictions,comparisons,draws=draws)
        main=source.set_index('method').loc['main']
        for family,methods in comparisons.items():
            delta=float(main.image_auroc-source.set_index('method').loc[methods].image_auroc.mean())
            paired_rows.append({'dataset':dataset,'comparison':family,'delta_auroc':delta,
                                'ci95_lower':intervals[family][0],'ci95_upper':intervals[family][1],
                                'bootstrap_draws':draws,'bootstrap_seed':42,'bootstrap_unit':'image',
                                'stratified_by_label':True})
        prevalence_rows.append({'dataset':dataset,'normal':int((y==0).sum()),'abnormal':int((y==1).sum()),
                                'image_positive_prevalence':float(y.mean()),'main_image_auprc':float(main.image_auprc)})
        a=json.loads((folder/'asls.json').read_text())
        uv=json.loads((folder/'uvarfs_main.json').read_text())
        selected=next(point for point in uv['lambda_path'] if point['lambda']==uv['lambda'])
        lambda_rows.extend({'dataset':dataset,**point,'selected':point['lambda']==uv['lambda']}
                           for point in uv['lambda_path'])
        if 'zero_lambda_diagnostic' in uv:
            zero_rows.append({'dataset':dataset,**uv['zero_lambda_diagnostic']})
        diagnostics.append({'dataset':dataset,'layers':a['selected_layers'],'layer_count':len(a['selected_layers']),
                            'feature_dim':int(main.feature_dim),'asls_geometry_error':a['geometry_error'],
                            **a['diagnostics'], 'uvarfs_geometry_error':uv['geometry_error'],
                            'uvarfs_feasible':uv['feasible'],'uvarfs_converged':uv['solver_converged'],
                            'asls_feasible':a['feasible'],
                            'geometry_representation':a['geometry_representation'],
                            'projected_residual':uv['projected_residual'],'relative_change':selected['relative_change'],
                            'objective_converged':uv.get('objective_converged'),
                            'box_optimality_gap':uv.get('box_optimality_gap'),
                            'discrete_selection':a.get('discrete_selection','gate_prefix'),
                            'sparsity_strategy':uv.get('sparsity_strategy','top_weights'),
                            'budgeted_objective':uv.get('budgeted_objective'),
                            'budgeted_box_optimality_gap':uv.get('budgeted_box_optimality_gap'),
                            'fixed_support_box_gap':uv.get('fixed_support_box_gap'),
                            'lambda':uv['lambda'],'iterations':uv['solver_iterations'],
                            'mask_coverage':{k:len(v) if isinstance(v,list) else v for k,v in cov.items()},
                            'config':cfg})
        source['method_family']=source.method.str.replace(r'_seed\d+$','',regex=True)
        metrics=['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro','feature_dim',
                 'selected_layer_count','compression_ratio','matching_and_aggregation_seconds']
        for family,group in source.groupby('method_family'):
            result={'dataset':dataset,'method_family':family,'runs':len(group)}
            for name in metrics:
                result[name+'_mean']=float(group[name].mean())
                result[name+'_std']=float(group[name].std(ddof=1)) if len(group)>1 else 0.0
            summary_rows.append(result)
        print(f'[audit] {dataset}: {len(predictions)} predictions verified; paired bootstrap complete',flush=True)
    if len(config_hashes)!=1 or len(code_hashes)!=1 or len(revisions)!=1:
        raise ValueError('cross-dataset run provenance differs')
    summary=pd.DataFrame(summary_rows)
    returned_summary=pd.read_csv(results/'summary_metrics.csv')
    for _,row in returned_summary.iterrows():
        reference=summary[(summary.dataset==row.dataset)&(summary.method_family==row.method_family)].iloc[0]
        for column in ['image_auroc_mean','image_auroc_std','image_auprc_mean','image_auprc_std','feature_dim_mean']:
            np.testing.assert_allclose(row[column],reference[column],rtol=1e-10,atol=1e-10)
    summary.to_csv(output/'family_summary.csv',index=False)
    pd.DataFrame(paired_rows).to_csv(output/'paired_auc_differences.csv',index=False)
    pd.DataFrame(prevalence_rows).to_csv(output/'image_prevalence.csv',index=False)
    pd.DataFrame(lambda_rows).to_csv(output/'main_lambda_path.csv',index=False)
    pd.DataFrame(zero_rows).to_csv(output/'zero_lambda_diagnostics.csv',index=False)
    shutil.copyfile(results/'all_metrics.csv',output/'all_metrics.csv')
    shutil.copyfile(results/'layer_selection.csv',output/'layer_selection.csv')
    if source_hashes(results)!=hashes:
        raise ValueError('original experiment artifacts changed during the audit')
    data={'experiment_version':versions[0],'git_revision':next(iter(revisions)),
          'config_fingerprint':next(iter(config_hashes)),'code_fingerprint':next(iter(code_hashes)),
          'rows':len(df),'diagnostics':diagnostics,'source_sha256':hashes,
          'bootstrap_draws':draws,'bootstrap_unit':'image','bootstrap_seed':42,
          'bootstrap_batch_size':32,'macro':'equal dataset weight, random seeds averaged within dataset'}
    # Verify the recorded source against its Git blobs, independent of host CRLF.
    repo=Path(__file__).resolve().parents[1]
    try:
        revision=data['git_revision']
        tree=subprocess.check_output(['git','ls-tree','-r','--name-only',revision],cwd=repo,text=True).splitlines()
        paths=sorted(p for p in tree if Path(p).parent.as_posix()=='uvarfs' and p.endswith('.py'))+['scripts/run_all.py']
        digest=hashlib.sha256()
        for path in paths:
            digest.update(path.encode()); digest.update(subprocess.check_output(['git','show',f'{revision}:{path}'],cwd=repo))
        data['git_blob_code_verified']=digest.hexdigest()==data['code_fingerprint']
        if not data['git_blob_code_verified']:
            raise ValueError('saved code fingerprint differs from its recorded Git source')
    except (OSError,subprocess.CalledProcessError):
        data['git_blob_code_verified']=False
    (output/'analysis_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    report=write_report if versions[0]=='gpu-eval-v4-protocol-fixes' else write_current_report
    report(output,df,summary,pd.DataFrame(paired_rows),pd.DataFrame(prevalence_rows),data)


def write_current_report(output,df,summary,paired,prevalence,data):
    main=df[df.method=='main'].set_index('dataset').loc[DATASETS]
    macro=summary.groupby('method_family').image_auroc_mean.mean().sort_values(ascending=False)
    table=main[['selected_layers','feature_dim','image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].reset_index()
    for column in ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']:
        table[column]=table[column].map(lambda x:f'{100*x:.2f}' if pd.notna(x) else 'n.a.')
    comparison=pd.DataFrame({'dataset':DATASETS})
    families=['main','legacy_main','asls_top_weights_uvarfs','asls_gate_prefix_uvarfs',
              'asls_pooled_uvarfs','asls_raw','fixed4_raw','all_raw','randomk_raw','asls_random','asls_pca']
    wins={}
    for family in families:
        if family not in macro:
            continue
        rows=summary[summary.method_family==family].set_index('dataset').loc[DATASETS]
        comparison[family]=[f'{100*r.image_auroc_mean:.2f} ± {100*r.image_auroc_std:.2f}' if r.runs>1 else
                            f'{100*r.image_auroc_mean:.2f}' for _,r in rows.iterrows()]
        wins[family]=int((main.image_auroc>rows.image_auroc_mean).sum())
    diagnostic=pd.DataFrame([{'dataset':d['dataset'],'layers':d['layer_count'],
        'ASLS_error':d['asls_geometry_error'],'ASLS_feasible':d['asls_feasible'],
        'UVarFS_error':d['uvarfs_geometry_error'],'UVarFS_feasible':d['uvarfs_feasible'],
        'continuous_objective_converged':d['objective_converged'],'continuous_box_gap':d['box_optimality_gap'],
        'budgeted_box_gap':d.get('budgeted_box_optimality_gap'),
        'fixed_support_box_gap':d.get('fixed_support_box_gap'),
        'selected_lambda':d['lambda']} for d in data['diagnostics']])
    gate_diagnostic=pd.DataFrame([{'dataset':d['dataset'],
        'consensus_gram_off_diagonal_mean':d['consensus_off_diagonal_mean'],
        'gate_probability_span':d['probability_span'],
        'gate_variability_rank_correlation':d['gate_variability_rank_correlation'],
        'gates_near_all_open':d['gates_near_all_open']} for d in data['diagnostics']])
    lambda_path=pd.read_csv(output/'main_lambda_path.csv')
    budget_rows=[]
    for dataset,group in lambda_path.groupby('dataset',sort=False):
        point=group.loc[group['lambda'].idxmin()]
        budget_rows.append({'dataset':dataset,'min_lambda':point['lambda'],
            'continuous_active_features':point.continuous_active_features,
            'continuous_geometry_error':point.continuous_geometry_error,
            'retained_features':point.retained_features,'budgeted_geometry_error':point.geometry_error})
    pairs=paired[['dataset','comparison','delta_auroc','ci95_lower','ci95_upper']].copy()
    for column in ['delta_auroc','ci95_lower','ci95_upper']:
        pairs[column]=pairs[column].map(lambda x:f'{100*x:+.2f}')
    efficiency=main[['selected_layer_count','feature_dim','candidate_dim','compression_ratio','memory_size',
        'memory_vector_fp16_mib','matching_and_aggregation_seconds','fit_seconds','memory_seconds','eval_seconds','peak_gpu_gb']].reset_index()
    efficiency['compression_ratio']*=100
    efficiency=efficiency.round(3)
    lines=['# BMAD 回传实验分析','',
        f'版本：`{data["experiment_version"]}`；六数据集共 {data["rows"]} 行。Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究主线保持。','',
        f'Main Macro Image AUROC={100*macro["main"]:.2f}%，Fixed-4 Raw={100*macro["fixed4_raw"]:.2f}%，同 K Random Raw 五 seeds 均值={100*macro["randomk_raw"]:.2f}%。Main 在 {wins["fixed4_raw"]}/6 数据集高于 Fixed-4，在 {wins["randomk_raw"]}/6 高于 Random-K 均值。','']
    if 'asls_pooled_uvarfs' in macro:
        lines.extend([f'同轮 Main 比 pooled U-VaRFS 的 macro AUROC 高 {100*(macro["main"]-macro["asls_pooled_uvarfs"]):.2f} 个百分点。它比较 normal geometry 输入及其产生的选层；跨版本差异还含数值实现变化，不全部归因于 patch。',''])
    lines.extend(['## 来源与完整性','',
        f'Git `{data["git_revision"]}`；code `{data["code_fingerprint"]}`；config `{data["config_fingerprint"]}`。六份元数据一致，源码 Git blobs 验证结果：{data["git_blob_code_verified"]}。',
        '全局/逐数据集 CSV 一致；全部逐图 AUROC/AP 独立重算一致，family mean/sample SD 一致；正常 fit/memory manifest 与共享抽样预算通过核查。来源 SHA256 在 analysis_data.json，审计前后未改变原文件。',
        'Brain/Liver/RESC 异常 masks 完整；其余数据集无 pixel 指标。所有标签仅用于评价，没有用 test AUROC 选层、lambda、维度或 detector。', '',
        '## 主结果','', '指标均为百分数；不以高 Pixel AUROC 替代 Pixel AP/AUPRO。','',
        markdown_table(table,table.columns.tolist()),'', '![Performance](performance.png)','',
        '## 同轮公平比较','', 'Random 项均纳入全部五 seeds，报告 mean ± sample SD；Random-K 的 K 与 Main 相同。','',
        markdown_table(comparison,comparison.columns.tolist()),'',
        f'在相同 ASLS 层输入下，U-VaRFS 在 {wins["asls_raw"]}/6 数据集高于 Raw，在 {wins["asls_random"]}/6 高于同维度 Random feature 均值，在 {wins["asls_pca"]}/6 高于 PCA。不能据此宣称全部数据集优于固定层或 PCA。','',
        f'配对差值为 Main 减 baseline，单位百分点。{data["bootstrap_draws"]} 次按正常/异常分层的 image bootstrap；每次 draw 内取各 seed AUC 均值，不将 seed 预测均值当作 ensemble。','',
        markdown_table(pairs,pairs.columns.tolist()),'',
        '未提供患者/slide 分组，区间仅以图像为抽样单位；没有多重比较校正或预设非劣界值，不宣称患者级显著性或非劣。','',
        '## 正常训练几何与数值诊断','',markdown_table(diagnostic.round(8),diagnostic.columns.tolist()),'',
        'ASLS feasible 与 U-VaRFS feasible 分别对应实际预算后表示。前缀规则失败不证明全部组合不可行；geometry_search 穷尽整个预算仍失败才说明该采样正常几何下没有可行组合。连续 gates 接近全开，其微小排序差异不等于成功的自适应稀疏。','',
        markdown_table(gate_diagnostic.round(6),gate_diagnostic.columns.tolist()),'',
        'continuous_objective_converged/box gap 对应截断前的完整连续权重，不能用来证明截断后的表示优化充分。完整 lambda path 见 main_lambda_path.csv，lambda=0 只作为诊断，见 zero_lambda_diagnostics.csv，不参与选解。', '',
        '下表是既定 path 最小 lambda 的连续解与实际预算后几何误差，不用于增改 lambda grid。','',
        markdown_table(pd.DataFrame(budget_rows).round(6),list(budget_rows[0])), '',
        '连续解较好、Top-weight 截断后误差增大，说明预算/支持集截断值得单独诊断；连续解也不可行则还可能涉及既定 variability 正则与输入几何。不能仅提高迭代数、放宽容差或用测试结果挑 beta/lambda。', '',
        '## 按原主线实施的修改','',
        f'本轮 ASLS 离散规则为 {data["diagnostics"][0].get("discrete_selection","gate_prefix")}，特征稀疏策略为 {data["diagnostics"][0].get("sparsity_strategy","top_weights")}。规则由运行前配置和用户授权固定，不按 test AUROC 择优。历史 v7 使用 prefix 与组合搜索消融；v8 按用户指令升级组合搜索，并保留 prefix、旧 Top-weight 和 legacy_main 对照。','',
        'v8 特征剪枝使用原目标的精确单维删除损失，随后在固定支持集重优化同一目标。每个 lambda 保留旧截断候选，只有正常训练目标与几何误差均不变差的候选可被接受。这不保证异常 AUROC/AUPRO 提升；固定支持集 gap 不证明全局最佳稀疏支持集。','',
        'U-VaRFS 目标、beta/lambda、维度与数据预算不变，新增预算后原目标值/box gap 与固定支持集几何下界。H 元素非负，因此在固定支持集上将允许权重设为 1 可得其最小几何误差；这不是全体 256 维组合的最优性证明，不作为新的选解候选。', '',
        '## 压缩与效率','',
        'compression_ratio 包含层与特征两次裁剪，以全层 candidate_dim 为分母；仅 U-VaRFS 压缩需以选中层维度为分母。memory_vector_fp16_mib 是该方法向量容量；matching 时间只包含 NN 与 Top-K，不是全部方法总耗时。fit/memory/eval/peak 是共享整套方法开销，不能重复计作单方法耗时或宣称 backbone 加速。','',
        markdown_table(efficiency,efficiency.columns.tolist()),'',
        '## 类别比例','',markdown_table(prevalence.round(6),prevalence.columns.tolist()),'',
        'Image AP 需结合异常比例解释，尤其 X-ray 异常比例很高。','',
        '## 复跑边界','',
        '新版本结果写入独立版本目录，旧结果不覆盖/混合。服务器先验证 Liver 正常拟合、消融、manifest 和 pixel 指标，再运行全部 BMAD 六数据集。当前工作机缺少真实数据/模型与 CUDA PyTorch，本地数学/CPU fixture 检查不能证明新版真实性能。代码按默认设置提交推送，本报告和所有实验产物只留本地。',''])
    if 'brain' in main.index and 'fixed4_raw' in macro:
        fixed=df[(df.dataset=='brain')&(df.method=='fixed4_raw')].iloc[0]
        lines[5:5]=[f'Brain 是明确短板：Main Image AUROC={100*main.loc["brain","image_auroc"]:.2f}% vs Fixed-4 {100*fixed.image_auroc:.2f}%；Pixel AP={100*main.loc["brain","pixel_auprc"]:.2f}% vs {100*fixed.pixel_auprc:.2f}%。','']
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')


def write_report(output,df,summary,paired,prevalence,data):
    main=df[df.method=='main'].set_index('dataset').loc[DATASETS]
    macro=summary.groupby('method_family').image_auroc_mean.mean().sort_values(ascending=False)
    table=main[['selected_layers','feature_dim','image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].copy()
    for column in ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']:
        table[column]=table[column].map(lambda x:f'{x*100:.2f}' if pd.notna(x) else 'n.a.')
    table=table.reset_index()
    comparison=pd.DataFrame({'dataset':DATASETS})
    for family in ['main','fixed4_raw','asls_raw','randomk_raw','randomk_uvarfs','asls_random','asls_pca']:
        rows=summary[summary.method_family==family].set_index('dataset').loc[DATASETS]
        comparison[family]=[f'{r.image_auroc_mean*100:.2f} ± {r.image_auroc_std*100:.2f}'
                            if r.runs>1 else f'{r.image_auroc_mean*100:.2f}' for _,r in rows.iterrows()]
    efficiency=main[['selected_layer_count','feature_dim','candidate_dim','compression_ratio',
                     'memory_size','memory_vector_fp16_mib','matching_and_aggregation_seconds',
                     'fit_seconds','memory_seconds','eval_seconds','peak_gpu_gb']].copy().reset_index()
    efficiency['compression_ratio']*=100
    efficiency=efficiency.round(3)
    diagnostics=pd.DataFrame([{'dataset':d['dataset'],'pooled_gram_mean':f"{d['consensus_off_diagonal_mean']:.4f}",
                               'gate_span':f"{d['probability_span']:.6f}",
                               'gate_variability_rank':f"{d['gate_variability_rank_correlation']:.3f}",
                               'uvarfs_error':f"{d['uvarfs_geometry_error']:.5f}",
                               'step_change':f"{d['relative_change']:.2e}",
                               'projected_residual':f"{d['projected_residual']:.2e}"} for d in data['diagnostics']])
    pairs=paired.copy()
    for c in ['delta_auroc','ci95_lower','ci95_upper']:
        pairs[c]=pairs[c].map(lambda x:f'{x*100:+.2f}')
    lines=[
        '# BMAD v4 回传分析（2026-10-06）','',
        '归档说明：本报告评价数据来自 v4，修改建议记录批准前的 v5 状态。用户随后批准 patch 主方法；当前 v6 说明见 [升级记录](../../PATCH_ASLS_UPGRADE.md)，本报告不包含 v6 性能。','',
        f'这轮完成六数据集全部 {data["rows"]} 行（28 方法/数据集）和 Brain/Liver/RESC 像素评价。主方法 Macro Image AUROC={macro["main"]*100:.2f}%，Fixed-4 Raw={macro["fixed4_raw"]*100:.2f}%，同 K Random Raw 五 seed 均值={macro["randomk_raw"]*100:.2f}%。层选择仍是主要诊断对象，不能宣称主方法整体胜过固定/随机层。','',
        '主线始终是 Frozen DINOv3 + Adaptive Sparse Layer Selection + U-VaRFS 的无异常标签医学异常检测研究；本报告没有用测试标签选择层、lambda、维度或 detector 参数。','',
        '## 完整性与来源','',
        f'- 实验版本 `{data["experiment_version"]}`，Git revision `{data["git_revision"]}`；六份 config/code fingerprint 一致。',
        '- 全局 CSV 与逐数据集 CSV 相符；全部逐图 AUROC/AP 重新计算与 CSV 相符，family mean/SD 重新计算与回传 summary 相符。',
        '- 保存的 config 内容与指纹相符；'+('记录的 Git commit 源码字节与服务器 code fingerprint 核验相符。'
                                               if data.get('git_blob_code_verified') else '本地 Git 源码不可用，未确认记录的 code fingerprint。'),
        '- 训练与 memory manifest 为正常 train 数据；全部方法共享 memory 图像、patch、reservoir 抽样，均有 20000 bank rows。',
        '- Brain/Liver/RESC 异常 mask 分别覆盖 3075/3075、660/660、764/764；像素指标完整。',
        '- 只读取版本目录；根目录的 v3 汇总不能与 v4 混合。源文件哈希见 `analysis_data.json`。','',
        '## 主结果','', '指标为百分数；层与维度由 normal-only fit 决定。','',
        markdown_table(table,table.columns.tolist()),'',
        '![Image AUROC and localization](performance.png)','',
        'Liver localization 是正向证据：主方法 Pixel AUROC=98.34%、Pixel AP=17.32%、AUPRO=95.10%；RESC AUPRO 提升而 Pixel AUROC/AP 低于 All Raw；Brain 的 image 与 pixel 表现均明显落后 Fixed-4 Raw。Pixel AP 不能由很高的 Pixel AUROC 代替。','',
        '## 公平对照与配对不确定性','',
        'Image AUROC 百分数；全部 Random 行为五 seeds 的 mean ± sample SD。','',
        markdown_table(comparison,comparison.columns.tolist()),'',
        'Random-K 与 ASLS 的层数相同（本轮均为 K=2），五 seeds 全部纳入。主方法只在 Liver/OCT2017 的 Image AUROC 高于 Random-K Raw/Random-K U-VaRFS 均值。Random/PCA 仍为 baseline，不替换 main。', '',
        '在相同 ASLS 层输入下，U-VaRFS 在六数据集均高于同维度 Random feature 均值、四个数据集高于 PCA、五个数据集高于 Raw；RESC 相对 Raw 有下降。这支持继续诊断该模块，但尚不能支持完整方法跨模态优于固定/随机层。','',
        f'下表采用 {data["bootstrap_draws"]} 次 image-level、按正常/异常分层、同一图像配对 bootstrap；差值为 Main 减 baseline，单位百分点。Random 行每次 draw 内计算五个 seed 的 AUC 均值，未将预测均值当成 ensemble。区间仅评价固定方法，不用于拟合或调参。','',
        markdown_table(pairs[['dataset','comparison','delta_auroc','ci95_lower','ci95_upper']],['dataset','comparison','delta_auroc','ci95_lower','ci95_upper']), '',
        '没有独立患者/slide 分组 manifest，这些区间不控制同一患者或 slide 内相关性；没有进行多重比较校正或预设 non-inferiority 检验，不能据此宣称患者级显著性/非劣。', '',
        '## 正常训练诊断','',markdown_table(diagnostics,diagnostics.columns.tolist()), '',
        '六个 pooled consensus Gram 的 off-diagonal 均值 0.93–0.994，Brain/Liver/RESC 接近常数。所有 gates 接近全开且排序与 variability 同向。按当前目标，类似 Gram 的共同 gate p 使几何项约为 1-p；这一结构推动全开，微小 gate 差异再决定离散前缀。正常 pooled 几何可行性不代表局部 patch 几何或 anomaly 性能已保存。','',
        '可考虑在同一主线内用采样 normal patch geometry，保留原 sigmoid/Adam/loss 和 pooled 消融；主方法的输入切换需明确授权，并另立版本，不能根据 test AUROC 选版本。工程等价的 layer-Gram 内积矩阵可以避免存储全部 patch Gram。','',
        '六个 U-VaRFS main 均使用最小 lambda=1e-6，最终几何误差 0.052–0.071，全部不满足 0.05，使用原协议 minimum-error fallback。投影残差很小而相邻迭代变化未达 1e-5：不能把 solver_converged=false 简化为优化仍远离驻点，也不能凭小残差证明 feature support 稳定。需要严格 matmul 精度、逐 lambda momentum restart、原目标值与 box optimality-gap 证据。','',
        '保持 beta、lambda grid、geometry tolerance 和预算；额外 lambda=0 可作为 normal-only 不可行原因诊断，不能自动加入主选择路径。最终几何可行性与数值收敛是两件事。','',
        '## 稀疏与效率','',
        '主方法 2/12 层、87–250/4608 维；压缩率高不能自动推出总体显存或 backbone forward 加速。fit/memory/eval/peak 是整数据集全部方法共享量；matching_and_aggregation_seconds 只含 NN 与 image Top-K，未覆盖 feature transform、pixel metric 和绘图。','',
        '以全层 4608 维计算的 compression_ratio 包含层与特征两步裁剪；仅 U-VaRFS 阶段是选中两层的 768 维压缩到 87–250 维，不能把两步压缩率全部归功于特征选择。','',
        '下表 compression_ratio 为百分数，时间为秒，peak_gpu_gb 为 GiB；fit/memory/eval/peak 为共享整套方法开销，memory_vector_fp16_mib 仅为该方法向量容量。','',
        markdown_table(efficiency,['dataset','selected_layer_count','feature_dim','candidate_dim',
                                   'compression_ratio','memory_vector_fp16_mib','matching_and_aggregation_seconds']),'',
        '全部方法 memory bank 均为 20000 rows；下面的耗时/显存为整数据集全部方法共享量。','',
        markdown_table(efficiency,['dataset','fit_seconds','memory_seconds','eval_seconds','peak_gpu_gb']),'',
        '本轮带 mask 的 evaluation 开销远大于 NN matching。新版共享同图像 mask 连通域，pixel histogram 与 normal PRO counts 复用一次排序；保留原 1024 bins、100 thresholds、≥ ties 与 FPR=0.3。局部 CPU fixture 与 v4 对照的全部计数和最终指标完全一致，中位耗时比约 1.63（详见 pixel_equivalence_benchmark.json），不能外推为服务器整套实验加速。','',
        '回传 heatmap 使用固定 0–2 cosine distance 色阶，抽查样例对比度较低；不根据这些叠加图单独宣称定位效果，应结合 pixel AP/AUPRO 和原始分数。','',
        '## 类别比例','',
        markdown_table(prevalence.round(6),prevalence.columns.tolist()),'',
        'X-ray 异常比例约 95.46%，无技巧 AP 基线即该比例；主方法 AP=97.86% 要结合这一比例解释，同时保留 Image AUROC=71.33%。','',
        '## 修改边界与复跑','',
        '代码修改前检查：不改变研究问题、Frozen backbone、normal-only 约束、U-VaRFS 目标、cosine 1-NN 或统一预算；数值输出变化必须换版本与结果目录。ASLS geometry 输入若升级，需明确标注并保留 pooled 对照。','',
        '本次 v5 默认主 ASLS 仍用 pooled geometry；新增 asls_patch_raw/asls_patch_uvarfs 消融，并保留原 28 方法和五 seeds。patch 主方法切换需按 AGENTS.md 第 19 节得到明确指令，未静默启用。数值与诊断修改、复跑命令见 [修改说明](../../RESULT_DRIVEN_FIXES.md)。','',
        '当前工作机没有真实 BMAD 数据/权重或 CUDA PyTorch，因此本地只能验证数学、CPU fixture、统计与工程等价性。新版性能需先 Liver 验证，再六数据集复跑；不能用本轮 test metrics 选择新的训练超参数。', '',
        '来源：版本目录的 all_metrics/summary_metrics、六数据集 metrics/image_predictions/asls/uvarfs/fit_manifest/memory_manifest/run_metadata/mask_coverage/eval_progress。报告不会改写原实验产物。',''
    ]
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    family=pd.read_csv(output/'family_summary.csv')
    order=['main','asls_raw','fixed4_raw','all_raw','randomk_raw','randomk_uvarfs','asls_random','asls_pca']
    if 'asls_pooled_uvarfs' in family.method_family.values:
        order.insert(1,'asls_pooled_uvarfs')
    figure,axes=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1.5,1]})
    values=family.pivot(index='method_family',columns='dataset',values='image_auroc_mean').loc[order,DATASETS]*100
    im=axes[0].imshow(values.to_numpy(),vmin=40,vmax=100,cmap='viridis',aspect='auto')
    axes[0].set_xticks(range(6),DATASETS,rotation=30,ha='right'); axes[0].set_yticks(range(len(order)),order)
    axes[0].set_title('Image AUROC (%) — all seeds included')
    for i in range(len(order)):
        for j in range(6):
            axes[0].text(j,i,f'{values.iloc[i,j]:.1f}',ha='center',va='center',color='white' if values.iloc[i,j]<78 else 'black')
    figure.colorbar(im,ax=axes[0],shrink=.7)
    methods=['main','fixed4_raw','all_raw']
    for j,method in enumerate(methods):
        data=family[family.method_family==method].set_index('dataset').loc[['brain','liver','resc']]
        axes[1].bar(np.arange(3)+(j-1)*.25,data.aupro_mean*100,width=.25,label=method)
    axes[1].set_xticks(np.arange(3),['brain','liver','resc']); axes[1].set_ylabel('AUPRO (%)'); axes[1].set_ylim(0,100)
    axes[1].set_title('Localization (FPR ≤ 0.3)'); axes[1].legend(loc='lower right'); axes[1].grid(axis='y',alpha=.2)
    figure.tight_layout(); figure.savefig(output/'performance.png',dpi=180); figure.savefig(output/'performance.svg'); plt.close(figure)
    svg=output/'performance.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if args.plots_only:
        plot(args.out)
    else:
        if args.results is None or args.bootstrap<100:
            parser.error('--results and at least 100 bootstrap draws are required')
        audit(args.results,args.out,args.bootstrap)


if __name__=='__main__':
    main()
