"""Independent algebraic/array audit of fixed normal probes, without test files."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.normal_endpoint_diagnostics import read,atomic,sha256,strict_complete
from uvarfs.endpoint_diagnostics import VERSION,METHODS,STEPS,DIRECTIONS
from uvarfs.performance_uvarfs import project_bounded_simplex

def analyze(batch,out):
    project=Path(__file__).resolve().parents[1];batch=Path(batch).resolve();out=Path(out).resolve()
    if not batch.is_relative_to(project) or not out.is_relative_to(project) or not batch.is_relative_to((project/'results').resolve()) or not out.is_relative_to((project/'reports').resolve()) or out.exists():raise ValueError('fresh project-local audit required')
    start=read(batch/'startup.json');source=project/start['source_relative']
    if sha256(source/'state.json')!=start['source_state_sha256']:raise ValueError('closed source state changed')
    endpoints=[];probes=[];sources={};artifacts={}
    for dataset in ['liver','brain','resc','xray','camelyon16','oct2017']:
        root=batch/'results'/dataset;done=strict_complete(root);meta=read(root/'run_metadata.json')
        if done['dataset']!=dataset or meta['head']!=start['head'] or meta['code_fingerprint']!=start['code_fingerprint'] or meta['worker_sha256']!=start['worker_sha256'] or meta['config_sha256']!=start['config_sha256']:raise ValueError('mixed diagnostic provenance')
        original=read(source/'results/seed42'/dataset/'normal-fit.json')
        for rel,r in read(root/'source-sha256.json').items():
            p=(project/rel).resolve()
            if not p.is_relative_to(project) or sha256(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('normal source changed')
            sources[rel]=r
        for method in METHODS:
            path=root/(method+'.json');r=read(path);artifacts[path.relative_to(project).as_posix()]=sha256(path)
            p=np.array(r['weights']);g=np.array(r['gradient_p']);k=len(p);lower=1e-4/k;upper=min(1.,4/k)
            if r['active']!=original[method]['active'] or not np.array_equal(p,original[method]['weights']):raise ValueError('endpoint weights/support changed')
            residual=float((torch.tensor(p)-project_bounded_simplex(torch.tensor(p-g),1e-4,4.)).abs().max())
            if not np.isclose(residual,r['projected_residual'],rtol=1e-8,atol=1e-11) or [d['name'] for d in r['directions']]!=list(DIRECTIONS):raise ValueError('residual/direction plan differs')
            if method not in {'precision_dense_weighted','precision_dense_beta_zero'} and k!=256:raise ValueError('sparse K256 differs')
            with np.load(root/(method+'.npz'),allow_pickle=False) as arrays:
                nearest=arrays['nearest'];gap=arrays['gap']
                if len(nearest)!=r['normal_queries'] or np.any(gap<0) or not np.allclose(np.quantile(gap,[0,.01,.1,.5,1]),r['gap_quantiles'],atol=1e-12):raise ValueError('gap distribution differs')
                if int((arrays['exact_tie_count']>1).sum())!=r['exact_tied_queries'] or int((arrays['roundoff_near_tie_count']>1).sum())!=r['roundoff_near_tied_queries']:raise ValueError('tie counts differ')
                expected={'nearest','second','gap','exact_tie_count','roundoff_near_tie_count'};count=0
                for di,d in enumerate(r['directions']):
                    delta=np.array(d['delta_p']);slope=float(g@delta)
                    if abs(delta.sum())>1e-10 or (p+delta).min()<lower-1e-12 or (p+delta).max()>upper+1e-12:raise ValueError('direction infeasible')
                    if not np.isclose(slope,d['gradient_dot_direction'],rtol=1e-9,atol=1e-11) or not np.isclose(slope,d['independent_current_branch_slope'],rtol=1e-8,atol=1e-10):raise ValueError('slope calculation differs')
                    rows=[v for v in r['probes'] if v['direction']==d['name']]
                    if d['zero_direction']:
                        if rows:raise ValueError('zero direction falsely probed')
                        continue
                    if [v['step_fraction'] for v in rows]!=list(STEPS):raise ValueError('fixed probe grid differs')
                    for si,row in enumerate(rows):
                        count+=1;h=row['step_fraction'];change=row['actual_loss']-r['loss'];candidate=p+h*delta
                        names=[f'd{di}_h{si}_nearest',f'd{di}_h{si}_gap'];expected.update(names)
                        if int((arrays[names[0]]!=nearest).sum())!=row['changed_neighbor_queries'] or np.any(arrays[names[1]]<0):raise ValueError('saved nearest branch changes differ')
                        if not np.isclose(change,row['loss_difference'],rtol=1e-10,atol=1e-14) or not np.isclose(change/h,row['one_sided_quotient'],rtol=1e-10,atol=1e-12):raise ValueError('finite difference differs')
                        if not np.isclose(np.max(np.abs(candidate-p)),row['max_abs_p_delta'],rtol=1e-8,atol=1e-14) or abs(candidate.sum()-1)>1e-10 or candidate.min()<lower-1e-12 or candidate.max()>upper+1e-12:raise ValueError('probe feasibility differs')
                        probes.append(dict(dataset=dataset,method=method,source_seed=42,gradient_dot_direction=slope,exact_tie_directional_slope=d['observed_exact_tie_directional_slope'],**row))
                if set(arrays.files)!=expected or count!=len(r['probes']) or r['actual_objective_calls']!=count+2 or r['geometry_passes']!=count+1:raise ValueError('array/call coverage differs')
            endpoints.append(dict(dataset=dataset,method=method,source_seed=42,dimension=k,loss=r['loss'],residual=residual,queries=r['normal_queries'],
                minimum_neighbor_gap=r['gap_quantiles'][0],median_neighbor_gap=r['gap_quantiles'][3],exact_tied_queries=r['exact_tied_queries'],roundoff_near_tied_queries=r['roundoff_near_tied_queries'],
                actual_objective_calls=r['actual_objective_calls'],geometry_passes=r['geometry_passes']))
        print(f'[endpoint-audit] {dataset}: 5 endpoints and all fixed probes verified',flush=True)
    if len(endpoints)!=30:raise ValueError('complete predefined endpoint set required')
    out.mkdir(parents=True);f=pd.DataFrame(endpoints);p=pd.DataFrame(probes)
    f.to_csv(out/'all_30_endpoints.csv',index=False);p.to_csv(out/'all_directional_probes.csv',index=False)
    p['resolved_branch_descent_actual_increase']=(p.gradient_dot_direction<0)&(p.loss_difference>p.roundoff_scale)
    p.groupby(['method','direction']).agg(probes=('step_fraction','size'),changed_neighbor_probes=('changed_neighbor_queries',lambda s:int((s>0).sum())),roundoff_dominated_probes=('difference_within_roundoff_scale','sum'),resolved_branch_descent_actual_increases=('resolved_branch_descent_actual_increase','sum')).to_csv(out/'probe_summary.csv')
    atomic(out/'audit-provenance.json',dict(evaluation_version=VERSION,source_seed=42,blocks=6,endpoints=30,probes=len(p),
        independent_residual_slopes_differences_and_branch_arrays_verified=True,source_sha256=sources,endpoint_sha256=artifacts,
        test_data_used=False,selection_or_audit_features_used=False,optimizer_run=False,new_independent_sampling_claimed=False,goal_achieved=False))
    (out/'assessment.zh.md').write_text(f'# 纯正常停滞端点诊断\n\n全六seed42、30个固定端点、{len(p)}个非零方向步长探针完整；原端点/支持/目标未改变。所有实际目标、当前分支解析斜率、观测精确并列方向导数、单侧差商及近邻ID变化完整报告。\n\n小间隔或方向预测失配只是诊断记录：有限步有曲率、浮点舍入与邻居切换，不能单独判定原因或给出非光滑/全局最优证书。近似并列不提升为精确并列。未读test、未拟合/选择新权重、未运行检测，无性能结论。只有来源seed42，非新的三fit/患者独立验证。注册Main不变。\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.batch,a.out)
