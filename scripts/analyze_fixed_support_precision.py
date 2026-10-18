"""Audit every normal fit; never parse test data unless the global gate passed."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from uvarfs.precision_refit import VERSION,SETTINGS,weighted_names
from uvarfs.performance_uvarfs import project_bounded_simplex
from scripts.fixed_support_precision import read,sha256,strict_fit,strict_evaluation,validate_global_gate,METRICS
from scripts.analyze_results import WeightedRanking
from scripts.report_independent_fit_results import image_bootstrap,table

def analyze(batch,out,bootstrap=2000):
    project=Path(__file__).resolve().parents[1];batch=Path(batch).resolve();out=Path(out).resolve()
    if not batch.is_relative_to((project/'results').resolve()) or not out.is_relative_to((project/'reports').resolve()) or out.exists() or bootstrap!=2000:raise ValueError('fresh fixed project report required')
    start=read(batch/'startup.json');parent=(project/start['parent_relative']).resolve();gate=read(batch/'global-normal-gate.json')
    if sha256(parent/'state.json')!=start['parent_state_sha256']:raise ValueError('closed parent state changed')
    rows=[];tracked={}
    for seed in [42,123,3407]:
        for dataset in ['brain','liver','resc','oct2017','xray','camelyon16']:
            root=batch/'results'/f'seed{seed}'/dataset;done=strict_fit(root);meta=read(root/'run_metadata.json')
            if meta['fit_seed']!=seed or meta['dataset']!=dataset:raise ValueError('normal block identity differs')
            if meta['head']!=start['head'] or meta['code_fingerprint']!=start['code_fingerprint'] or meta['worker_sha256']!=start['worker_sha256']:raise ValueError('mixed source identity')
            specs=read(root/'representation-specs.json');parents=read(parent/'results'/f'seed{seed}'/dataset/'selection/representation-specs.json')
            for name,c in read(root/'normal-fit.json').items():
                p=torch.tensor(c['weights'],dtype=torch.float64);g=torch.tensor(c['final_gradient_p'],dtype=torch.float64)
                residual=float((p-project_bounded_simplex(p-g,1e-4,4.)).abs().max())
                if not np.isclose(residual,c['diagnostics']['residual'],rtol=1e-8,atol=1e-11):raise ValueError('independent original-unit residual differs')
                previous=parents[weighted_names()[name]]
                if c['active']!=previous['obj']['active'] or specs[name]['layers']!=previous['layers'] or not np.allclose(specs[name]['obj']['scales'],np.sqrt(p.numpy()),atol=1e-12,rtol=1e-12):raise ValueError('fixed support or layers changed')
                history=c['diagnostics']['training_history']
                if any(b>=a for a,b in zip(history,history[1:])) or not np.isclose(history[-1],c['training_objective'],atol=1e-12):raise ValueError('actual training loss not monotone')
                rows.append(dict(fit_seed=seed,dataset=dataset,method=name,dimension=len(p),residual=residual,precision_pass=residual<=SETTINGS['residual_tolerance'],
                    previous_residual=c['parent_residual'],training_loss=c['training_objective'],previous_training_loss=c['parent_training_objective'],
                    iterations=c['diagnostics']['iterations'],calls=c['diagnostics']['objective_gradient_calls'],termination=c['diagnostics']['termination'],seconds=c['diagnostics']['seconds']))
            for rel,record in read(root/'normal-source-sha256.json').items():
                path=(project/rel).resolve()
                if not path.is_relative_to((project/'results').resolve()) or sha256(path)!=record['sha256']:raise ValueError('normal source changed')
            for name in ['normal-fit.json','representation-specs.json','normal-complete.json']:tracked[(root/name).relative_to(project).as_posix()]=sha256(root/name)
            print(f'[precision-audit] seed{seed}/{dataset}: {done["precision_pass_count"]}/14 residual gates',flush=True)
    normal=pd.DataFrame(rows);passed=int(normal.precision_pass.sum())
    if len(normal)!=252 or gate['precision_pass_count']!=passed or gate['all_252_pass']!=(passed==252):raise ValueError('global gate/count differs')
    out.mkdir(parents=True);normal.to_csv(out/'normal_precision_all_252.csv',index=False)
    normal.groupby('method')[['residual','previous_residual','training_loss','previous_training_loss','iterations','calls','seconds']].agg(['mean','median','min','max']).to_csv(out/'normal_precision_method_summary.csv')
    normal.groupby(['dataset','method']).precision_pass.agg(['sum','count']).to_csv(out/'normal_precision_pass_counts.csv')
    report=['# 固定支持权重精度验证','',f'全部18区组、252个固定支持权重拟合已完成；原单位投影残差<=1e-5为{passed}/252。',
        'ASLS/支持/normal144+48+64角色/温度/原正常1NN+完整KL+nuisance目标保持；初值为各自不可变800步结果。未使用新的selection或audit值挑权重。不是新增患者或独立正常抽样。','',
        f'预设全局精度门槛：{gate["all_252_pass"]}。当前最近邻分支残差不是非光滑/全局最优证书。不同初值/支持仍可能达到不同局部点。']
    detector_rows=0
    if not gate['all_252_pass']:
        if any((batch/'results'/f'seed{s}'/d/'evaluation-complete.json').exists() for s in [42,123,3407] for d in ['brain','liver','resc','oct2017','xray','camelyon16']):raise ValueError('detector evaluation occurred despite failed normal gate')
        report+=['','按事先固定规则停止本轮检测评价，没有解析本轮或旧轮test成绩进行选择。本轮只能判定固定预算下尚未取得全组可比的当前分支精度，不能以此宣布检测改善或否定稀疏目标。旧630及更早所有控制仍在不可变项目来源，保留身份/成本。']
    else:
        validate_global_gate(gate,project);frames=[];oldframes=[];diffs=[];intervals=[]
        for seed in [42,123,3407]:
            for di,dataset in enumerate(['brain','liver','resc','oct2017','xray','camelyon16']):
                root=batch/'results'/f'seed{seed}'/dataset;strict_evaluation(root)
                f=pd.read_csv(root/'metrics.csv');old=pd.read_csv(parent/'results'/f'seed{seed}'/dataset/'metrics.csv')
                for frame in [f,old]:
                    for m in METRICS:
                        if m not in frame:frame[m]=np.nan
                frames.append(f);oldframes.append(old)
                pred=pd.read_csv(root/'image_predictions.csv',float_precision='round_trip');op=pd.read_csv(parent/'results'/f'seed{seed}'/dataset/'image_predictions.csv',float_precision='round_trip')
                values=f.set_index('method');groups={n:[n] for n in ['precision_sparse','precision_beta_zero','precision_dense_weighted','precision_dense_beta_zero','asls_raw']}
                for label,stem in [('precision Random weighted mean','precision_random_weighted'),('precision Random beta0 mean','precision_random_beta_zero')]:groups[label]=[stem+'_seed'+str(s) for s in [42,123,3407,2026,2027]]
                for name in ['mechanism_sparse','mechanism_sparse_uniform','dense_weighted','mechanism_beta_zero','asls_pca','fixed4_raw','all_raw']:
                    key='parent_'+name;pred[key]=op[name];values.loc[key]=old.set_index('method').loc[name];groups[key]=[key]
                groups['parent Random weighted mean']=[];groups['parent Random uniform mean']=[]
                for family,label in [('random_weighted','parent Random weighted mean'),('random_uniform','parent Random uniform mean')]:
                    for s in [42,123,3407,2026,2027]:
                        name=family+'_seed'+str(s);key='parent_'+name;pred[key]=op[name];values.loc[key]=old.set_index('method').loc[name];groups[label].append(key)
                for name in {v for group in groups.values() for v in group}:
                    auc,ap=WeightedRanking(pred.label,pred[name]).metrics(np.ones(len(pred)))
                    if not np.allclose([auc[0],ap[0]],values.loc[name,METRICS[:2]].astype(float),atol=1e-10,rtol=1e-10):raise ValueError('independent image metric mismatch')
                for name,members in groups.items():
                    if name!='precision_sparse':diffs.append(dict(fit_seed=seed,dataset=dataset,comparison=name,**(values.loc['precision_sparse',METRICS].astype(float)-values.loc[members,METRICS].astype(float).mean()).to_dict()))
                intervals += [dict(fit_seed=seed,dataset=dataset,**v) for v in image_bootstrap(pred,groups,'precision_sparse',bootstrap,seed+1000003*di)]
        fresh=pd.concat(frames);previous=pd.concat(oldframes);detector_rows=len(fresh);delta=pd.DataFrame(diffs);macro=delta.groupby(['fit_seed','comparison'])[METRICS].mean().reset_index()
        fresh.to_csv(out/'new_evaluation_metrics.csv',index=False);previous.to_csv(out/'previous_630_metrics.csv',index=False);delta.to_csv(out/'paired_differences.csv',index=False)
        macro.to_csv(out/'paired_macro_by_fit.csv',index=False);macro.groupby('comparison')[METRICS].agg(['mean','std','count']).to_csv(out/'macro_mean_sample_sd.csv')
        pd.DataFrame(intervals).to_csv(out/'conditional_image_intervals.csv',index=False)
        display=[dict(comparison=name,**{m:f'{100*v[m].mean():+.4f} ± {100*v[m].std(ddof=1):.4f}' if v[m].notna().any() else '无mask' for m in METRICS}) for name,v in macro.groupby('comparison')]
        report+=['','以下为主对象precision_sparse减去固定对照，百分点均值±三个来源fit样本SD；图像全六等权，定位三mask等权。','',table(pd.DataFrame(display),['comparison',*METRICS])]
    report+=['','旧630/6336/尾部108/岭180/pursuit180均保留原身份和成本，不把重复锚点或Random子集seed当独立Main重复。正常缓存与源fit复用，并非新患者；历史test多轮研发，未证明患者隔离。',
        '若进行了检测，2000次图像bootstrap条件于单fit，不是跨患者/跨fitCI，无完整像素地图bootstrap、多比较未校正。低正常loss/残差、工程提速或压缩不证明检测成功；注册Main不自动更换。']
    (out/'assessment.zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    for rel,h in tracked.items():
        if sha256(project/rel)!=h:raise ValueError('normal audit input changed')
    proof=dict(evaluation_version=VERSION,normal_blocks=18,normal_weighted_fits=252,precision_pass_count=passed,global_precision_gate_passed=passed==252,
        detector_evaluations=detector_rows,previous_mechanism_rows=630,retained_control_rows=6336,previous_tail_rows=108,previous_ridge_rows=180,previous_pursuit_rows=180,
        normal_source_sha256=tracked,all_used_inputs_unchanged=True,test_data_parsed=passed==252,bootstrap=bootstrap if passed==252 else None,
        main_replaced=False,new_independent_normal_sampling_claimed=False,goal_marked_complete=False)
    (out/'audit-provenance.json').write_text(json.dumps(proof,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args();analyze(a.batch,a.out,a.bootstrap)
