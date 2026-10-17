"""Independent audit of the fixed support-by-weighting experiment."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.evaluate_mechanism_validation import VERSION,METHODS,METRICS,read,sha256,strict_complete,normal_audit_methods
from scripts.evaluate_pursuit_support import strict_complete as pursuit_complete
from scripts.evaluate_ridge_support import strict_complete as ridge_complete
from scripts.report_independent_fit_results import image_bootstrap,table
from scripts.analyze_results import WeightedRanking
from uvarfs.mechanism_validation import SETTINGS,RANDOM_SEEDS,best_supports,artifact_digest
from uvarfs.pursuit_support import digest


def audit_fit(path):
    specs=read(path/'selection/representation-specs.json');asls=read(path/'selection/asls.json')
    selected=min(asls['selection_path'],key=lambda c:(c['selection_score'],c['layers']))
    if selected['layers']!=asls['selected_layers']:raise ValueError('ASLS nonminimum normal selection')
    width=len(asls['selected_layers'])*384
    for label,weighted,uniform in [('nuisance','mechanism_sparse','mechanism_sparse_uniform'),('beta_zero','mechanism_beta_zero','mechanism_beta_zero_uniform')]:
        data=read(path/'selection'/(label+'-fit.json'));pool=data['support_pool']
        if digest(pool)!=data['support_pool_fit_only_sha256'] or digest(pool[:data['base_pool_count']])!=data['base_pool_sha256']:raise ValueError('training support pool hash differs')
        parents=best_supports(pool[:data['base_pool_count']],SETTINGS['support_search']['starts'])
        if [a['parent_candidate_id'] for a in data['search_attempts']]!=[p['candidate_id'] for p in parents]:raise ValueError('non-fit-only support starting rule')
        supports=best_supports(pool,SETTINGS['support_search']['final_supports'])
        if [a['active'] for a in data['learned']]!=[p['active'] for p in supports]:raise ValueError('non-fit-only final support rule')
        clean={key:copy.deepcopy(data[key]) for key in ['support_pool','learned','random','dense']}
        for c in [*clean['learned'],*clean['random'].values(),clean['dense']]:
            for key in ['selection_objective','selection_terms','selection_score']:c.pop(key)
        if artifact_digest(clean)!=data['fit_only_artifacts_sha256']:raise ValueError('preselection fit artifact digest differs')
        for attempt in data['search_attempts']:
            for trace in attempt['accepted_trace']:
                if not trace['after']<trace['before']-1e-8 or abs(pool[trace['candidate_id']]['training_objective']-trace['after'])>1e-10:raise ValueError('nondecreasing support continuation')
        random=data['random']
        for seed in RANDOM_SEEDS:
            c=random[str(seed)]
            expected=np.sort(np.random.default_rng(seed).choice(width,256,replace=False)).tolist()
            if c['active']!=expected:raise ValueError('Random256 support differs')
            name=f'{"random_weighted" if label=="nuisance" else "random_beta_zero"}_seed{seed}'
            if specs[name]['obj']['active']!=c['active'] or not np.allclose(specs[name]['obj']['scales'],np.sqrt(c['weights']),rtol=1e-12,atol=1e-12):raise ValueError('weighted random representation differs')
        for c in [*data['learned'],*random.values(),data['dense']]:
            p=np.asarray(c['weights']);active=np.asarray(c['active']);k=len(p)
            if (len(set(active))!=k or active.min()<0 or active.max()>=width or abs(p.sum()-1)>1e-10 or
                p.min()<1e-4/k-1e-12 or p.max()>min(1,4/k)+1e-12 or c['initialization']!='uniform' or
                c['weight_solver']!='original_bounded_simplex_BB_Armijo' or c['diagnostics']['iterations']>800 or
                c['training_objective']>c['initial_uniform_objective']+1e-10):raise ValueError('common solver or feasible weights differ')
            if label=='beta_zero' and c['training_terms']['variability_term']!=0:raise ValueError('beta-zero nuisance retained')
            if not np.isclose(c['selection_score'],c['selection_objective']+c['training_terms']['variability_term']):raise ValueError('normal selection loss differs')
        winner=min(range(len(data['learned'])),key=lambda i:(data['learned'][i]['selection_score'],i))
        c=data['learned'][winner]
        if winner!=data['winner_index'] or specs[weighted]['obj']['active']!=c['active'] or specs[uniform]['obj']!={'active':c['active']} or not np.allclose(specs[weighted]['obj']['scales'],np.sqrt(c['weights']),rtol=1e-12,atol=1e-12):raise ValueError('selected learned representation differs')
        dense_name='dense_weighted' if label=='nuisance' else 'dense_beta_zero'
        if data['dense']['active']!=list(range(width)) or specs[dense_name]['obj']['active']!=list(range(width)):raise ValueError('full-dimensional control truncated')
    split=read(path/'selection/normal-role-split.json');audit=read(path/'normal-full-audit/protocol.json')
    manifest=read(path/'controls/fit_manifest.json');expected=[manifest['normal_train_paths'][i] for i in split['audit_image_positions']]
    memory=read(path/'normal-full-audit/memory_manifest.json')
    images=pd.read_csv(path/'normal-full-audit/image_scores.csv',float_precision='round_trip')
    if audit['used_for_selection'] or not audit['audit_queries_excluded_from_memory'] or set(expected)&set(memory['normal_train_paths']) or images.image_path.tolist()!=expected:raise ValueError('full normal audit query/reference identity differs')
    scores=images.drop(columns='image_path')
    if set(scores)!=set(normal_audit_methods(specs)) or not np.isfinite(scores).all().all():raise ValueError('normal audit method coverage invalid')
    errors=pd.read_csv(path/'normal-full-audit/normal_errors.csv').set_index('method')
    for name in scores:
        if not np.isclose(np.mean((scores[name]-scores.asls_raw)**2),errors.loc[name,'image_score_mse'],atol=1e-14,rtol=1e-10):raise ValueError('normal audit image MSE does not reproduce')


def analyze(batch,out,draws=2000):
    project=Path(__file__).resolve().parents[1];batch=Path(batch).resolve();out=Path(out).resolve()
    if not batch.is_relative_to((project/'results').resolve()) or not out.is_relative_to((project/'reports').resolve()) or out.exists() or draws!=2000:raise ValueError('fresh project-local fixed full audit required')
    start=read(batch/'startup.json');history=(project/start['history_batch_relative']).resolve()
    previous_start=read(history/'startup.json');ridge=(project/previous_start['previous_ridge_batch_relative']).resolve()
    if sha256(history/'state.json')!=start['history_state_sha256'] or sha256(ridge/'state.json')!=previous_start['previous_ridge_state_sha256']:raise ValueError('historical closed batch state differs')
    frames=[];oldframes=[];historyframes=[];tailframes=[];ridgeframes=[];differences=[];intervals=[];costs=[];normal=[];tracked={};blocks=[];configs=set()
    names=['brain','liver','resc','oct2017','xray','camelyon16']
    for seed in [42,123,3407]:
        for di,dataset in enumerate(names):
            path=batch/'results'/f'seed{seed}'/dataset;block=strict_complete(path);blocks.append(block)
            meta=read(path/'run_metadata.json');audit_fit(path)
            for key,target in [('evaluation_git_head','head'),('evaluation_code_fingerprint','code_fingerprint'),('worker_script_sha256','worker_sha256'),('fitter_script_sha256','fitter_sha256')]:
                if meta[key]!=start[target]:raise ValueError('mixed evaluation source')
            cfg=dict(meta['source_config']);cfg.pop('seed');cfg.pop('results_dir');configs.add(json.dumps(cfg,sort_keys=True))
            f=pd.read_csv(path/'metrics.csv');o=pd.read_csv(path/'controls/metrics.csv')
            hp=project/meta['history_source_relative'];cp=project/meta['cache_source_relative'];rp=ridge/'results'/f'seed{seed}'/dataset
            pursuit_complete(hp);ridge_complete(rp)
            h,t,r=[pd.read_csv(p/'metrics.csv') for p in [hp,cp,rp]]
            for frame in [f,o,h,t,r]:
                for metric in METRICS:
                    if metric not in frame:frame[metric]=np.nan
            frames.append(f);oldframes.append(o.assign(fit_seed=seed));historyframes.append(h);tailframes.append(t);ridgeframes.append(r)
            c=read(path/'costs.json');costs.append(dict(fit_seed=seed,dataset=dataset,**c))
            nm=pd.read_csv(path/'normal-full-audit/normal_errors.csv');normal.append(nm.assign(fit_seed=seed,dataset=dataset))
            values=f.set_index('method');groups={n:[n] for n in ['mechanism_sparse','mechanism_sparse_uniform','mechanism_beta_zero','mechanism_beta_zero_uniform','asls_raw','asls_pca','dense_weighted','dense_beta_zero','fixed4_raw','all_raw','last_raw']}
            for label,stem in [('Random weighted mean','random_weighted'),('Random uniform mean','random_uniform'),('Random beta0 mean','random_beta_zero'),('Random-K layer mean','randomk_raw')]:groups[label]=[f'{stem}_seed{s}' for s in RANDOM_SEEDS]
            pred=pd.read_csv(path/'image_predictions.csv',float_precision='round_trip')
            sources=[('historical_v22_',o,path/'controls',['main','asls_raw','v21_main','local_main']),
                     ('historical_pursuit_',h,hp,['pursuit_matching']),('historical_tail_',t,cp,['nuisance_tail']),('historical_ridge_',r,rp,['ridge_matching'])]
            for prefix,frame,root,methods in sources:
                pp=pd.read_csv(root/'image_predictions.csv',float_precision='round_trip')
                for key in ['image_path','label','mask_path']:
                    if not pred[key].fillna('').equals(pp[key].fillna('')):raise ValueError('historical evaluation roster differs')
                indexed=frame.set_index('method')
                for method in methods:
                    name=prefix+method;pred[name]=pp[method];values.loc[name]=indexed.loc[method];groups[name]=[name]
            for method in set(METHODS)|{n for members in groups.values() for n in members}:
                auc,ap=WeightedRanking(pred.label,pred[method]).metrics(np.ones(len(pred)))
                if not np.allclose([auc[0],ap[0]],values.loc[method,METRICS[:2]].to_numpy(dtype=float),atol=1e-10,rtol=1e-10):raise ValueError('independent image point estimate differs')
            target=values.loc['mechanism_sparse',METRICS].astype(float)
            for name,members in groups.items():
                if name!='mechanism_sparse':differences.append(dict(fit_seed=seed,dataset=dataset,comparison=name,**(target-values.loc[members,METRICS].astype(float).mean()).to_dict()))
            for row in image_bootstrap(pred,groups,'mechanism_sparse',draws,seed+1000003*di):intervals.append(dict(fit_seed=seed,dataset=dataset,**row))
            for rel in ['metrics.csv','image_predictions.csv','run_metadata.json','selection/nuisance-fit.json','selection/beta_zero-fit.json','normal-full-audit/image_scores.csv']:
                tracked[(path/rel).relative_to(project).as_posix()]=sha256(path/rel)
            print(f'[mechanism-audit] seed{seed}/{dataset}: 35 fixed results, four anchors and isolated normal audit verified',flush=True)
    if len(configs)!=1:raise ValueError('mixed source protocol')
    for rel,h in tracked.items():
        if sha256(project/rel)!=h:raise ValueError('analysis input changed')
    fresh=pd.concat(frames,ignore_index=True);delta=pd.DataFrame(differences);macro=delta.groupby(['fit_seed','comparison'])[METRICS].mean().reset_index()
    out.mkdir(parents=True)
    outputs={'new_evaluation_metrics':fresh,'retained_352_control_metrics':pd.concat(oldframes,ignore_index=True),
             'historical_pursuit_metrics':pd.concat(historyframes,ignore_index=True),'historical_tail_metrics':pd.concat(tailframes,ignore_index=True),
             'historical_ridge_metrics':pd.concat(ridgeframes,ignore_index=True),'paired_differences_by_fit':delta,'paired_macro_by_fit':macro,
             'conditional_image_intervals':pd.DataFrame(intervals),'new_costs':pd.DataFrame(costs),'isolated_normal_full_audit':pd.concat(normal,ignore_index=True)}
    for name,frame in outputs.items():frame.to_csv(out/(name+'.csv'),index=False)
    delta.groupby(['dataset','comparison'])[METRICS].agg(['mean','std','count']).to_csv(out/'dataset_delta_mean_sample_sd.csv')
    macro.groupby('comparison')[METRICS].agg(['mean','std','count']).to_csv(out/'macro_delta_mean_sample_sd.csv')
    absolute=fresh.groupby(['fit_seed','method'])[METRICS].mean();absolute.to_csv(out/'absolute_macro_by_fit.csv')
    absolute.groupby('method')[METRICS].agg(['mean','std','count']).to_csv(out/'absolute_macro_mean_sample_sd.csv')
    random=fresh[fresh.method.str.startswith(('random_weighted_','random_uniform_','random_beta_zero_','randomk_raw_'))].copy()
    random['family']=random.method.str.replace(r'_seed\d+$','',regex=True);random.groupby(['fit_seed','dataset','family'])[METRICS].agg(['mean','std','count']).to_csv(out/'five_random_within_fit_mean_sd.csv')
    display=[]
    for comparison,part in macro.groupby('comparison'):
        display.append({'comparison':comparison,**{m:f'{100*part[m].mean():+.4f} ± {100*part[m].std(ddof=1):.4f}' if part[m].notna().any() else '无mask' for m in METRICS}})
    raw=macro[macro.comparison.eq('asls_raw')];gate=bool(raw.image_auroc.mean()>0 and raw.pixel_auprc.mean()>0)
    lines=['# 支持×权重机制实验：固定全六、三个来源种子','',
           '630新评价/18区组，旧6336、尾部108、岭180及pursuit180保留原身份。正常缓存复用；所有ASLS/支持/权重重新拟合144fit/48selection，64图在表示冻结后审计。不是新增独立正常抽样/新患者。',
           '主对象mechanism_sparse，历史Raw输入/拟合协议不同，historical_*只作历史比较。新同输入Raw为asls_raw。',
           '以下为百分点均值±三个来源fit样本SD。图像全六等权，定位三mask集等权。','',table(pd.DataFrame(display),['comparison',*METRICS]),'',
           f'预设研发方向判据（Image AUROC与Pixel AUPRC相对新Raw均值均为正）：{gate}。不等于独立患者确认，也不自动更换注册Main。',
           '五Random weighted/uniform/β0的within-fit SD单独报告，不当Main重复。正常audit从当前ASLS/特征选择中隔离，但这些图像历史已用于研发；真实full-grid audit memory排除全部64 query图像。',
           '相同权重solver/初始化/最大步数不保证相同收敛精度；报告实际残差/迭代/成本。正常目标/审计不证明未知异常判别。',
           '2000次图像bootstrap条件于单个fit，不是患者/跨fitCI；无完整像素地图bootstrap，多比较未校正，历史BMAD test多轮研发且未证明患者隔离。后续确认需未接触患者/站点或预注册外层验证。']
    (out/'assessment.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    proof=dict(evaluation_version=VERSION,blocks=blocks,new_evaluations=len(fresh),retained_control_rows=6336,
        previous_tail_rows=108,previous_ridge_rows=180,previous_pursuit_rows=180,bootstrap=draws,
        source_sha256=tracked,all_used_inputs_unchanged=True,normal_roles=SETTINGS['normal_roles'],
        independent_source_fit_seeds=[42,123,3407],new_independent_normal_sampling_claimed=False,
        four_anchors_per_block_exact=True,main_replaced=False,directional_rnd_gate_passed=gate)
    (out/'audit-provenance.json').write_text(json.dumps(proof,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args();analyze(a.batch,a.out,a.bootstrap)
