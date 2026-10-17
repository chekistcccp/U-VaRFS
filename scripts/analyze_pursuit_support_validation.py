#!/usr/bin/env python3
"""Audit all 18 projected-pursuit blocks, including the prior ridge controls."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.evaluate_pursuit_support import VERSION,METRICS,METHODS,read,sha256,strict_complete
from scripts.report_independent_fit_results import families,image_bootstrap,table
from scripts.analyze_results import WeightedRanking
from uvarfs.pursuit_support import digest,VERSION as FIT_VERSION
from uvarfs.normal_tail_selection import validate_frozen_pool
from scripts.evaluate_ridge_support import strict_complete as ridge_complete
from scripts.fit_pursuit_support import EXPECTED


def audit_fitted_extension(path):
    specs=read(path/'selection/representation-specs.json')
    report=read(path/'selection/report.json')
    if report['labels_or_masks_used'] or report['registered_main_replaced']:
        raise ValueError('normal-only extension scope differs')
    for label,source_file,names in [
        ('nuisance','uvarfs_main.json',['pursuit_matching','pursuit_only','v22_anchor']),
        ('beta_zero','uvarfs_asls_no_variability.json',['pursuit_beta_zero','pursuit_only_beta_zero','v22_beta_zero_anchor'])]:
        fit=read(path/'selection'/(label+'-fit.json'));source=read(path/'controls'/source_file)
        if (fit['source']['start_policy']!='lowest_fit_loss_unique_supports' or fit['source']['settings']!=EXPECTED['solver']):
            raise ValueError('pursuit fit-only starting policy or fixed solver settings differ')
        width=len(source['budgeted_weights']);original=validate_frozen_pool(source,width)
        count=len(source['candidate_pool']);pool=fit['candidate_pool'];k=source['fixed_feature_budget']
        unique={}
        for c in source['candidate_pool']:
            key=tuple(c['active'])
            if key not in unique or (c['training_objective'],c['candidate_id'])<(unique[key]['training_objective'],unique[key]['candidate_id']):unique[key]=c
        parents=sorted(unique.values(),key=lambda c:(c['training_objective'],c['candidate_id']))[:EXPECTED['solver']['starts']]
        if [a['parent_candidate_id'] for a in fit['source']['attempts']]!=[c['candidate_id'] for c in parents]:
            raise ValueError('pursuit starts do not follow training-only rule')
        for attempt in fit['source']['attempts']:
            for trace in attempt['pursuit']['accepted_trace']:
                if (not trace['after']<trace['before']-1e-8 or
                    abs(pool[trace['candidate_id']]['training_objective']-trace['after'])>1e-10):
                    raise ValueError('pursuit continuation did not improve actual training objective')
        if (fit['extension_version']!=FIT_VERSION or fit['objective_definition']!=source['objective_definition'] or
            fit['support_generation_uses_holdout'] or fit['source']['generation_uses_holdout'] or
            fit['source']['source_pool_sha256']!=original or digest(pool[:count])!=original or
            digest(pool)!=fit['candidate_pool_fit_only_sha256'] or fit['beta']!=source['beta']):
            raise ValueError('changed objective, original pool, or fit-only extension hash')
        for i,c in enumerate(pool):
            p=np.asarray(c['weights'],dtype=float);active=np.asarray(c['active'])
            if (c['candidate_id']!=i or p.shape!=(k,) or len(active)!=k or len(set(active))!=k or
                not np.issubdtype(active.dtype,np.integer) or active.min()<0 or active.max()>=width or
                not np.isfinite(p).all() or abs(p.sum()-1)>1e-10 or
                p.min()<source['weight_floor_mass']/k-1e-12 or p.max()>min(1,source['weight_cap_factor']/k)+1e-12):
                raise ValueError('invalid sparse pursuit candidate budget/weights')
            if source['beta']==0 and c['training_terms']['variability_term']!=0:
                raise ValueError('beta-zero retained nuisance loss')
            if c.get('validation_valid') and not np.isclose(c['selection_score'],
                c['validation_objective']+c['training_terms']['variability_term'],rtol=1e-10,atol=1e-12):
                raise ValueError('normal selection loss differs')
        valid=[c for c in pool if c.get('validation_valid')]
        selected=min(valid,key=lambda c:(c['selection_score'],c['candidate_id']))
        pursuit=[c for c in valid if c['candidate_id']>=count]
        pursuit_selected=min(pursuit,key=lambda c:(c['selection_score'],c['candidate_id'])) if pursuit else selected
        if selected['candidate_id']!=fit['winner_candidate_id'] or pursuit_selected['candidate_id']!=fit['pursuit_only_candidate_id']:
            raise ValueError('nonminimum normal selection')
        for name,c in zip(names[:2],[selected,pursuit_selected]):
            obj=specs[name]['obj'];uniform=specs[name+'_uniform']['obj']
            if obj['active']!=c['active'] or uniform['active']!=c['active'] or not np.allclose(obj['scales'],np.sqrt(c['weights']),atol=1e-12,rtol=1e-10):
                raise ValueError('fitted representation differs from normal-selected candidate')
        if specs[names[2]]['obj']!={'active':source['active'],'scales':source['scales']}:
            raise ValueError('v22 anchor representation differs')


def summarize(frame,keys):
    rows=[]
    for key,part in frame.groupby(keys):
        key=key if isinstance(key,tuple) else (key,);row=dict(zip(keys,key))
        for metric in METRICS:
            values=100*part[metric].dropna()
            row[metric]=f'{values.mean():+.4f} ± {values.std(ddof=1):.4f}' if len(values) else '无mask'
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(batch,out,draws=2000):
    project=Path(__file__).resolve().parents[1];batch=Path(batch).resolve();out=Path(out).resolve()
    if not batch.is_relative_to((project/'results').resolve()) or not out.is_relative_to((project/'reports').resolve()) or out.exists():
        raise ValueError('project-local input and fresh project reports output required')
    startup=read(batch/'startup.json');source_frames=[];new_frames=[];tail_frames=[];ridge_frames=[];differences=[];intervals=[];tracked={}
    previous=(project/startup['previous_ridge_batch_relative']).resolve()
    if (not previous.is_relative_to((project/'results').resolve()) or sha256(previous/'state.json')!=startup['previous_ridge_state_sha256']):
        raise ValueError('previous ridge project path/state differs')
    tracked[str((previous/'state.json').relative_to(project))]=startup['previous_ridge_state_sha256']
    datasets=['brain','liver','resc','oct2017','xray','camelyon16'];settings=set();source_codes=set();blocks=[]
    for seed in [42,123,3407]:
        for di,dataset in enumerate(datasets):
            path=batch/'results'/f'seed{seed}'/dataset;block=strict_complete(path)
            if block['fit_seed']!=seed or block['dataset']!=dataset:raise ValueError('fit/dataset identity differs')
            meta=read(path/'run_metadata.json')
            if meta['evaluation_git_head']!=startup['head'] or meta['evaluation_code_fingerprint']!=startup['code_fingerprint'] or meta['worker_script_sha256']!=startup['worker_sha256'] or meta['fitter_script_sha256']!=startup['fitter_sha256']:
                raise ValueError('mixed evaluation source identities')
            cfg=dict(meta['source_config']);cfg.pop('seed');cfg.pop('results_dir')
            settings.add(json.dumps(cfg,sort_keys=True));source_codes.add(meta['source_code_fingerprint'])
            for relative in ['metrics.csv','controls/metrics.csv','image_predictions.csv','controls/image_predictions.csv','run_metadata.json','previous_tail/metrics.csv','previous_tail/image_predictions.csv','selection/report.json','selection/nuisance-fit.json','selection/beta_zero-fit.json']:
                tracked[str((path/relative).relative_to(project))]=sha256(path/relative)
            fresh=pd.read_csv(path/'metrics.csv');old=pd.read_csv(path/'controls/metrics.csv')
            previous=pd.read_csv(path/'previous_tail/metrics.csv')
            previous['method']='tail_'+previous.method
            for metric in METRICS:
                if metric not in previous:previous[metric]=np.nan
            tail_frames.append(previous)
            ridge_path=(project/startup['previous_ridge_batch_relative']/'results'/f'seed{seed}'/dataset).resolve()
            rb=ridge_complete(ridge_path)
            if rb['fit_seed']!=seed or rb['dataset']!=dataset:raise ValueError('previous ridge paired identity differs')
            ridge=pd.read_csv(ridge_path/'metrics.csv');ridge['method']='previous_ridge_'+ridge.method
            for metric in METRICS:
                if metric not in ridge:ridge[metric]=np.nan
            ridge_frames.append(ridge)
            for relative in ['metrics.csv','image_predictions.csv','run_metadata.json']:
                tracked[str((ridge_path/relative).relative_to(project))]=sha256(ridge_path/relative)
            audit_fitted_extension(path)
            for metric in METRICS:
                if metric not in fresh:fresh[metric]=np.nan
                if metric not in old:old[metric]=np.nan
            new_frames.append(fresh);source_frames.append(old.assign(fit_seed=seed))
            f=fresh.set_index('method');o=old.set_index('method');t=previous.set_index('method');r=ridge.set_index('method')
            groups=families(set(old.method),'',[42,123,3407,2026,2027])
            comps={name:f.loc[name,METRICS].astype(float) for name in sorted(METHODS-{'pursuit_matching'})}
            comps.update({name:t.loc[name,METRICS].astype(float) for name in t.index})
            comps.update({name:r.loc[name,METRICS].astype(float) for name in r.index})
            comps.update({'source_'+name:o.loc[names,METRICS].mean() for name,names in groups.items()})
            for name,values in comps.items():
                differences.append({'fit_seed':seed,'dataset':dataset,'comparison':name,
                                    **(f.loc['pursuit_matching',METRICS].astype(float)-values).to_dict()})
            old_groups={'source_'+name:names for name,names in groups.items()}
            bootgroups={name:[name] for name in sorted(METHODS)}|{name:[name] for name in t.index}|{name:[name] for name in r.index}|old_groups
            needed={name for names in bootgroups.values() for name in names}
            a=pd.read_csv(path/'image_predictions.csv',float_precision='round_trip')
            b=pd.read_csv(path/'controls/image_predictions.csv',usecols=lambda c:c in needed,float_precision='round_trip')
            c=pd.read_csv(path/'previous_tail/image_predictions.csv',float_precision='round_trip')
            c=c.drop(columns=['image_path','label','mask_path']).add_prefix('tail_')
            d=pd.read_csv(ridge_path/'image_predictions.csv',float_precision='round_trip')
            for column in ['image_path','label','mask_path']:
                if not a[column].fillna('').equals(d[column].fillna('')):raise ValueError('previous ridge test roster differs')
            d=d.drop(columns=['image_path','label','mask_path']).add_prefix('previous_ridge_')
            predictions=pd.concat([a,b,c,d],axis=1)
            for name in needed:
                values=f.loc[name] if name in f.index else t.loc[name] if name in t.index else r.loc[name] if name in r.index else o.loc[name]
                auc,ap=WeightedRanking(predictions.label,predictions[name]).metrics(np.ones(len(predictions)))
                if not np.allclose([auc[0],ap[0]],values[METRICS[:2]].to_numpy(dtype=float),atol=1e-10,rtol=1e-10):
                    raise ValueError('independent image point estimates differ')
            for row in image_bootstrap(predictions,bootgroups,'pursuit_matching',draws,seed+1000003*di):
                intervals.append({'fit_seed':seed,'dataset':dataset,**row})
            blocks.append(block);print(f'[pursuit-audit] seed{seed}/{dataset}: ten new evaluations and all 352 controls verified',flush=True)
    if len(settings)!=1 or len(source_codes)!=1:raise ValueError('source fits must share all settings except seed/results directory')
    for rel,digest in tracked.items():
        if sha256(project/rel)!=digest:raise ValueError('analysis inputs changed')
    new=pd.concat(new_frames,ignore_index=True);old=pd.concat(source_frames,ignore_index=True)
    delta=pd.DataFrame(differences);macro=delta.groupby(['fit_seed','comparison'])[METRICS].mean().reset_index()
    out.mkdir(parents=True)
    for name,frame in [('new_evaluation_metrics',new),('retained_352_control_metrics',old),
                       ('previous_tail_metrics',pd.concat(tail_frames,ignore_index=True)),('previous_ridge_metrics',pd.concat(ridge_frames,ignore_index=True)),('paired_differences_by_fit',delta),('paired_macro_by_fit',macro),
                       ('conditional_image_intervals',pd.DataFrame(intervals))]:frame.to_csv(out/(name+'.csv'),index=False)
    delta.groupby(['dataset','comparison'])[METRICS].agg(['mean','std','count']).to_csv(out/'dataset_delta_mean_sample_sd.csv')
    macro.groupby('comparison')[METRICS].agg(['mean','std','count']).to_csv(out/'macro_delta_mean_sample_sd.csv')
    display=summarize(macro,['comparison'])
    lines=['# 梯度投影支持追踪：全六、三个来源fit的独立配对延续','',
           '每区组新拟合pursuit支持及权重并评价10表示，18区组180行；旧352对照共6336行、尾部108行及岭支持180行分别保留。',
           '来源fit/memory seeds42/123/3407沿用冻结v22及其normal缓存；新增支持拟合，但没有新增三次独立抽样或新患者测试。',
           '原正常1-NN分数/完整KL/nuisance目标、ASLS/输入/K256/memory/detector保持。主要比较对象预先指定为pursuit_matching；pursuit_only和β0/uniform用于解释，不能按test切换主结论。',
           '下表为pursuit_matching减对照的百分点均值±三个来源fit样本SD，图像全六等权、定位仅三个mask集等权。','',
           table(display,['comparison',*METRICS]),'',
           '2000次图像bootstrap条件于单个拟合，不是患者/跨fit置信区间；多比较未校正，没有完整像素地图重采样。',
           '旧v22及β0逐图分数必须完全复现、全部适用image/pixel指标差<=1e-12且normal memory抽样一致；原候选池逐哈希保留。',
           '全坐标梯度用于原始坐标的联合投影/支持追踪，精确投影只证明单步欧氏子问题最优；不证明非凸匹配全局最优或异常判别保证。',
           '历史test用于多轮研发；64正常留出曾用于ASLS。本分支不自动替换注册Main，不以低正常loss/压缩/工程提速宣布成功。完整新成本见每区组costs.json，旧拟合成本保持原身份。']
    (out/'assessment.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic={'evaluation_version':VERSION,'blocks':blocks,'previous_tail_rows':sum(len(f) for f in tail_frames),'previous_ridge_rows':sum(len(f) for f in ridge_frames),'new_evaluations':len(new),'retained_control_rows':len(old),
            'independent_source_fit_seeds':[42,123,3407],'new_fit_repeats_claimed':False,'bootstrap':draws,
            'source_sha256':tracked,'all_used_inputs_unchanged':True,'main_replaced':False}
    (out/'audit-provenance.json').write_text(json.dumps(atomic,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap!=2000:p.error('fixed pursuit validation uses 2000 image bootstrap draws')
    analyze(a.batch,a.out,a.bootstrap)


if __name__=='__main__':main()
