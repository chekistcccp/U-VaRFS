#!/usr/bin/env python3
"""Audit all 18 paired tail evaluations and report every retained control."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.evaluate_normal_tail import VERSION,METRICS,METHODS,read,sha256,strict_complete
from scripts.report_independent_fit_results import families,image_bootstrap,table
from scripts.analyze_results import WeightedRanking


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
    startup=read(batch/'startup.json');source_frames=[];new_frames=[];differences=[];intervals=[];tracked={}
    datasets=['brain','liver','resc','oct2017','xray','camelyon16'];settings=set();source_codes=set();blocks=[]
    for seed in [42,123,3407]:
        for di,dataset in enumerate(datasets):
            path=batch/'results'/f'seed{seed}'/dataset;block=strict_complete(path)
            if block['fit_seed']!=seed or block['dataset']!=dataset:raise ValueError('fit/dataset identity differs')
            meta=read(path/'run_metadata.json')
            if meta['evaluation_git_head']!=startup['head'] or meta['evaluation_code_fingerprint']!=startup['code_fingerprint'] or meta['worker_script_sha256']!=startup['worker_sha256']:
                raise ValueError('mixed evaluation source identities')
            cfg=dict(meta['source_config']);cfg.pop('seed');cfg.pop('results_dir')
            settings.add(json.dumps(cfg,sort_keys=True));source_codes.add(meta['source_code_fingerprint'])
            for relative in ['metrics.csv','controls/metrics.csv','image_predictions.csv','controls/image_predictions.csv','run_metadata.json']:
                tracked[str((path/relative).relative_to(project))]=sha256(path/relative)
            fresh=pd.read_csv(path/'metrics.csv');old=pd.read_csv(path/'controls/metrics.csv')
            for metric in METRICS:
                if metric not in fresh:fresh[metric]=np.nan
                if metric not in old:old[metric]=np.nan
            new_frames.append(fresh);source_frames.append(old.assign(fit_seed=seed))
            f=fresh.set_index('method');o=old.set_index('method')
            groups=families(set(old.method),'',[42,123,3407,2026,2027])
            comps={name:f.loc[name,METRICS].astype(float) for name in sorted(METHODS-{'nuisance_tail'})}
            comps.update({'source_'+name:o.loc[names,METRICS].mean() for name,names in groups.items()})
            for name,values in comps.items():
                differences.append({'fit_seed':seed,'dataset':dataset,'comparison':name,
                                    **(f.loc['nuisance_tail',METRICS].astype(float)-values).to_dict()})
            old_groups={'source_'+name:names for name,names in groups.items()}
            bootgroups={name:[name] for name in sorted(METHODS)}|old_groups
            needed={name for names in bootgroups.values() for name in names}
            a=pd.read_csv(path/'image_predictions.csv',float_precision='round_trip')
            b=pd.read_csv(path/'controls/image_predictions.csv',usecols=lambda c:c in needed,float_precision='round_trip')
            predictions=pd.concat([a,b],axis=1)
            for name in needed:
                values=f.loc[name] if name in f.index else o.loc[name]
                auc,ap=WeightedRanking(predictions.label,predictions[name]).metrics(np.ones(len(predictions)))
                if not np.allclose([auc[0],ap[0]],values[METRICS[:2]].to_numpy(dtype=float),atol=1e-10,rtol=1e-10):
                    raise ValueError('independent image point estimates differ')
            for row in image_bootstrap(predictions,bootgroups,'nuisance_tail',draws,seed+1000003*di):
                intervals.append({'fit_seed':seed,'dataset':dataset,**row})
            blocks.append(block);print(f'[tail-audit] seed{seed}/{dataset}: six new evaluations and all 352 controls verified',flush=True)
    if len(settings)!=1 or len(source_codes)!=1:raise ValueError('source fits must share all settings except seed/results directory')
    for rel,digest in tracked.items():
        if sha256(project/rel)!=digest:raise ValueError('analysis inputs changed')
    new=pd.concat(new_frames,ignore_index=True);old=pd.concat(source_frames,ignore_index=True)
    delta=pd.DataFrame(differences);macro=delta.groupby(['fit_seed','comparison'])[METRICS].mean().reset_index()
    out.mkdir(parents=True)
    for name,frame in [('new_evaluation_metrics',new),('retained_352_control_metrics',old),
                       ('paired_differences_by_fit',delta),('paired_macro_by_fit',macro),
                       ('conditional_image_intervals',pd.DataFrame(intervals))]:frame.to_csv(out/(name+'.csv'),index=False)
    delta.groupby(['dataset','comparison'])[METRICS].agg(['mean','std','count']).to_csv(out/'dataset_delta_mean_sample_sd.csv')
    macro.groupby('comparison')[METRICS].agg(['mean','std','count']).to_csv(out/'macro_delta_mean_sample_sd.csv')
    display=summarize(macro,['comparison'])
    lines=['# 正常图像尾部保真：全六、三个独立来源fit的配对验证','',
           '六个新评价表示/区组，18区组共108行；旧352对照/区组共6336行单独保留，未重拟合。',
           '三个来源fit/memory seeds42/123/3407来自冻结v22，是同一组来源fit的配对延续，不冒充额外三次新拟合或未接触测试。',
           '图像全六等权、定位仅Brain/Liver/RESC等权。下表为nuisance_tail减对照的百分点均值±跨fit样本SD；五Random先在fit内平均。','',
           table(display,['comparison',*METRICS]),'',
           '每项图像bootstrap2000次，条件于单个拟合；不是患者/跨fit置信区间，多比较未校正。没有完整像素预测重采样。',
           '原mean与β0的逐图分数、image/pixel点估计及正常memory抽样严格复现，才允许引用旧控制。全部原始指标与预测保留；不会按test选择分支、参数或删除不利结果。',
           '每图16个正常采样patch的Top-1%是最大值，正常64图留出曾用于ASLS；one-SE是选择启发式。只有真实检测/定位比较支持收益时才能讨论目标是否得到支持。',
           '此分支不替换注册Main；正常CVaR、压缩或工程提速不代表成功。完整成本在每区组costs.json，历史拟合成本在controls/metrics.csv。']
    (out/'assessment.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic={'evaluation_version':VERSION,'blocks':blocks,'new_evaluations':len(new),'retained_control_rows':len(old),
            'independent_source_fit_seeds':[42,123,3407],'new_fit_repeats_claimed':False,'bootstrap':draws,
            'source_sha256':tracked,'all_used_inputs_unchanged':True,'main_replaced':False}
    (out/'audit-provenance.json').write_text(json.dumps(atomic,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap!=2000:p.error('fixed paired validation uses 2000 image bootstrap draws')
    analyze(a.batch,a.out,a.bootstrap)


if __name__=='__main__':main()
