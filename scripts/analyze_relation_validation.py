"""Independent fixed-roster relation study audit; never picks a new Main."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.evaluate_relation_validation import VERSION,METHODS,METRICS,read,sha256,strict_complete,ANCHORS
from scripts.report_independent_fit_results import image_bootstrap,table
from scripts.analyze_results import WeightedRanking
from uvarfs.relation_selection import RANDOM_SEEDS,SETTINGS

def analyze(batch,out,draws=2000):
    project=Path(__file__).resolve().parents[1];batch=Path(batch).resolve();out=Path(out).resolve()
    if not batch.is_relative_to((project/'results').resolve()) or not out.is_relative_to((project/'reports').resolve()) or out.exists() or draws!=2000:raise ValueError('fresh project-local complete audit required')
    startup=read(batch/'startup.json');frames=[];oldframes=[];differences=[];intervals=[];costs=[];normal=[];tracked={};blocks=[];configs=set()
    for seed in [42,123,3407]:
        for di,dataset in enumerate(['brain','liver','resc','oct2017','xray','camelyon16']):
            path=batch/'results'/f'seed{seed}'/dataset;blocks.append(strict_complete(path))
            meta=read(path/'run_metadata.json')
            for key,target in [('head','head'),('code_fingerprint','code_fingerprint'),('worker_sha256','worker_sha256'),('settings_sha256','config_sha256')]:
                if meta[key]!=startup[target]:raise ValueError('mixed relation source')
            cfg=dict(meta['source_config']);cfg.pop('seed');cfg.pop('results_dir');configs.add(json.dumps(cfg,sort_keys=True))
            parent=(project/meta['parent_relative']).resolve()
            if not parent.is_relative_to((project/'results').resolve()):raise ValueError('parent outside project')
            fresh=pd.read_csv(path/'metrics.csv');old=pd.read_csv(parent/'metrics.csv')
            pred=pd.read_csv(path/'image_predictions.csv',float_precision='round_trip');oldpred=pd.read_csv(parent/'image_predictions.csv',float_precision='round_trip')
            for key in ['image_path','label','mask_path']:
                if not pred[key].fillna('').equals(oldpred[key].fillna('')):raise ValueError('paired historical roster differs')
            values=fresh.set_index('method');groups={n:[n] for n in METHODS if not n.startswith(('random_weighted_','random_uniform_','random_beta_zero_','randomk_raw_'))}
            for label,stem in [('Random weighted mean','random_weighted'),('Random uniform mean','random_uniform'),('Random beta0 mean','random_beta_zero'),('Random-K layer mean','randomk_raw')]:
                groups[label]=[f'{stem}_seed{s}' for s in RANDOM_SEEDS]
            for metric in METRICS:
                if metric not in values:values[metric]=np.nan
            for method in METHODS:
                auc,ap=WeightedRanking(pred.label,pred[method]).metrics(np.ones(len(pred)))
                if not np.allclose([auc[0],ap[0]],values.loc[method,METRICS[:2]].to_numpy(dtype=float),rtol=1e-10,atol=1e-10):raise ValueError('independent image point estimates differ')
            target=values.loc['relation_sparse',METRICS].astype(float)
            for name,members in groups.items():
                if name!='relation_sparse':differences.append(dict(fit_seed=seed,dataset=dataset,comparison=name,**(target-values.loc[members,METRICS].astype(float).mean()).to_dict()))
            intervals.extend(dict(fit_seed=seed,dataset=dataset,**r) for r in image_bootstrap(pred,groups,'relation_sparse',draws,seed+1000003*di))
            frames.append(fresh);oldframes.append(old);costs.append(dict(fit_seed=seed,dataset=dataset,**read(path/'costs.json')))
            normal.append(pd.read_csv(path/'normal-full-audit/normal_errors.csv').assign(fit_seed=seed,dataset=dataset))
            for rel in ['metrics.csv','image_predictions.csv','selection/representation-specs.json','selection/nuisance-fit.json','selection/beta_zero-fit.json','run_metadata.json','normal-full-audit/image_scores.csv']:
                tracked[(path/rel).relative_to(project).as_posix()]=sha256(path/rel)
            print(f'[relation-audit] seed{seed}/{dataset}: 41 representations, eight exact anchors, fixed graph and roles verified',flush=True)
    if len(configs)!=1:raise ValueError('mixed base protocol')
    for rel,h in tracked.items():
        if sha256(project/rel)!=h:raise ValueError('analysis source changed')
    fresh=pd.concat(frames,ignore_index=True);delta=pd.DataFrame(differences)
    macro=delta.groupby(['fit_seed','comparison'])[METRICS].mean().reset_index()
    out.mkdir(parents=True)
    outputs={'new_evaluation_metrics':fresh,'retained_mechanism_metrics':pd.concat(oldframes,ignore_index=True),
        'paired_differences_by_fit':delta,'paired_macro_by_fit':macro,'conditional_image_intervals':pd.DataFrame(intervals),
        'new_costs':pd.DataFrame(costs),'isolated_full_normal_audit':pd.concat(normal,ignore_index=True)}
    for name,frame in outputs.items():frame.to_csv(out/(name+'.csv'),index=False)
    macro.groupby('comparison')[METRICS].agg(['mean','std','count']).to_csv(out/'macro_delta_mean_sample_sd.csv')
    delta.groupby(['dataset','comparison'])[METRICS].agg(['mean','std','count']).to_csv(out/'dataset_delta_mean_sample_sd.csv')
    fresh.groupby(['fit_seed','method'])[METRICS].mean().groupby('method').agg(['mean','std','count']).to_csv(out/'absolute_macro_mean_sample_sd.csv')
    random=fresh[fresh.method.str.startswith(('random_weighted_','random_uniform_','random_beta_zero_','randomk_raw_'))].copy()
    random['family']=random.method.str.replace(r'_seed\d+$','',regex=True)
    random.groupby(['fit_seed','dataset','family'])[METRICS].agg(['mean','std','count']).to_csv(out/'five_random_within_fit_mean_sd.csv')
    display=[]
    for name,part in macro.groupby('comparison'):
        display.append(dict(comparison=name,**{m:f'{100*part[m].mean():+.4f} ± {100*part[m].std(ddof=1):.4f}' if part[m].notna().any() else '无mask' for m in METRICS}))
    primary=macro[macro.comparison.isin(['asls_raw','Random weighted mean'])]
    directional=bool(primary.groupby('comparison')[['image_auroc','pixel_auprc']].mean().ge(.005).all().all())
    lines=['# 正常关系稀疏选择：固定全六三个来源fit', '',
        '唯一主比较relation_sparse。738表示评价/18配对区组，含重复控制；28项/区组为新拟合或同支持uniform表示，13项继承表示。原机制630行及其历史控制保留来源身份。',
        'ASLS输入/PCA/缓存复用，不是额外独立抽样fit或新患者；五Random子集不当Main重复。',
        '下表单位百分点，均值±三个来源fit样本SD；图像六集等权，定位三mask集等权。','',table(pd.DataFrame(display),['comparison',*METRICS]),'',
        f'预设探索性筛选：对Raw与同目标Random weighted，Image AUROC及Pixel AUPRC均值均提升至少0.5pp：{directional}。这不是临床效应阈值、显著性判定或确认成功。',
        '主要比较仅Raw与Random weighted，其余解释控制完整报告，不按最好分支更换主对象。所有图像bootstrap区间为条件性描述、未多比较校正，不据此作显著性/SOTA声明。',
        '完整网格正常audit排除64 query的memory且不用于选择；这些正常图像历史已暴露。图像bootstrap不是患者/跨fitCI，无完整像素地图重采样。',
        '历史test已多轮用于研发。必须报告逐集退化与实际新增/继承开销；外部完整方法和独立患者/新来源确认尚未由本轮完成。',
        '正常关系可能编码身份、设备或背景；loss下降/更稀疏/工程通过均不证明异常性能。注册Main保持v22，不自动升级。']
    (out/'assessment.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    proof=dict(evaluation_version=VERSION,blocks=blocks,new_evaluation_rows=len(fresh),retained_mechanism_rows=630,
        bootstrap=draws,settings=SETTINGS,primary_object='relation_sparse',registered_main_replaced=False,
        new_independent_normal_sampling_claimed=False,source_sha256=tracked,eight_anchors_per_block_exact=True,
        exploratory_screen_passed=directional,research_goal_achieved=False)
    (out/'audit-provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args();analyze(a.batch,a.out,a.bootstrap)
