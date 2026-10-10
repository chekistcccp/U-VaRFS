#!/usr/bin/env python3
"""Read-only BMAD audit/report; --fit-only explicitly handles partial returns.

No layer, lambda, representation budget or detector setting is selected here.
The bootstrap compares the mean AUC of random seeds, not an ensemble score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uvarfs.representation import normalization_modes, normalization_prefix
from uvarfs.objectives import COSINE, QUADRATIC, PERFORMANCE, experiment_branches
from uvarfs.selection_diagnostics import cosine_feasibility_diagnostics

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


def paired_bootstrap(frame, comparisons, draws=2000, seed=42, batch_size=32, target='main'):
    y=frame['label'].to_numpy(dtype=np.int64)
    normal=np.flatnonzero(y==0); abnormal=np.flatnonzero(y==1)
    if not len(normal) or not len(abnormal):
        raise ValueError('paired AUROC bootstrap needs both classes')
    names=sorted({target}|{name for group in comparisons.values() for name in group})
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
            differences[family].extend((aucs[target]-baseline).tolist())
    return {family:np.percentile(values,[2.5,97.5]).tolist() for family,values in differences.items()}


def markdown_table(frame, columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(str(row[column]) for column in columns)+' |')
    return '\n'.join(lines)


def audit_sparse_path(result, cfg):
    """Verify normal-only budget guards, independently of anomaly metrics."""
    if result.get('objective_definition')==PERFORMANCE:
        return audit_performance_path(result,cfg)
    if result.get('objective_definition')==COSINE:
        return audit_cosine_path(result,cfg)
    path=result['lambda_path']
    dimension=len(result['weights'])
    minimum=min(int(cfg['min_features']),dimension)
    maximum=min(int(cfg['max_features']),dimension)
    if [p['lambda'] for p in path]!=[float(x) for x in cfg['lambda_grid']]:
        raise ValueError('U-VaRFS selection path differs from the configured lambdas')
    for point in path:
        if not minimum<=point['retained_features']<=maximum:
            raise ValueError('U-VaRFS path exceeds its feature budget')
        expected=point['geometry_error']<=cfg['geometry_tolerance'] and point['nonzero_features']>=minimum
        if bool(point['feasible'])!=expected:
            raise ValueError('U-VaRFS path feasibility differs from the actual budgeted geometry')
    feasible=[p for p in path if p['feasible']]
    selected=min(feasible,key=lambda p:(p['retained_features'],p['geometry_error'])) if feasible else min(path,key=lambda p:p['geometry_error'])
    if selected['lambda']!=result['lambda']:
        raise ValueError('U-VaRFS selected lambda violates its normal-only selection rule')
    for key in ['legacy_top_weights','prune_refit','exchange_refit']:
        if key not in result:
            continue
        old=result[key]['lambda_path']
        if len(old)!=len(path):
            raise ValueError('U-VaRFS control has a different lambda path')
        for actual,reference in zip(path,old):
            if actual['lambda']!=reference['lambda'] or actual['budgeted_objective']>reference['budgeted_objective']+1e-7 or actual['geometry_error']>reference['geometry_error']+1e-7:
                raise ValueError('U-VaRFS refinement violates its per-lambda objective/geometry guard')
    audit=result.get('budget_geometry_audit')
    if audit is not None:
        mass=np.asarray(audit['coordinate_reference_alignment'],dtype=np.float64)
        if (audit['selection_candidate'] is not False or
            audit['scope']!='all_supports_with_at_most_max_features_and_box_weights' or
            len(mass)!=dimension or not np.isfinite(mass).all() or (mass<0).any() or
            audit['max_features']!=maximum or audit['geometry_tolerance']!=cfg['geometry_tolerance'] or
            audit['numerical_reporting_margin']!=1e-6):
            raise ValueError('global budget geometry audit has an incorrect scope')
        zero=audit['zero_reference_gram']
        if not np.isclose(mass.sum(),0 if zero else 1,atol=1e-10):
            raise ValueError('global budget geometry mass is not normalized')
        cumulative=np.cumsum(np.sort(mass)[::-1])
        floor=0 if zero else max(0,1-cumulative[maximum-1])
        necessary=0 if zero else min(dimension,int(np.searchsorted(cumulative,1-cfg['geometry_tolerance']))+1)
        ruled_out=floor>cfg['geometry_tolerance']+audit['numerical_reporting_margin']
        if (not np.isclose(floor,audit['geometry_error_lower_bound'],atol=1e-10) or
            necessary!=audit['necessary_features_lower_bound'] or
            ruled_out!=audit['budget_ruled_out_at_working_precision'] or
            any(p['geometry_error']+1e-6<floor for p in path)):
            raise ValueError('global budget geometry lower bound or feasibility is inconsistent')
    sources={}
    for point in path:
        key=point.get('budget_source','top_weights'); sources[key]=sources.get(key,0)+1
    return {'points':len(path),'budget_sources':sources,'selected_budget_source':selected.get('budget_source','top_weights'),
            'selected_continuous_geometry_error':selected['continuous_geometry_error'],
            'selected_nonzero_features':selected['nonzero_features'],
            'selection_and_guards_verified':True}


def audit_cosine_refit(refit,cfg):
    """Validate same-objective SQP guards without mistaking success for a certificate."""
    d=refit['polish']; maximum=cfg.get('cosine_polish_max_iter',200)
    numeric=['before','after','residual_before','residual_after','seconds']
    if (d['method']!='slsqp_fixed_support' or not np.isfinite([d[k] for k in numeric]).all() or
        any(d[k]<0 for k in numeric) or not 0<=d['iterations']<=maximum or
        d['objective_evaluations']<0 or not 0<=refit['projected_refit_iterations']<=cfg.get('cosine_refit_max_iter',cfg['max_iter']) or
        refit['solver_iterations']!=refit['projected_refit_iterations']+d['iterations']):
        raise ValueError('cosine polish diagnostics/budget are invalid')
    if (d['accepted'] and (not d['attempted'] or d['after']>=d['before'] or d['termination']!='strict_objective_descent') or
        not d['accepted'] and d['after']!=d['before'] or
        not d['attempted'] and (d['iterations']!=0 or d['objective_evaluations']!=0 or d['optimizer_success'] is not None) or
        d['attempted'] and (not isinstance(d['optimizer_success'],bool) or d['optimizer_status'] is None or maximum==0)):
        raise ValueError('cosine polish acceptance is not strict same-objective descent')
    if (not np.isclose(refit['fixed_support_first_order_residual'],d['residual_after'],atol=1e-12) or
        refit['fixed_support_converged']!=(d['residual_after']<=cfg['tol'])):
        raise ValueError('cosine polish convergence must use actual residual')
    if any(not np.isfinite([step['before'],step['after']]).all() or step['after']>=step['before']
           for step in refit.get('descent_history',[])):
        raise ValueError('cosine polish refit history is not descent')
    return d


def audit_cosine_pruning(result,cfg):
    """Validate actual joint proposal guards and all finite-deletion work."""
    manifest=result['solver_manifest']; strategy=cfg.get('cosine_pruning_strategy','gradient_direction')
    if (manifest['pruning_strategy']!=strategy or strategy not in {'gradient_direction','exact_objective_deletion'} or
        manifest['pruning_scope']!='one_coordinate_finite_deletion_proposal_then_joint_objective_guard' or
        manifest['pruning_acceptance']!='strict_same_objective_improvement_over_legacy_proposal'):
        raise ValueError('cosine pruning manifest differs from preset constraints')
    pool=result['candidate_pool']; events=result['pruning_events']; dimension=len(result['weights'])
    used=[]
    for index,e in enumerate(events):
        parent=e['parent_candidate_id']
        if not 0<=parent<len(pool) or e['event_id']!=index:
            raise ValueError('cosine pruning parent/event is invalid')
        before=pool[parent]
        if (e['strategy']!=strategy or e['source']!=before['source'] or e['source'] not in ['original_gram','normal_energy'] or
            e['source_features']!=before['retained_features'] or not 1<=e['target_features']<e['source_features'] or
            e['target_features'] not in manifest['support_sizes'] or
            not np.array_equal(e['active_before'],before['active']) or
            not np.isclose(e['base_objective_before'],before['base_objective'],atol=1e-9,rtol=1e-8) or
            not np.isclose(e['geometry_before'],before['geometry_error'],atol=1e-9,rtol=1e-8) or
            not np.isfinite([e['seconds'],e['first_order_score_span'],e['fixed_support_first_order_residual']]).all() or
            min(e['seconds'],e['first_order_score_span'],e['fixed_support_first_order_residual'],e['objective_evaluations'])<0):
            raise ValueError('cosine pruning source/cost disagrees')
        deletions=e['finite_deletions']
        if strategy=='exact_objective_deletion':
            if [d['removed_feature'] for d in deletions]!=list(e['active_before']):
                raise ValueError('cosine finite deletions omit active coordinates')
        elif deletions or e['exact_proposal'] is not None:
            raise ValueError('legacy cosine pruning unexpectedly used finite deletions')
        for d in deletions:
            if d['remaining_features']!=e['source_features']-1:
                raise ValueError('cosine deletion changes the wrong support size')
            if d['valid']:
                if (not np.isfinite([d['base_objective'],d['loss_delta'],d['geometry_error']]).all() or
                    min(d['base_objective'],d['geometry_error'])<0 or
                    not np.isclose(d['loss_delta'],d['base_objective']-e['base_objective_before'],atol=1e-9,rtol=1e-8)):
                    raise ValueError('cosine finite deletion loss disagrees')
            elif any(d[k] is not None for k in ['base_objective','loss_delta','geometry_error']):
                raise ValueError('invalid zero-row deletion has numeric loss')
        old=e['legacy_proposal']; exact=e['exact_proposal']
        for proposal in [old,exact]:
            if proposal is not None:
                active=np.asarray(proposal['active'])
                if (len(active)!=e['target_features'] or len(set(active.tolist()))!=len(active) or
                    (active<0).any() or (active>=dimension).any() or
                    not np.isfinite([proposal['base_objective'],proposal['geometry_error']]).all() or
                    min(proposal['base_objective'],proposal['geometry_error'])<0):
                    raise ValueError('cosine joint pruning proposal is invalid')
        accepted=exact is not None and (old is None or exact['base_objective']<old['base_objective'])
        chosen=exact if accepted else old
        name='exact_objective_deletion' if accepted else 'gradient_direction' if old is not None else 'invalid'
        if (e['exact_proposal_accepted']!=accepted or e['selected_proposal']!=name or
            e['selected_initial_objective']!=(None if chosen is None else chosen['base_objective']) or
            e['selected_initial_geometry']!=(None if chosen is None else chosen['geometry_error'])):
            raise ValueError('cosine pruning violates its joint objective guard')
    for c in pool:
        event=c.get('pruning_event_id')
        if event is None: continue
        if not 0<=event<len(events) or event in used:
            raise ValueError('cosine pruning candidate/event link is invalid')
        used.append(event);e=events[event]
        if (e['parent_candidate_id']>=c['candidate_id'] or e['source']!=c['source'] or
            e['target_features']!=c['retained_features'] or e['selected_initial_objective'] is None or
            c['base_objective']>e['selected_initial_objective']+1e-8):
            raise ValueError('cosine pruned candidate increases its guarded initial objective')
    if set(used)!={i for i,e in enumerate(events) if e['selected_proposal']!='invalid'}:
        raise ValueError('cosine pruning events/candidate links are incomplete')
    expected={'events':len(events),'exact_proposals_accepted':sum(e['exact_proposal_accepted'] for e in events),
              'objective_evaluations':sum(e['objective_evaluations'] for e in events),'seconds':sum(e['seconds'] for e in events)}
    if any((not np.isclose(result['pruning_summary'][k],v,atol=1e-10,rtol=1e-12) if k=='seconds' else result['pruning_summary'][k]!=v) for k,v in expected.items()):
        raise ValueError('cosine pruning totals omit scoring/proposal work')


def audit_cosine_path(result,cfg):
    """Check the finite-pool/new-metric selection; no old convex certificate."""
    if (result.get('geometry_metric')!='actual_cosine_gram_relative_frobenius' or
        result.get('objective_certificate_scope')!='fixed_support_first_order_only_nonconvex' or
        result.get('global_optimality_claimed') is not False or
        result.get('lambda_optimization_scope')!='finite_generated_candidate_pool' or
        result.get('box_optimality_gap') is not None or result.get('budget_geometry_audit') is not None):
        raise ValueError('cosine objective has a wrong metric or certificate scope')
    dimension=len(result['weights']); minimum=min(cfg['min_features'],dimension); maximum=min(cfg['max_features'],dimension)
    floor=cfg.get('cosine_weight_floor_mass',1e-4); tolerance=cfg['geometry_tolerance']
    pool=result['candidate_pool']; path=result['lambda_path']
    if not pool or [p['lambda'] for p in path]!=[float(x) for x in cfg['lambda_grid']]:
        raise ValueError('cosine candidate pool or lambda grid differs')
    manifest=result['solver_manifest']
    sizes=sorted(set(range(minimum,maximum+1,cfg.get('cosine_support_step',32)))|{maximum,min(maximum,max(minimum,result.get('original_gram_selected_dimension',maximum)))},reverse=True)
    # The exact seed dimension is recorded separately; every generated size must
    # be preset or the selected original-control size, never a test-tuned budget.
    if (manifest['weight_floor_mass']!=floor or manifest['support_sizes']!=sizes or
        manifest['initializations']!=['original_gram_selected','original_gram','normal_energy'] or
        manifest['refit_max_iter']!=cfg.get('cosine_refit_max_iter',cfg['max_iter']) or
        manifest['geometry_chunk']!=cfg.get('cosine_geometry_chunk',512) or
        manifest['line_search_steps']!=cfg.get('cosine_line_search_steps',20) or
        manifest['initial_step']!=cfg.get('cosine_initial_step',.05) or manifest['tolerance']!=cfg['tol'] or
        manifest['exchange_max_steps']!=cfg.get('cosine_exchange_max_steps',cfg.get('support_exchange_max_steps',8)) or
        manifest['exchange_candidates']!=cfg.get('cosine_exchange_candidates',8) or
        manifest['exchange_min_improvement']!=cfg.get('support_exchange_min_improvement',1e-8) or
        manifest['zero_row_rule']!='candidate invalid; never normalized with epsilon'):
        raise ValueError('cosine solver manifest differs from preset constraints')
    polished='polish_method' in manifest
    if any(k in cfg for k in ['cosine_polish_max_iter','cosine_polish_ftol']) and not polished:
        raise ValueError('cosine polish manifest is missing')
    if polished and (manifest['polish_method']!='slsqp_fixed_support' or
        manifest['polish_max_iter']!=cfg.get('cosine_polish_max_iter',200) or
        manifest['polish_ftol']!=cfg.get('cosine_polish_ftol',1e-12) or
        manifest['polish_acceptance']!='projected_endpoint_strict_same_objective_descent'):
        raise ValueError('cosine polish manifest differs from preset constraints')
    pruned='pruning_strategy' in manifest
    if 'cosine_pruning_strategy' in cfg and not pruned:
        raise ValueError('cosine pruning manifest is missing')
    if pruned:
        audit_cosine_pruning(result,cfg)
    polishes=[]; iterations=0
    for index,candidate in enumerate(pool):
        weights=np.asarray(candidate['weights'],dtype=float); active=np.asarray(candidate['active'],dtype=int)
        if (weights.shape!=(dimension,) or not np.isfinite(weights).all() or (weights<0).any() or
            candidate['candidate_id']!=index or not minimum<=len(active)<=maximum or
            len(set(active.tolist()))!=len(active) or (active<0).any() or (active>=dimension).any()):
            raise ValueError('cosine candidate support or weights are invalid')
        nonzero=np.flatnonzero(weights>0)
        if not np.array_equal(nonzero,active) or not np.isclose(weights.sum(),1,atol=1e-6) or (weights[active]<floor/len(active)-1e-9).any():
            raise ValueError('cosine candidate violates simplex/positive-support constraints')
        if (candidate['nonzero_features']!=len(active) or candidate['retained_features']!=len(active) or
            candidate['zero_norm_rows']!=0 or candidate['retained_features'] not in sizes or
            candidate['source'] not in manifest['initializations']):
            raise ValueError('cosine candidate dimension/source disagrees')
        numeric=['geometry_error','representation_term','variability_term','relative_variability',
                 'base_objective','fixed_support_first_order_residual','simplex_residual']
        if not np.isfinite([candidate[k] for k in numeric]).all() or any(candidate[k]<0 for k in numeric):
            raise ValueError('nonfinite or negative cosine diagnostics')
        if (not np.isclose(candidate['representation_term'],.5*candidate['geometry_error']**2,atol=1e-9) or
            not np.isclose(candidate['variability_term'],cfg['beta']*candidate['relative_variability'],atol=1e-9) or
            not np.isclose(candidate['base_objective'],candidate['representation_term']+candidate['variability_term'],atol=1e-9) or
            not np.isclose(candidate['effective_weight_dimension'],1/np.square(weights).sum(),atol=1e-5) or
            not np.isclose(candidate['simplex_residual'],abs(weights.sum()-1),atol=1e-10) or
            not np.isclose(candidate['minimum_relative_weight'],weights[active].min(),atol=1e-9) or
            candidate['fixed_support_converged']!=(candidate['fixed_support_first_order_residual']<=cfg['tol'])):
            raise ValueError('cosine candidate objective/residual fields disagree')
        for history in [candidate['refit_descent_history'],candidate['exchange_history']]:
            if any(not np.isfinite([p['before'],p['after']]).all() or p['after']>=p['before'] for p in history):
                raise ValueError('cosine accepted search step is not descent')
        if polished:
            refits=[candidate['initial_refit']]+[e['refit'] for e in candidate['exchange_history']]
            for refit in refits:
                polishes.append(audit_cosine_refit(refit,cfg))
                iterations+=refit['solver_iterations']
            final=refits[-1]
            if (not np.isclose(candidate['base_objective'],final['polish']['after'],atol=1e-9) or
                not np.isclose(candidate['fixed_support_first_order_residual'],final['polish']['residual_after'],atol=1e-10)):
                raise ValueError('cosine candidate disagrees with its final polished refit')
    if polished:
        expected={'attempts':sum(d['attempted'] for d in polishes),'accepted':sum(d['accepted'] for d in polishes),
                  'iterations':sum(d['iterations'] for d in polishes),
                  'objective_evaluations':sum(d['objective_evaluations'] for d in polishes),
                  'seconds':sum(d['seconds'] for d in polishes)}
        if result['solver_iterations']!=iterations or any((not np.isclose(result['polish_summary'][k],v,atol=1e-10,rtol=1e-12) if k=='seconds' else result['polish_summary'][k]!=v) for k,v in expected.items()):
            raise ValueError('cosine polish totals omit candidate/exchange work')
    for point in path:
        lam=point['lambda']
        best=min(pool,key=lambda c:(c['base_objective']+lam*c['retained_features'],c['retained_features'],c['geometry_error'],c['candidate_id']))
        if point['candidate_id']!=best['candidate_id']:
            raise ValueError('cosine lambda point is not the best generated full-objective candidate')
        for key in ['retained_features','nonzero_features','geometry_error','base_objective','representation_term','variability_term','effective_weight_dimension','fixed_support_first_order_residual']:
            if not np.isclose(point[key],best[key],atol=1e-10):
                raise ValueError('cosine lambda fields disagree with selected candidate')
        if (not np.isclose(point['sparsity_term'],lam*best['retained_features'],atol=1e-10) or
            not np.isclose(point['objective'],best['base_objective']+point['sparsity_term'],atol=1e-10) or
            point['feasible']!=(point['geometry_error']<=tolerance and point['nonzero_features']>=minimum)):
            raise ValueError('cosine lambda objective/feasibility is inconsistent')
    feasible=[p for p in path if p['feasible']]
    selected=min(feasible,key=lambda p:(p['retained_features'],p['geometry_error'])) if feasible else min(path,key=lambda p:p['geometry_error'])
    if result['lambda']!=selected['lambda'] or result['candidate_id']!=selected['candidate_id'] or result['feasible']!=selected['feasible']:
        raise ValueError('cosine selected lambda violates its normal-only rule')
    chosen=pool[selected['candidate_id']]
    if (result['selection_reason']!=('sparsest_feasible' if selected['feasible'] else 'minimum_geometry_error_fallback') or
        result['solver_converged']!=selected['fixed_support_converged'] or
        not np.isclose(result['geometry_error'],selected['geometry_error'],atol=1e-10) or
        not np.isclose(result['objective'],selected['objective'],atol=1e-10) or
        result['lambda_at_grid_boundary']!=(result['lambda'] in (min(cfg['lambda_grid']),max(cfg['lambda_grid'])))):
        raise ValueError('cosine selected result fields disagree with its path')
    for key in ['weights','budgeted_weights']:
        np.testing.assert_array_equal(result[key],chosen['weights'])
    np.testing.assert_array_equal(result['active'],chosen['active'])
    np.testing.assert_allclose(np.square(result['scales']),np.asarray(chosen['weights'])[chosen['active']],rtol=1e-6,atol=1e-9)
    return {'points':len(path),'candidate_pool_size':len(pool),'selected_nonzero_features':selected['nonzero_features'],
            'objective_definition':COSINE,'certificate_scope':result['objective_certificate_scope'],
            'selection_and_guards_verified':True,'polish_guards_verified':polished,'pruning_guards_verified':pruned,'global_optimality_claimed':False,
            **cosine_feasibility_diagnostics(result,cfg)}


def audit_refitted_exchange_search(result,cfg):
    """Audit accepted support transitions and every attempted refit cost."""
    search=result['exchange_search'];manifest=result['solver_manifest'];pool=result['candidate_pool']
    maximum=int(cfg.get('performance_exchange_max_steps',8));proposals=int(cfg.get('performance_exchange_candidates',8))
    gain=float(cfg.get('support_exchange_min_improvement',1e-8));tolerance=cfg['geometry_tolerance']
    count=min(cfg['max_features'],len(result['weights']));cap=cfg.get('performance_weight_cap_factor',4.)
    floor=cfg.get('cosine_weight_floor_mass',1e-4)
    if (search['strategy']!='refit_before_accept' or result['sparsity_strategy']!='fixed_budget_refitted_support_exchange' or
        manifest.get('performance_exchange_strategy')!='refit_before_accept' or
        manifest.get('performance_exchange_max_steps')!=maximum or manifest.get('performance_exchange_candidates')!=proposals or
        not 0<search['reference_candidate_count']<=len(pool) or
        search['reference_selected_candidate_id']>=search['reference_candidate_count']):
        raise ValueError('refitted exchange manifest/reference scope differs')
    references=pool[:search['reference_candidate_count']]
    eligible_references=[c for c in references if c['geometry_error']<=tolerance]
    reference_chosen=min(eligible_references,key=lambda c:(c['base_objective'],c['geometry_error'],c['candidate_id'])) if eligible_references else min(references,key=lambda c:(c['geometry_error'],c['base_objective'],c['candidate_id']))
    if reference_chosen['candidate_id']!=search['reference_selected_candidate_id']:
        raise ValueError('refitted exchange reference selection differs')
    parents=[c['candidate_id'] for c in references if c['candidate_type']=='optimized']
    if [t['parent_candidate_id'] for t in search['trajectories']]!=parents:
        raise ValueError('refitted exchange parent candidates differ')
    polishes=[];iterations=0;refits=0;accepted=0;uphill=0;linked=[]
    for trace in search['trajectories']:
        parent=pool[trace['parent_candidate_id']];active=np.asarray(parent['active'])
        value=parent['base_objective'];geometry=parent['geometry_error'];used=0
        if not np.isclose(trace['initial_objective'],value,atol=1e-8) or not np.isclose(trace['initial_geometry'],geometry,atol=1e-7):
            raise ValueError('refitted exchange initial state differs')
        attempts=trace['attempts'];steps=trace['steps']
        if len(steps)>maximum or len(attempts)>maximum*proposals:
            raise ValueError('refitted exchange exceeds configured solve budget')
        for index,a in enumerate(attempts):
            if (a['attempt_id']!=index or not 0<=a['step']<maximum or
                not 0<=a['proposal_rank']<proposals or a['removed']==a['added']):
                raise ValueError('refitted exchange proposal identity differs')
            if a['initial_valid']:
                polishes.append(audit_cosine_refit(a['refit'],cfg));refits+=1
                used+=a['refit']['solver_iterations']
                if not np.isfinite([a['before_refit'],a['after_refit'],a['geometry_after']]).all():
                    raise ValueError('refitted exchange has nonfinite objective')
                eligible=a['after_refit']<a['before']-gain and (a['geometry_before']>tolerance or a['geometry_after']<=tolerance)
                if a['eligible']!=eligible or a['after_refit']>a['before_refit']+1e-7:
                    raise ValueError('refitted exchange endpoint guard differs')
                history=a['refit']['descent_history']
                endpoint=history[-1]['after'] if history else a['refit']['polish']['after']
                if not np.isclose(endpoint,a['after_refit'],atol=1e-7):
                    raise ValueError('refitted exchange endpoint differs from refit')
                uphill+=a['before_refit']>=a['before']-gain
            elif a['refit'] is not None or a['accepted'] or a['eligible']:
                raise ValueError('invalid zero-row exchange was refitted or accepted')
        for index,step in enumerate(steps):
            group=[a for a in attempts if a['step']==index]
            eligible=[a for a in group if a['eligible']]
            chosen=min(eligible,key=lambda a:(a['after_refit'],a['proposal_rank'])) if eligible else None
            if (step['step']!=index or chosen is None or step['attempt_id']!=chosen['attempt_id'] or
                not chosen['accepted'] or chosen['removed'] not in active or chosen['added'] in active or
                not np.isclose(step['before'],value,atol=1e-8) or not np.isclose(step['geometry_before'],geometry,atol=1e-7)):
                raise ValueError('refitted exchange accepted transition differs')
            proposed=np.sort(np.where(active==chosen['removed'],chosen['added'],active))
            weights=np.asarray(step['relative_weights']);observed=np.asarray(step['active'])
            if (not np.array_equal(proposed,observed) or len(weights)!=count or not np.isfinite(weights).all() or
                not np.isclose(weights.sum(),1,atol=1e-6) or weights.min()<floor/count-1e-9 or weights.max()>min(1.,cap/count)+1e-7 or
                step['after']>=value-gain or geometry<=tolerance and step['geometry_after']>tolerance or
                not np.isclose(step['after'],chosen['after_refit'],atol=1e-8) or
                not np.isclose(step['geometry_after'],chosen['geometry_after'],atol=1e-7)):
                raise ValueError('refitted exchange lost fixed K/cap/geometry or strict descent')
            for a in group:
                if not np.isclose(a['before'],value,atol=1e-8) or not np.isclose(a['geometry_before'],geometry,atol=1e-7):
                    raise ValueError('refitted exchange proposals use different parent states')
            active=observed;value=step['after'];geometry=step['geometry_after'];accepted+=1
        terminal=[a for a in attempts if a['step']==len(steps)]
        if terminal and (any(a['eligible'] or a['accepted'] or not np.isclose(a['before'],value,atol=1e-8) or
                not np.isclose(a['geometry_before'],geometry,atol=1e-7) or a['removed'] not in active or a['added'] in active for a in terminal) or
                trace['termination']!='no_improving_refitted_proposal'):
            raise ValueError('refitted exchange terminal rejection differs')
        if any(a['step']>len(steps) for a in attempts):
            raise ValueError('refitted exchange attempts lack a preceding accepted transition')
        if sum(a['accepted'] for a in attempts)!=len(steps) or used!=trace['solver_iterations']:
            raise ValueError('refitted exchange omitted accepted steps or rejected refit costs')
        if trace['candidate_id'] is not None:
            linked.append(trace['candidate_id']);c=pool[trace['candidate_id']]
            if (not steps or c['candidate_type']!='refitted_exchange' or c['parent_candidate_id']!=parent['candidate_id'] or
                not np.array_equal(c['active'],active) or not np.allclose(np.asarray(c['weights'])[active],steps[-1]['relative_weights'],atol=1e-8)):
                raise ValueError('refitted exchange candidate linkage differs')
        elif steps:raise ValueError('accepted refitted exchange lacks its candidate')
        if (not np.isclose(trace['final_objective'],value,atol=1e-8) or not np.isclose(trace['final_geometry'],geometry,atol=1e-7) or
            trace['seconds']<0 or trace['objective_evaluations']<len(attempts)):
            raise ValueError('refitted exchange final state/cost differs')
        iterations+=used
    if (linked!=list(range(search['reference_candidate_count'],len(pool))) or search['solver_iterations']!=iterations or
        search['seconds']<0 or search['objective_evaluations']<sum(t['objective_evaluations'] for t in search['trajectories'])):
        raise ValueError('refitted exchange total candidate linkage/cost differs')
    if 'performance_objective_cache_mib' in manifest:
        cache=search.get('objective_cache')
        limit=int(float(cfg.get('performance_objective_cache_mib',128))*1024**2)
        if (cache is None or manifest['performance_objective_cache_mib']!=cfg.get('performance_objective_cache_mib',128) or
            cache['cache_limit_bytes']!=limit or cache['peak_cache_bytes']>limit or
            any(not isinstance(cache[k],int) or cache[k]<0 for k in ['cache_limit_bytes','peak_cache_bytes',
                'objective_requests','geometry_evaluations','geometry_reuses','identical_result_reuses']) or
            cache['objective_requests']!=search['objective_evaluations'] or
            cache['geometry_evaluations']+cache['geometry_reuses']!=cache['objective_requests'] or
            cache['identical_result_reuses']>cache['geometry_reuses']):
            raise ValueError('exact cosine cache memory/cost accounting differs')
    return polishes,iterations,{'support_refits_attempted':refits,'support_exchanges_accepted':accepted,
        'uphill_initial_proposals_refitted':uphill,'refitted_exchange_guards_verified':True}


def audit_performance_path(result,cfg):
    """Verify the fixed budget/cap and normal-only finite-pool selection."""
    dimension=len(result['weights']); count=min(int(cfg['max_features']),dimension)
    cap=float(cfg.get('performance_weight_cap_factor',4.)); floor=float(cfg.get('cosine_weight_floor_mass',1e-4))
    tolerance=float(cfg['geometry_tolerance']); manifest=result['solver_manifest']
    expected={'fixed_features':count,'weight_floor_mass':floor,'weight_cap_factor':cap,
        'initializations':['original_gram','normal_energy','original_gram_selected_expanded'],
        'candidate_types':['uniform_reference','optimized','refitted_exchange'] if 'exchange_search' in result else ['uniform_reference','optimized'],
        'selection_metric':'normal_feasibility_then_representation_plus_variability',
        'geometry_chunk':cfg.get('cosine_geometry_chunk',512),
        'refit_max_iter':cfg.get('cosine_refit_max_iter',cfg['max_iter']),
        'polish_max_iter':cfg.get('cosine_polish_max_iter',200),'polish_ftol':cfg.get('cosine_polish_ftol',1e-12),
        'exchange_max_steps':cfg.get('cosine_exchange_max_steps',8),'exchange_candidates':cfg.get('cosine_exchange_candidates',8),
        'exchange_min_improvement':cfg.get('support_exchange_min_improvement',1e-8),
        'line_search_steps':cfg.get('cosine_line_search_steps',20),'initial_step':cfg.get('cosine_initial_step',.05),
        'tolerance':cfg['tol'],'zero_row_rule':'candidate invalid; never normalized with epsilon',
        'stability_reference':'uniform full-input relative weights'}
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError('performance-first manifest differs from fixed constraints')
    if (result['lambda_optimization_scope']!='not_used_fixed_budget' or result['lambda']!=0. or
        result['sparsity_term']!=0. or result['lambda_at_grid_boundary'] is not False or
        result['objective_certificate_scope']!='fixed_support_first_order_only_nonconvex' or
        result['geometry_metric']!='actual_cosine_gram_relative_frobenius' or
        result['global_optimality_claimed'] is not False or result.get('box_optimality_gap') is not None or
        result.get('budget_geometry_audit') is not None or result['selection_policy']!='fixed_budget_feasible_objective'):
        raise ValueError('performance-first objective/selection scope is inconsistent')
    pool=result['candidate_pool']; polishes=[]; iterations=0
    if not pool: raise ValueError('empty performance-first candidate pool')
    for index,c in enumerate(pool):
        w=np.asarray(c['weights'],dtype=float);active=np.asarray(c['active'],dtype=int)
        if (w.shape!=(dimension,) or not np.isfinite(w).all() or (w<0).any() or
            c['candidate_id']!=index or c['retained_features']!=count or c['nonzero_features']!=count or
            len(active)!=count or not np.array_equal(np.flatnonzero(w>0),active) or
            not np.isclose(w.sum(),1,atol=1e-6) or w[active].min()<floor/count-1e-9 or
            w.max()>min(1.,cap/count)+1e-7 or c['zero_norm_rows']!=0 or
            c['source'] not in expected['initializations'] or c['candidate_type'] not in expected['candidate_types']):
            raise ValueError('performance-first support/capped weights violate constraints')
        if (not np.isfinite([c[k] for k in ['base_objective','geometry_error','representation_term','variability_term','fixed_support_first_order_residual']]).all() or
            not np.isclose(c['base_objective'],c['representation_term']+c['variability_term'],atol=1e-9) or
            not np.isclose(c['representation_term'],.5*c['geometry_error']**2,atol=1e-9) or
            not np.isclose(c['variability_term'],cfg['beta']*c['relative_variability'],atol=1e-8) or
            not np.isclose(c['effective_weight_dimension'],1/np.square(w).sum(),atol=1e-6) or
            not np.isclose(c['maximum_relative_weight'],w.max(),atol=1e-9) or
            c['effective_weight_dimension']+1e-5<count/min(cap,count) or
            c['fixed_support_converged']!=(c['fixed_support_first_order_residual']<=cfg['tol'])):
            raise ValueError('performance-first objective/weight diagnostics disagree')
        if c['candidate_type']=='uniform_reference':
            if c['initial_refit'] is not None or c['exchange_history'] or not np.allclose(w[active],1/count,atol=1e-8):
                raise ValueError('performance-first uniform reference was refitted')
        elif c['candidate_type']=='optimized':
            refits=[c['initial_refit']]+[e['refit'] for e in c['exchange_history']]
            for r in refits:
                polishes.append(audit_cosine_refit(r,cfg));iterations+=r['solver_iterations']
            for e in c['exchange_history']:
                if e['after']>=e['before'] or e['removed']==e['added']:
                    raise ValueError('performance-first exchange does not improve the normal objective')
        elif c['initial_refit'] is not None or c['exchange_history']:
            raise ValueError('refitted exchange candidate duplicates refit costs')
    search_audit={}
    if 'exchange_search' in result:
        search_polishes,used,search_audit=audit_refitted_exchange_search(result,cfg)
        polishes.extend(search_polishes);iterations+=used
    elif result['sparsity_strategy']!='fixed_budget_capped_simplex_support_search':
        raise ValueError('performance exchange strategy lacks its search diagnostics')
    eligible=[c for c in pool if c['geometry_error']<=tolerance]
    chosen=min(eligible,key=lambda c:(c['base_objective'],c['geometry_error'],c['candidate_id'])) if eligible else min(pool,key=lambda c:(c['geometry_error'],c['base_objective'],c['candidate_id']))
    if (result['candidate_id']!=chosen['candidate_id'] or result['feasible']!=bool(eligible) or
        result['selection_reason']!=('fixed_budget_best_feasible_objective' if eligible else 'minimum_geometry_error_fallback') or
        len(result['lambda_path'])!=1 or result['lambda_path'][0]['candidate_id']!=chosen['candidate_id'] or
        result['solver_iterations']!=iterations or result['weight_cap_factor']!=cap or
        not np.isclose(result['effective_dimension_lower_bound'],count/min(cap,count)) or
        result['solver_converged']!=chosen['fixed_support_converged']):
        raise ValueError('performance-first chosen point violates its normal-only rule')
    for key in ['active','weights','budgeted_weights']:
        np.testing.assert_array_equal(result[key],chosen['active'] if key=='active' else chosen['weights'])
    np.testing.assert_allclose(np.square(result['scales']),np.asarray(chosen['weights'])[chosen['active']],rtol=1e-6,atol=1e-9)
    for key in ['geometry_error','fixed_support_first_order_residual','effective_weight_dimension','maximum_relative_weight','base_objective']:
        if not np.isclose(result[key],chosen[key],atol=1e-10):
            raise ValueError('performance-first selected fields differ from candidate')
    for key in ['lambda','objective','sparsity_term','feasible','retained_features','geometry_error','candidate_id']:
        if result['lambda_path'][0][key]!=result[key]:
            raise ValueError('performance-first reporting point differs from selected result')
    if not np.isclose(result['objective'],chosen['base_objective'],atol=1e-10):
        raise ValueError('performance-first selected objective differs')
    totals={'attempts':sum(d['attempted'] for d in polishes),'accepted':sum(d['accepted'] for d in polishes),
        'iterations':sum(d['iterations'] for d in polishes),'objective_evaluations':sum(d['objective_evaluations'] for d in polishes),
        'seconds':sum(d['seconds'] for d in polishes)}
    if any(not np.isclose(result['polish_summary'][k],v,atol=1e-10) for k,v in totals.items()):
        raise ValueError('performance-first costs omit refit/polish work')
    return {'points':1,'candidate_pool_size':len(pool),'selected_nonzero_features':count,
            'objective_definition':PERFORMANCE,'certificate_scope':result['objective_certificate_scope'],
            'selection_and_guards_verified':True,'global_optimality_claimed':False,
            **search_audit,**cosine_feasibility_diagnostics(result,cfg)}


def audit_representation_protocol(folder, cfg, frame, primary_fit):
    """Paired input branches must share data and perturbations, not feature values."""
    if 'representation' not in cfg:
        return []  # Historical v9 and earlier protocols have raw input only.
    primary,modes=normalization_modes(cfg)
    manifest=json.loads((folder/'representation_manifest.json').read_text())
    observed=dict(zip(frame.method,frame.layer_normalization))
    if (not manifest['shared_fit_images_and_patches'] or not manifest['shared_backbone_forwards'] or
        manifest['primary_layer_normalization']!=primary or
        manifest['ablation_layer_normalization']!=(modes[1] if len(modes)>1 else None) or
        manifest['normalization_by_method']!=observed):
        raise ValueError('representation manifest and per-method input conventions disagree')
    if 'objective' in cfg.get('uvarfs',{}):
        if manifest.get('objective_branches')!=experiment_branches(cfg):
            raise ValueError('objective branch manifest differs')
        if 'objective_branch' in frame and manifest.get('objective_branch_by_method')!=dict(zip(frame.method,frame.objective_branch)):
            raise ValueError('objective per-method manifest differs from CSV')
    records=[]
    asls_by_input={}; fitted_by_input={}
    for branch in experiment_branches(cfg):
        mode=branch['layer_normalization']; prefix=branch['prefix']
        output=folder/branch['folder']
        fit=json.loads((output/'fit_manifest.json').read_text())
        if fit['layer_normalization']!=mode or fit['variability_feature_space']!=mode:
            raise ValueError('fit and variability use different feature input conventions')
        for key in ['normal_train_indices','normal_train_paths','fit_images','variability_images',
                    'patches_per_image','sampled_patches','perturbation_rng','backbone_raw_patch_norms']:
            if fit[key]!=primary_fit[key]:
                raise ValueError('representation ablation has a different normal data/perturbation budget')
        own=prefix+'main'
        if observed[own]!=mode:
            raise ValueError('main input convention does not match fit')
        a=json.loads((output/'asls.json').read_text())
        u=json.loads((output/'uvarfs_main.json').read_text())
        if u.get('objective_definition',QUADRATIC)!=branch['objective']:
            raise ValueError('main objective does not match its declared branch')
        normal=json.loads((output/'normal_geometry_audit.json').read_text())
        count=sum(batch['batch_images']*len(batch['patch_indices']) for batch in fit['sampled_patches'])
        if a['layer_normalization']!=mode or normal['layer_feature_normalization']!=mode or normal['selection_candidate'] is not False or normal['normal_geometry_rows']!=count:
            raise ValueError('normal geometry diagnostics use the wrong input convention or rows')
        if mode in asls_by_input and a!=asls_by_input[mode]:
            raise ValueError('objective ablation changed ASLS')
        asls_by_input[mode]=a
        fitted_by_input.setdefault(mode,{})[branch['objective']]=u
        own_rows=frame.set_index('method')
        if 'feature_dim' in own_rows and len(u['active'])!=own_rows.loc[own].feature_dim:
            raise ValueError('main CSV dimension disagrees with fitted support')
        for name in ['asls_pca','asls_selected_raw']+[f'asls_random_seed{s}' for s in cfg.get('random_baseline_seeds',[cfg['seed']])]:
            if 'feature_dim' in own_rows and prefix+name in own_rows.index and own_rows.loc[prefix+name].feature_dim!=len(u['active']):
                raise ValueError('PCA/Random feature dimension differs from its own objective branch')
        if 'asls_selected_raw' in cfg['methods']:
            control=json.loads((output/'selected_support_control.json').read_text())
            if (control['source_main']!=own or control['source_objective']!=branch['objective'] or
                control['layer_normalization']!=mode or control['layers']!=u['layers'] or
                control['active']!=u['active'] or control['feature_dim']!=len(u['active']) or
                control['kind']!='main_support_without_relative_weights' or
                control['representation_weighting']!='uniform_on_main_support' or
                control['extra_fit'] is not False or control['selection_candidate'] is not False):
                raise ValueError('selected-support control differs from its own Main')
            diagnostic=normal['methods']['asls_selected_raw']
            if (diagnostic['feature_dim']!=len(u['active']) or diagnostic['layers']!=u['layers'] or
                diagnostic['support_source_method']!=own or
                diagnostic['representation_weighting']!='uniform_on_main_support'):
                raise ValueError('selected-support geometry differs from its own Main')
            if 'support_source_method' in own_rows and (own_rows.loc[prefix+'asls_selected_raw'].support_source_method!=own or
                own_rows.loc[prefix+'asls_selected_raw'].representation_weighting!='uniform_on_main_support'):
                raise ValueError('selected-support CSV source/weighting differs')
        if 'asls_fixed_budget_uvarfs' in cfg['methods']:
            fixed=json.loads((output/'uvarfs_asls_fixed_budget.json').read_text())
            diag=normal['methods']['asls_fixed_budget_uvarfs']
            if (fixed['objective_definition']!=PERFORMANCE or fixed['layers']!=u['layers'] or
                fixed['layer_normalization']!=mode or fixed['sparsity_strategy']!='fixed_budget_capped_simplex_support_search' or
                diag['feature_dim']!=len(fixed['active']) or not np.isclose(fixed['geometry_error'],diag['vs_selected_input_cosine']['cosine_gram_relative_error'],atol=2e-6,rtol=1e-5) or
                'feature_dim' in own_rows and own_rows.loc[prefix+'asls_fixed_budget_uvarfs'].feature_dim!=len(fixed['active']) or
                'uvarfs_objective_definition' in own_rows and own_rows.loc[prefix+'asls_fixed_budget_uvarfs'].uvarfs_objective_definition!=PERFORMANCE):
                raise ValueError('v17 fixed-budget control differs from same-input protocol')
            audit_sparse_path(fixed,cfg['uvarfs'])
            if branch['objective']==PERFORMANCE and cfg['uvarfs'].get('performance_exchange_strategy')=='refit_before_accept':
                if ('exchange_search' not in u or u['candidate_pool'][:u['exchange_search']['reference_candidate_count']]!=fixed['candidate_pool']):
                    raise ValueError('Main failed to preserve the complete v17 reference pool')
        if 'asls_compression_uvarfs' in cfg['methods']:
            compression=json.loads((output/'uvarfs_asls_compression.json').read_text())
            diag=normal['methods']['asls_compression_uvarfs']
            if (compression['objective_definition']!=COSINE or compression['layers']!=u['layers'] or
                compression['layer_normalization']!=mode or diag['layers']!=u['layers'] or
                diag['feature_dim']!=len(compression['active']) or
                not np.isclose(compression['geometry_error'],diag['vs_selected_input_cosine']['cosine_gram_relative_error'],atol=2e-6,rtol=1e-5)):
                raise ValueError('compression control differs from same-input ASLS/cosine protocol')
            if 'feature_dim' in own_rows and own_rows.loc[prefix+'asls_compression_uvarfs'].feature_dim!=len(compression['active']):
                raise ValueError('compression control CSV dimension differs from support')
            if 'uvarfs_objective_definition' in own_rows and own_rows.loc[prefix+'asls_compression_uvarfs'].uvarfs_objective_definition!=COSINE:
                raise ValueError('compression control CSV objective differs')
            audit_sparse_path(compression,cfg['uvarfs'])
        if branch['objective'] in {COSINE,PERFORMANCE} and not np.isclose(u['geometry_error'],normal['methods']['main']['vs_selected_input_cosine']['cosine_gram_relative_error'],atol=2e-6,rtol=1e-5):
            raise ValueError('cosine selection geometry disagrees with actual detector representation')
        if branch['objective'] in {COSINE,PERFORMANCE} and 'uvarfs_feasibility_scope' in own_rows:
            expected=cosine_feasibility_diagnostics(u,cfg['uvarfs'])
            for key,value in expected.items():
                column='uvarfs_'+key
                if column not in own_rows:
                    raise ValueError('cosine CSV feasibility scope is incomplete')
                observed_value=own_rows.loc[own,column]
                matches=(pd.isna(observed_value) if value is None else
                         not pd.isna(observed_value) and (np.isclose(observed_value,value,atol=1e-12,rtol=1e-10)
                         if isinstance(value,float) else observed_value==value))
                if not matches:
                    raise ValueError('cosine CSV feasibility scope differs from generated pool/path')
        records.append({'method':own,'layer_normalization':mode,'layers':a['selected_layers'],
                        'objective_definition':branch['objective'],
                        'sparse_path_audit':audit_sparse_path(u,cfg['uvarfs']),
                        'budget_geometry_audit':u.get('budget_geometry_audit'),
                        'normal_cosine_geometry':normal})
    for objectives in fitted_by_input.values():
        new_objective=PERFORMANCE if PERFORMANCE in objectives else COSINE
        if new_objective in objectives and QUADRATIC in objectives:
            new,old=objectives[new_objective],objectives[QUADRATIC]
            if new['original_gram_selected_dimension']!=len(old['active']) or new['original_gram_selected_active']!=sorted(old['active']):
                raise ValueError('cosine warm start disagrees with original same-input control')
    return records


def verify_recorded_code(metadata):
    """Read only recorded Git blobs; an unavailable revision stays unverified."""
    revision=metadata.get('git_head','')
    if not re.fullmatch(r'[0-9a-f]{40,64}',revision):
        return False
    repo=Path(__file__).resolve().parents[1]
    try:
        tree=subprocess.check_output(['git','ls-tree','-r','--name-only',revision],cwd=repo,text=True).splitlines()
        paths=sorted(p for p in tree if Path(p).parent.as_posix()=='uvarfs' and p.endswith('.py'))+['scripts/run_all.py']
        digest=hashlib.sha256()
        for path in paths:
            digest.update(path.encode())
            digest.update(subprocess.check_output(['git','show',f'{revision}:{path}'],cwd=repo).replace(b'\r\n',b'\n'))
    except (OSError,subprocess.CalledProcessError):
        return False
    if digest.hexdigest()!=metadata['code_fingerprint']:
        raise ValueError('saved code fingerprint differs from its recorded Git source')
    return True


def audit_fit_only(results,output):
    """Normal-fit diagnosis, never a substitute for missing anomaly metrics."""
    from uvarfs.protocol import config_fingerprint, expected_method_names
    validate_output(results,output)
    hashes=source_hashes(results)
    rows=[]; datasets=[]; versions=set()
    for dataset in DATASETS:
        folder=results/dataset
        if not (folder/'run_metadata.json').is_file():
            continue
        metadata=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        cfg=metadata['config']; versions.add(metadata['experiment_version'])
        if metadata['config_fingerprint']!=config_fingerprint(cfg) or sorted(metadata['expected_methods'])!=expected_method_names(cfg):
            raise ValueError(f'{dataset}: saved normal-fit protocol differs')
        if metadata['dataset']!=dataset:
            raise ValueError(f'{dataset}: dataset identity differs')
        # This frame is ONLY the declared method/input protocol, not fabricated
        # evaluation data. No scores, labels, feature dimensions or metrics.
        branches=sorted(experiment_branches(cfg),key=lambda b:len(b['prefix']),reverse=True)
        declared=[]
        for method in expected_method_names(cfg):
            branch=next(b for b in branches if method.startswith(b['prefix']))
            declared.append({'method':method,'layer_normalization':branch['layer_normalization'],
                             'objective_branch':branch['objective']})
        fit=json.loads((folder/'fit_manifest.json').read_text(encoding='utf-8'))
        paths=fit['normal_train_paths']; ids=fit['normal_train_indices']; budget=cfg['data']
        if (len(paths)!=len(ids) or len(set(ids))!=len(ids) or len(set(paths))!=len(paths) or
            any('/train/good/' not in '/'+p.replace('\\','/').lower() for p in paths)):
            raise ValueError(f'{dataset}: fit normal training manifest is inconsistent')
        if (fit['fit_images']!=len(paths) or fit['fit_images']>budget['fit_images'] or
            fit['variability_images']!=min(budget['variability_images'],len(paths)) or
            fit['patches_per_image']!=budget['patches_per_image'] or
            sum(b['batch_images'] for b in fit['sampled_patches'])!=len(paths) or
            any(len(set(b['patch_indices']))!=len(b['patch_indices']) or len(b['patch_indices'])>budget['patches_per_image'] for b in fit['sampled_patches']) or
            ('train_images' in metadata and len(paths)!=min(metadata['train_images'],budget['fit_images']))):
            raise ValueError(f'{dataset}: recorded normal fit budget differs from config')
        records=audit_representation_protocol(folder,cfg,pd.DataFrame(declared),fit)
        solver_files=list(folder.rglob('uvarfs_*.json'))
        for path in solver_files:
            audit_sparse_path(json.loads(path.read_text(encoding='utf-8')),cfg['uvarfs'])
        memory_path=folder/'memory_manifest.json'
        memory_available=memory_path.is_file()
        if memory_available:
            memory=json.loads(memory_path.read_text(encoding='utf-8'))
            if (not memory['shared_sampling_across_methods'] or sorted(memory['memory_rows'])!=expected_method_names(cfg) or
                len(set(memory['memory_rows'].values()))!=1 or max(memory['memory_rows'].values())>cfg['data']['memory_size'] or
                memory['memory_capacity']!=budget['memory_size'] or memory['reservoir_seed']!=cfg['seed'] or
                len(memory['normal_train_paths'])>budget['memory_images'] or
                len(memory['normal_train_paths'])!=len(memory['normal_train_indices']) or
                len(set(memory['normal_train_indices']))!=len(memory['normal_train_indices']) or
                any('/train/good/' not in '/'+p.replace('\\','/').lower() for p in memory['normal_train_paths'])):
                raise ValueError(f'{dataset}: unequal/non-normal memory protocol')
        for record,branch in zip(records,experiment_branches(cfg)):
            obj=json.loads((folder/branch['folder']/'uvarfs_main.json').read_text(encoding='utf-8'))
            actual=record['normal_cosine_geometry']['methods']['main']['vs_selected_input_cosine']
            pool=obj.get('candidate_pool',[])
            rows.append({'dataset':dataset,'method':record['method'],'objective':record['objective_definition'],
                'layers':';'.join(map(str,obj['layers'])),'feature_dim':len(obj['active']),
                'selection_geometry_error':obj['geometry_error'],'geometry_feasible':obj['feasible'],
                'actual_cosine_geometry_error':actual['cosine_gram_relative_error'],
                'normal_patch_1nn_agreement':actual['normal_patch_1nn_agreement'],
                'selected_lambda':obj['lambda'],'solver_converged':obj['solver_converged'],
                'projected_residual':obj['projected_residual'],
                'effective_weight_dimension':obj.get('effective_weight_dimension'),
                'candidate_count':len(pool),'stationary_candidates':sum(c['fixed_support_converged'] for c in pool),
                'line_search_stalled_candidates':sum(c['solver_status']=='line_search_stalled' for c in pool),
                **(cosine_feasibility_diagnostics(obj,cfg['uvarfs']) if obj.get('objective_definition') in {COSINE,PERFORMANCE} else {})})
        datasets.append({'dataset':dataset,'solver_files_checked':len(solver_files),
            'git_blob_code_verified':verify_recorded_code(metadata),'git_head':metadata['git_head'],
            'memory_manifest_available':memory_available,'evaluation_progress':json.loads((folder/'eval_progress.json').read_text(encoding='utf-8')) if (folder/'eval_progress.json').exists() else None,
            'final_metrics_available':(folder/'metrics.csv').is_file(),
            'predictions_available':(folder/'image_predictions.csv').is_file()})
    if not datasets or len(versions)!=1:
        raise ValueError('fit-only audit requires returned fit records from one experiment version')
    if source_hashes(results)!=hashes:
        raise ValueError('original experiment artifacts changed during the fit-only audit')
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows); frame.to_csv(output/'normal_fit_summary.csv',index=False)
    data={'analysis_scope':'normal_fit_only','anomaly_performance_assessed':False,
          'experiment_version':next(iter(versions)),'datasets':datasets,'source_sha256':hashes}
    (output/'analysis_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    columns=['dataset','method','layers','feature_dim','geometry_feasible','actual_cosine_geometry_error',
             'normal_patch_1nn_agreement','projected_residual','effective_weight_dimension']
    display=frame.copy()
    for column in ['actual_cosine_geometry_error','normal_patch_1nn_agreement','projected_residual','effective_weight_dimension']:
        display[column]=display[column].map(lambda value:f'{value:.6g}' if pd.notna(value) else 'n.a.')
    lines=['# 回传正常拟合诊断','',f'版本：{next(iter(versions))}；可核查数据集：'+', '.join(d['dataset'] for d in datasets)+'。','',
           '本报告只分析已回传的正常训练/拟合产物，不计算或推断异常检测性能。逐图预测与最终指标是否回传：','',
           markdown_table(pd.DataFrame(datasets),['dataset','final_metrics_available','predictions_available','git_blob_code_verified']),'',
           '## 正常几何与稀疏','',markdown_table(display,columns),'',
           'geometry_feasible 只表示对应目标的训练几何满足预设容差。actual cosine 全局 Gram 误差降低不保证局部最近邻更好；normal_patch_1nn_agreement 仅是采样正常 patch 的诊断，不是 AUROC/AUPRO。','',
           '固定支持残差与几何可行性分别报告。line_search_stalled 或 iteration_limit 不能表述为已收敛；新目标也没有原 quadratic 的全局凸证书。','',
           'test 进度 100% 仅表示 DINO/NN 前向完成。原实现随后逐方法 bootstrap，预测在全部统计后才落盘；没有退出日志时不能将该状态判定为崩溃。','',
           '训练目标、ASLS、raw Main、beta/lambda/K、正常数据预算与 cosine detector 均保持；此次仅修改统计收尾的等价计算、预测提前保存与阶段进度。最终仍须完整 BMAD 六数据集和全部对照，不能用缺失的检测指标调参。','']
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    return data


def completed_return_scope(results, frame):
    """Reject inconsistent partial returns; include only fully evaluated datasets."""
    states=[]; completed=[]
    for dataset in DATASETS:
        folder=results/dataset
        progress_path=folder/'eval_progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        metrics=(folder/'metrics.csv').is_file()
        done=progress.get('stage')=='complete'
        if metrics!=done:
            raise ValueError(f'{dataset}: metrics and complete evaluation state disagree')
        if done:
            completed.append(dataset)
        states.append({'dataset':dataset,'status':'complete' if done else
                       'incomplete' if folder.is_dir() else 'absent',
                       'evaluation_stage':progress.get('stage')})
    if not completed or set(frame.dataset)!=set(completed) or frame.duplicated(['dataset','method']).any():
        raise ValueError('partial aggregate must contain exactly the completed datasets without duplicate methods')
    return completed,states


def audit(results: Path, output: Path, draws: int, completed_only=False):
    validate_output(results,output)
    hashes=source_hashes(results)
    aggregate=results/('all_metrics.csv' if (results/'all_metrics.csv').is_file() or not completed_only else 'all_metrics.partial.csv')
    df=pd.read_csv(aggregate)
    datasets,states=completed_return_scope(results,df) if completed_only else (DATASETS,[])
    if not completed_only and (sorted(df.dataset.unique())!=sorted(DATASETS) or df.duplicated(['dataset','method']).any()):
        raise ValueError('run must contain exactly six datasets without duplicate methods')
    versions=df.experiment_version.unique()
    if len(versions)!=1:
        raise ValueError('mixed experiment versions are not comparable')
    output.mkdir(parents=True,exist_ok=True)
    summary_rows=[]; diagnostics=[]; paired_rows=[]; prevalence_rows=[]; lambda_rows=[]; zero_rows=[]
    config_hashes=set(); code_hashes=set(); revisions=set()
    for dataset in datasets:
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
        if meta['experiment_version']!=versions[0]:
            raise ValueError(f'{dataset}: metadata and metric versions disagree')
        cfg=meta['config']; seeds=cfg['random_baseline_seeds']
        if completed_only and json.loads((folder/'uvarfs_main.json').read_text()).get('objective_definition') not in {COSINE,PERFORMANCE}:
            raise ValueError('completed-only reporting currently requires a cosine/simplex primary objective')
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
        for method in ['asls_fixed_budget_uvarfs','asls_compression_uvarfs','asls_selected_raw','all_raw','all_uvarfs','fixed4_uvarfs','asls_pooled_uvarfs','asls_pca','asls_geometry_search_uvarfs','asls_gate_prefix_uvarfs',
                       'asls_top_weights_uvarfs','asls_prune_refit_uvarfs','asls_exchange_refit_uvarfs','legacy_main',
                       'layer_l2_main','raw_input_main','gram_main']:
            if method in source.method.values:
                comparisons[method]=[method]
        intervals=paired_bootstrap(predictions,comparisons,draws=draws)
        main=source.set_index('method').loc['main']
        for family,methods in comparisons.items():
            delta=float(main.image_auroc-source.set_index('method').loc[methods].image_auroc.mean())
            paired_rows.append({'dataset':dataset,'target_method':'main','comparison':family,'delta_auroc':delta,
                                'ci95_lower':intervals[family][0],'ci95_upper':intervals[family][1],
                                'bootstrap_draws':draws,'bootstrap_seed':42,'bootstrap_unit':'image',
                                'stratified_by_label':True})
        # Compare each other input branch against its OWN baselines.
        branches=experiment_branches(cfg)
        for branch in branches[1:]:
            prefix=branch['prefix']; target=prefix+'main'
            groups={prefix+family:[prefix+family] for family in [
                'fixed4_raw','all_raw','all_uvarfs','asls_raw','asls_pca','asls_fixed_budget_uvarfs','asls_compression_uvarfs','asls_selected_raw','legacy_main',
                'asls_prune_refit_uvarfs','asls_exchange_refit_uvarfs'] if prefix+family in source.method.values}
            groups.update({prefix+family:[f'{prefix}{family}_seed{s}' for s in seeds]
                           for family in ['randomk_raw','randomk_uvarfs','asls_random']})
            controls=[other for other in branches if other['layer_normalization']==branch['layer_normalization'] and other['objective']==QUADRATIC]
            if branch['objective'] in {COSINE,PERFORMANCE} and controls:
                control=controls[0]['prefix']+'main'; groups[control]=[control]
            intervals=paired_bootstrap(predictions,groups,draws=draws,target=target)
            for family,methods in groups.items():
                delta=float(source.set_index('method').loc[target].image_auroc-source.set_index('method').loc[methods].image_auroc.mean())
                paired_rows.append({'dataset':dataset,'target_method':target,'comparison':family,'delta_auroc':delta,
                                    'ci95_lower':intervals[family][0],'ci95_upper':intervals[family][1],
                                    'bootstrap_draws':draws,'bootstrap_seed':42,'bootstrap_unit':'image',
                                    'stratified_by_label':True})
        for branch in branches:
            prefix=branch['prefix']
            target=prefix+'asls_raw'
            groups={prefix+family:[prefix+family] for family in ['fixed4_raw','all_raw']}
            groups[prefix+'randomk_raw']=[f'{prefix}randomk_raw_seed{s}' for s in seeds]
            intervals=paired_bootstrap(predictions,groups,draws=draws,target=target)
            for family,methods in groups.items():
                delta=float(source.set_index('method').loc[target].image_auroc-source.set_index('method').loc[methods].image_auroc.mean())
                paired_rows.append({'dataset':dataset,'target_method':target,'comparison':family,'delta_auroc':delta,
                                    'ci95_lower':intervals[family][0],'ci95_upper':intervals[family][1],
                                    'bootstrap_draws':draws,'bootstrap_seed':42,'bootstrap_unit':'image',
                                    'stratified_by_label':True})
        prevalence_rows.append({'dataset':dataset,'normal':int((y==0).sum()),'abnormal':int((y==1).sum()),
                                'image_positive_prevalence':float(y.mean()),'main_image_auprc':float(main.image_auprc)})
        a=json.loads((folder/'asls.json').read_text())
        uv=json.loads((folder/'uvarfs_main.json').read_text())
        sparse_audit=audit_sparse_path(uv,cfg['uvarfs'])
        representation_audit=audit_representation_protocol(folder,cfg,source,fit)
        normal_geometry=None
        if (folder/'normal_geometry_audit.json').is_file():
            normal_geometry=json.loads((folder/'normal_geometry_audit.json').read_text())
            expected_rows=sum(batch['batch_images']*len(batch['patch_indices']) for batch in fit['sampled_patches'])
            if normal_geometry['selection_candidate'] is not False or normal_geometry['normal_geometry_rows']!=expected_rows:
                raise ValueError(f'{dataset}: normal geometry audit has incorrect scope or sample count')
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
                            'objective_definition':uv.get('objective_definition',QUADRATIC),
                            'geometry_metric':uv.get('geometry_metric','unrenormalized_weighted_gram_relative_frobenius'),
                            'fixed_support_first_order_residual':uv.get('fixed_support_first_order_residual'),
                            'effective_weight_dimension':uv.get('effective_weight_dimension'),
                            'budgeted_box_optimality_gap':uv.get('budgeted_box_optimality_gap'),
                            'fixed_support_box_gap':uv.get('fixed_support_box_gap'),
                            'lambda':uv['lambda'],'iterations':uv['solver_iterations'],
                            'sparse_path_audit':sparse_audit,'normal_cosine_geometry':normal_geometry,
                            'representation_audit':representation_audit,
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
    # Incomplete datasets must still belong to this exact run, rather than
    # being silently ignored when their provenance conflicts with final rows.
    for state in states:
        path=results/state['dataset']/'run_metadata.json'
        if not path.is_file():
            if state['status']!='absent':
                raise ValueError(f"{state['dataset']}: missing partial run provenance")
            continue
        meta=json.loads(path.read_text())
        digest=hashlib.sha256(json.dumps(meta['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (meta['experiment_version']!=versions[0] or digest!=meta['config_fingerprint'] or
            digest not in config_hashes or meta['code_fingerprint'] not in code_hashes or meta['git_head'] not in revisions):
            raise ValueError(f"{state['dataset']}: partial run provenance differs")
    summary=pd.DataFrame(summary_rows)
    returned_summary=pd.read_csv(results/'summary_metrics.csv')
    if (not set(returned_summary.dataset)<=set(datasets) or
        returned_summary.duplicated(['dataset','method_family']).any()):
        raise ValueError('returned summary contains unaudited datasets or duplicate families')
    for _,row in returned_summary.iterrows():
        reference=summary[(summary.dataset==row.dataset)&(summary.method_family==row.method_family)].iloc[0]
        for column in ['image_auroc_mean','image_auroc_std','image_auprc_mean','image_auprc_std','feature_dim_mean']:
            np.testing.assert_allclose(row[column],reference[column],rtol=1e-10,atol=1e-10)
    summary.to_csv(output/'family_summary.csv',index=False)
    pd.DataFrame(paired_rows).to_csv(output/'paired_auc_differences.csv',index=False)
    pd.DataFrame(prevalence_rows).to_csv(output/'image_prevalence.csv',index=False)
    pd.DataFrame(lambda_rows).to_csv(output/'main_lambda_path.csv',index=False)
    pd.DataFrame(zero_rows).to_csv(output/'zero_lambda_diagnostics.csv',index=False)
    shutil.copyfile(aggregate,output/('completed_metrics.csv' if len(datasets)<len(DATASETS) else 'all_metrics.csv'))
    shutil.copyfile(results/'layer_selection.csv',output/'layer_selection.csv')
    if source_hashes(results)!=hashes:
        raise ValueError('original experiment artifacts changed during the audit')
    data={'experiment_version':versions[0],'git_revision':next(iter(revisions)),
          'config_fingerprint':next(iter(config_hashes)),'code_fingerprint':next(iter(code_hashes)),
          'analysis_scope':'completed_datasets_only' if len(datasets)<len(DATASETS) else 'all_six_datasets',
          'completed_datasets':datasets,'dataset_status':states,
          'source_aggregate':aggregate.name,'anomaly_performance_assessed':True,
          'all_six_complete':len(datasets)==len(DATASETS),
          'rows':len(df),'diagnostics':diagnostics,'source_sha256':hashes,
          'bootstrap_draws':draws,'bootstrap_unit':'image','bootstrap_seed':42,
          'bootstrap_batch_size':32,'macro':f'equal weight over {len(datasets)} completed datasets only, random seeds averaged within dataset'}
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
    report=write_report if versions[0]=='gpu-eval-v4-protocol-fixes' else write_cosine_report if diagnostics[0]['objective_definition'] in {COSINE,PERFORMANCE} else write_current_report
    report(output,df,summary,pd.DataFrame(paired_rows),pd.DataFrame(prevalence_rows),data)
    return data


def write_cosine_report(output,df,summary,paired,prevalence,data):
    """Report the approved new target without borrowing old convex claims."""
    datasets=data.get('completed_datasets',DATASETS)
    partial=len(datasets)<len(DATASETS)
    main=df[df.method=='main'].set_index('dataset').loc[datasets]
    macro=summary.groupby('method_family').image_auroc_mean.mean().sort_values(ascending=False)
    table=main[['selected_layers','feature_dim','image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].reset_index()
    for column in ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']:
        table[column]=table[column].map(lambda value:f'{value*100:.2f}' if pd.notna(value) else 'n.a.')
    comparisons=pd.DataFrame({'dataset':datasets})
    for family in ['main','gram_main','asls_fixed_budget_uvarfs','asls_compression_uvarfs','layer_l2_asls_compression_uvarfs','asls_raw','asls_pca','asls_selected_raw','gram_asls_selected_raw','layer_l2_asls_selected_raw','layer_l2_gram_asls_selected_raw','asls_random','fixed4_raw','all_raw','all_uvarfs',
                   'gram_asls_pca','gram_asls_random','layer_l2_main','layer_l2_gram_main','layer_l2_asls_raw']:
        if family not in macro:
            continue
        rows=summary[summary.method_family==family].set_index('dataset').loc[datasets]
        comparisons[family]=[f'{row.image_auroc_mean*100:.2f} ± {row.image_auroc_std*100:.2f}' if row.runs>1 else f'{row.image_auroc_mean*100:.2f}' for _,row in rows.iterrows()]
    pairs=paired.copy()
    for column in ['delta_auroc','ci95_lower','ci95_upper']:
        pairs[column]*=100
    normal_rows=[]; gram_bounds=[]; feasibility_rows=[]; control_rows=[]
    for diagnostic in data['diagnostics']:
        for record in diagnostic['representation_audit']:
            own=df[(df.dataset==diagnostic['dataset'])&(df.method==record['method'])].iloc[0]
            normal=record['normal_cosine_geometry']['methods']['main']
            normal_rows.append({'dataset':diagnostic['dataset'],'method':record['method'],
                'objective':record['objective_definition'],'feature_dim':own.feature_dim,
                'selection_geometry_error':own.uvarfs_geometry_error,'feasible':own.uvarfs_geometry_feasible,
                'actual_cosine_error':normal['vs_selected_input_cosine']['cosine_gram_relative_error'],
                'normal_fit_patch_1nn_agreement':normal['vs_selected_input_cosine']['normal_patch_1nn_agreement'],
                'fixed_support_first_order_residual':own.get('uvarfs_fixed_support_first_order_residual'),
                'effective_weight_dimension':own.get('uvarfs_effective_weight_dimension'),
                'continuous_box_gap':own.get('uvarfs_box_optimality_gap')})
            sparse=record['sparse_path_audit']
            if record['objective_definition'] in {COSINE,PERFORMANCE}:
                feasibility_rows.append({'dataset':diagnostic['dataset'],'method':record['method'],
                    **{k:sparse[k] for k in ['generated_feasible_candidates','lambda_path_feasible_points',
                       'generated_min_geometry_error','generated_sparsest_feasible_dimension',
                       'feasibility_diagnosis','feasibility_scope','global_budget_infeasibility_claimed']}})
            control=record['normal_cosine_geometry']['methods'].get('asls_selected_raw')
            if control is not None:
                control_rows.append({'dataset':diagnostic['dataset'],'source_main':record['method'],
                    'feature_dim':control['feature_dim'],
                    'cosine_error_vs_selected_input':control['vs_selected_input_cosine']['cosine_gram_relative_error'],
                    'cosine_error_vs_weighted_main':control['vs_weighted_main_cosine']['cosine_gram_relative_error'],
                    'normal_1nn_agreement_vs_weighted_main':control['vs_weighted_main_cosine']['normal_patch_1nn_agreement']})
            bound=record.get('budget_geometry_audit')
            if bound is not None:
                gram_bounds.append({'dataset':diagnostic['dataset'],'method':record['method'],
                    'original_gram_budget_lower_bound':bound['geometry_error_lower_bound'],
                    'original_budget_ruled_out':bound['budget_ruled_out_at_working_precision']})
    normal_frame=pd.DataFrame(normal_rows)
    normal_frame.to_csv(output/'normal_cosine_geometry.csv',index=False)
    selected_columns=[name for name in ['dataset','method','layer_normalization','objective_branch','feature_dim','selected_layer_count',
        'compression_ratio','memory_size','uvarfs_effective_weight_dimension','matching_and_aggregation_seconds',
        'fit_seconds','memory_seconds','eval_seconds','peak_gpu_gb'] if name in df]
    efficiency=df[df.method.isin(['main','gram_main','asls_raw','all_raw','layer_l2_main','layer_l2_gram_main'])][selected_columns].copy()
    lines=['# Cosine/simplex U-VaRFS '+('部分回传分析' if partial else '全六回传分析'),'',
        f'版本 `{data["experiment_version"]}`；Git `{data["git_revision"]}`；共 {len(df)} 行。训练不使用异常标签或 masks，所有 test labels 仅用于本报告评价。', '',
        f'Main {"Image" if len(datasets)==1 else "Macro Image"} AUROC={100*macro["main"]:.2f}%。仅纳入 {len(datasets)}/6 个完整数据集：{", ".join(datasets)}；随机方法先按五 seeds 聚合，不据 test AUROC 选主输入、目标、lambda 或预算。', '',
        '## 主结果','',markdown_table(table,table.columns.tolist()),'',
        '## 同输入、同层与同维对照','',markdown_table(comparisons,comparisons.columns.tolist()),'',
        'gram_* 保留原 quadratic objective 和 v11 稀疏求解；每个输入/目标分支的 PCA/Random 匹配自己的 Main 维数，Random-K 匹配 ASLS 层数。原/新目标会产生不同支持集与维数，应同时看性能和压缩，不能把跨输入变化归因于特征选择。', '',
        '原算法的 Top-weight/prune/exchange/legacy 对照仍是 quadratic 目标，不因它们出现在 cosine 主分支而改写其数学身份；CSV 分别记录 objective_branch 和 uvarfs_objective_definition。', '',
        '## 配对图像评价','',markdown_table(pairs,pairs.columns.tolist()),'',
        f'{data["bootstrap_draws"]} 次按正常/异常分层的 image bootstrap，差值为 target_method 减 comparison（百分点）。随机 seeds 在每次 draw 内平均 AUC，不做预测 ensemble。无患者/slide 分组、多重比较校正或预设非劣界值。', '',
        '## 实际正常几何与证书作用域','',markdown_table(normal_frame.round(7),normal_frame.columns.tolist()),'',
        'cosine_simplex_cardinality 的 feasible 使用 actual cosine Gram error。权重总和为 1，维数惩罚是 λ·非零维数，variability 相对 full-input uniform weights 标定；beta/lambda 数字固定，但目标单位与原公式不同。', '',
        '新目标非凸且包含离散支持。已独立核验有限候选池的每个 lambda 最低完整目标、simplex/floor/预算、权重/scales、一阶残差字段和 normal-only 选解。固定支持一阶残差不能证明全局稀疏最优；原 box gap、fixed-support Gram floor 和全局 Gram 预算下界只适用于原目标。', '',
        '正常近邻仅排除自身，包含同图 patches；它不是官方 memory/test 成绩。正常 geometry 已保存且有明确来源，但未回传 fit features 时不能宣称报告重算了原始正常 Gram。', '',
        '## 压缩与效率','',markdown_table(efficiency.round(6),efficiency.columns.tolist()),'',
        '实际非零维数与有效权重维数均报告，以免用极小权重充数。fit/memory/eval/peak 为整套方法共享开销，包括原目标 warm start 和完整对照；不能重复计作单方法成本，也不能从 GPU 求解宣称科学创新或真实加速。', '',
        '## 来源与实验边界','',
        f'Git blobs 与保存 code fingerprint 核验：{data["git_blob_code_verified"]}。已完成数据集的汇总/逐数据集 CSV、逐图 AUROC/AP、随机 mean/sample SD、全部正常清单、ASLS 共享和方法维数核验通过。原始文件 SHA256 在 analysis_data.json，读取前后保持。', '',
        '原始输入 Main 固定，完整 L2 消融保留；本轮目标升级授权不等于 L2 Main 切换。旧结果不能 resume 或混入当前版本。性能变化需由全部六数据集和 Brain/Liver/RESC localization 一起评估，不能把数学或 fixture 通过当作正向 AD 效果。', '',
        '实验产物与本报告仅保留本地；代码与开发交接按默认授权同步仓库。', '']
    if PERFORMANCE in set(df.get('uvarfs_objective_definition',[])):
        lines=[line.replace('Cosine/simplex U-VaRFS','性能优先 U-VaRFS') for line in lines]
        lines=[line for line in lines if not line.startswith(('cosine_simplex_cardinality 的 feasible','新目标非凸且包含离散支持。'))]
        lines+=['## 性能优先协议','',
            'Main 使用 cosine_simplex_fixed_budget：固定既定最大 K，保留原正常 cosine/variability 项，权重 floor/K≤p≤4/K（具体 cap 以 config 为准），不扫 lambda 或优先减少维数。有限候选池先满足正常几何约束，再比较正常完整目标；没有可行候选时明确 fallback，不宣称预算全局不可行。', '',
            'v18 如存在 refitted_support_exchange，交换提案先在同一 capped simplex 重拟合后再比较正常目标；全部 v17 候选保留，并以 asls_fixed_budget_uvarfs 同轮评价。接受时保护已有正常几何可行状态，拒绝提案成本也计入，不证明异常性能必增。', '',
            '旧 cosine/cardinality/lambda Main 以 asls_compression_uvarfs 单独保留，原 quadratic 仍是完整 gram_* 分支。same-support/PCA/Random 匹配各支 Main K。权重 cap 的有效维数下界只限制系数集中，不证明异常 AUROC/AUPRO 改善或独立信息维数。', '']
    if partial:
        states=pd.DataFrame(data['dataset_status'])
        lines[4:4]=['仅评价上述已完成数据集；其余数据集尚无最终评价，不构成全六性能结论，也不能据局部 test 结果修改主输入、目标或超参。','',
                    markdown_table(states,states.columns.tolist()),'']
    if feasibility_rows:
        frame=pd.DataFrame(feasibility_rows); frame.to_csv(output/'cosine_feasibility_scope.csv',index=False)
        lines+=['## 有限候选池与 lambda 路径可行性','',markdown_table(frame.round(7),frame.columns.tolist()),'',
                '原 cosine/cardinality 控制的 feasible 评价规定 lambda 路径最优点；performance-first 的 feasible 评价固定 K 生成池，且不做 lambda 搜索（路径数量为缺省）。两种有限集合都不证明全局预算不可行，报告以 feasibility_scope 区分。','']
    if control_rows:
        frame=pd.DataFrame(control_rows); frame.to_csv(output/'selected_support_geometry.csv',index=False)
        lines+=['## 相同支持去权重对照','',markdown_table(frame.round(7),frame.columns.tolist()),'',
                'asls_selected_raw 复用各分支 Main 的有序层与 active 索引，均匀相对权重，不重新拟合或选维。对照的异常检测评价用于区分已选支持与相对权重的作用，不用于选择 Main、lambda、支持或主输入；实际零行与正常几何记录在诊断 JSON。','']
    if gram_bounds:
        frame=pd.DataFrame(gram_bounds); frame.to_csv(output/'original_gram_budget_geometry.csv',index=False)
        lines+=['## 原目标预算下界（只作对照）','',markdown_table(frame.round(7),frame.columns.tolist()),'',
                '这些下界覆盖原 `[0,1]` box 权重与 unrenormalized Gram；不能用于排除新 cosine/simplex 目标的可行性。','']
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')


def write_current_report(output,df,summary,paired,prevalence,data):
    main=df[df.method=='main'].set_index('dataset').loc[DATASETS]
    macro=summary.groupby('method_family').image_auroc_mean.mean().sort_values(ascending=False)
    table=main[['selected_layers','feature_dim','image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']].reset_index()
    for column in ['image_auroc','image_auprc','pixel_auroc','pixel_auprc','aupro']:
        table[column]=table[column].map(lambda x:f'{100*x:.2f}' if pd.notna(x) else 'n.a.')
    comparison=pd.DataFrame({'dataset':DATASETS})
    families=['main','layer_l2_main','raw_input_main','legacy_main','asls_prune_refit_uvarfs','asls_exchange_refit_uvarfs','asls_top_weights_uvarfs','asls_gate_prefix_uvarfs',
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
        'uniform_gate_loss_slope':float(d['config']['asls']['sparsity_weight'])-1,
        'gates_near_all_open':d['gates_near_all_open']} for d in data['diagnostics']])
    lambda_path=pd.read_csv(output/'main_lambda_path.csv')
    budget_rows=[]
    for dataset,group in lambda_path.groupby('dataset',sort=False):
        point=group.loc[group['lambda'].idxmin()]
        budget_rows.append({'dataset':dataset,'min_lambda':point['lambda'],
            'continuous_active_features':point.continuous_active_features,
            'continuous_geometry_error':point.continuous_geometry_error,
            'retained_features':point.retained_features,'budgeted_geometry_error':point.geometry_error})
    pairs=paired.assign(target_method=paired.get('target_method','main'))[
        ['dataset','target_method','comparison','delta_auroc','ci95_lower','ci95_upper']].copy()
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
    if 'legacy_main' in macro:
        lines.extend([f'同轮旧整体流程 legacy_main Macro Image AUROC={100*macro["legacy_main"]:.2f}%；Main 相差 {100*(macro["main"]-macro["legacy_main"]):+.2f} 个百分点。该对照与 Main 共享本轮正常扰动和 memory，避免将跨版本随机实现差异误归因于稀疏算法。',''])
        if 'asls_top_weights_uvarfs' in macro:
            lines.extend([f'同一 Main 选层上的特征算法对照 asls_top_weights_uvarfs={100*macro["asls_top_weights_uvarfs"]:.2f}%，Main 差值为 {100*(macro["main"]-macro["asls_top_weights_uvarfs"]):+.2f} 个百分点。结合 prefix 对照，可区分选层变化与特征求解变化；不能将稀疏/几何达标写成检测性能已改善。',''])
    lines.extend(['## 来源与完整性','',
        f'Git `{data["git_revision"]}`；code `{data["code_fingerprint"]}`；config `{data["config_fingerprint"]}`。六份元数据一致，源码 Git blobs 验证结果：{data["git_blob_code_verified"]}。',
        '全局/逐数据集 CSV 一致；全部逐图 AUROC/AP 独立重算一致，family mean/sample SD 一致；正常 fit/memory manifest 与共享抽样预算通过核查。来源 SHA256 在 analysis_data.json，审计前后未改变原文件。',
        'Brain/Liver/RESC 异常 masks 完整；其余数据集无 pixel 指标。所有标签仅用于评价，没有用 test AUROC 选层、lambda、维度或 detector。', '',
        '## 主结果','', '指标均为百分数；不以高 Pixel AUROC 替代 Pixel AP/AUPRO。','',
        markdown_table(table,table.columns.tolist()),'', '![Performance](performance.png)','',
        '## 同轮公平比较','', 'Random 项均纳入全部五 seeds，报告 mean ± sample SD；每个输入分支的 Random-K 与该分支 Main 的 K 相同。','',
        markdown_table(comparison,comparison.columns.tolist()),'',
        f'在相同 ASLS 层输入下，U-VaRFS 在 {wins["asls_raw"]}/6 数据集高于 Raw，在 {wins["asls_random"]}/6 高于同维度 Random feature 均值，在 {wins["asls_pca"]}/6 高于 PCA。不能据此宣称全部数据集优于固定层或 PCA。','',
        f'配对差值为各行 target_method 减 comparison，单位百分点。{data["bootstrap_draws"]} 次按正常/异常分层的 image bootstrap；每次 draw 内取各 seed AUC 均值，不将 seed 预测均值当作 ensemble。各输入分支与自己的 baseline 比较，不能把跨输入改善全部归因于 U-VaRFS。','',
        markdown_table(pairs,pairs.columns.tolist()),'',
        '未提供患者/slide 分组，区间仅以图像为抽样单位；没有多重比较校正或预设非劣界值，不宣称患者级显著性或非劣。','',
        '## 正常训练几何与数值诊断','',markdown_table(diagnostic.round(8),diagnostic.columns.tolist()),'',
        '均匀 gates p=t·1 时，当前 loss 精确化为 (1-t) + gamma·mean(v) + rho·t，其方向导数是 rho-1。默认 rho=0.02，方向导数 -0.98；增大所有 gates 会降低 loss。该数学诊断与 gates 全开现象相容，不能把离散组合搜索的压缩直接归功于学得了稀疏 gates。这里没有按测试指标调整 rho 或重定义 ASLS loss。','',
        'ASLS feasible 指等权单位层 Gram 的离散容差，U-VaRFS feasible 指原目标的预算后加权 Gram 容差；它们不能直接证明 detector 再归一化后的 cosine 几何或正常近邻被保持。前缀规则失败不证明全部组合不可行；geometry_search 穷尽整个预算仍失败才说明该采样正常几何下没有可行组合。连续 gates 接近全开，其微小排序差异不等于成功的自适应稀疏。','',
        markdown_table(gate_diagnostic.round(6),gate_diagnostic.columns.tolist()),'',
        'continuous_objective_converged/box gap 对应截断前的完整连续权重，不能用来证明截断后的表示优化充分。完整 lambda path 见 main_lambda_path.csv，lambda=0 只作为诊断，见 zero_lambda_diagnostics.csv，不参与选解。', '',
        '下表是既定 path 最小 lambda 的连续解与实际预算后几何误差，不用于增改 lambda grid。','',
        markdown_table(pd.DataFrame(budget_rows).round(6),list(budget_rows[0])), '',
        '连续解较好、Top-weight 截断后误差增大，说明预算/支持集截断值得单独诊断；连续解也不可行则还可能涉及既定 variability 正则与输入几何。不能仅提高迭代数、放宽容差或用测试结果挑 beta/lambda。', '',
        '## 按原主线实施的修改','',
        f'本轮 ASLS 离散规则为 {data["diagnostics"][0].get("discrete_selection","gate_prefix")}，特征稀疏策略为 {data["diagnostics"][0].get("sparsity_strategy","top_weights")}。规则由运行前配置和用户授权固定，不按 test AUROC 择优。v8 保留 prefix、旧 Top-weight 和 legacy_main 对照；v9 再保留同层的 v8 prune/refit 对照。','',
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
    normal_rows=[]
    for diagnostic in data['diagnostics']:
        audit=diagnostic.get('normal_cosine_geometry')
        if audit is None:
            continue
        point=audit['methods']['main']
        normal_rows.append({'dataset':diagnostic['dataset'],
            'layer_normalization':audit.get('layer_feature_normalization','none'),
            'unit_layer_vs_full_input_error':audit['unit_layer_consensus_vs_raw_full_cosine_error'],
            'main_vs_full_input_cosine_error':point['vs_raw_full_hierarchy']['cosine_gram_relative_error'],
            'normal_1nn_agreement':point['vs_raw_full_hierarchy']['normal_patch_1nn_agreement'],
            'main_vs_selected_input_cosine_error':point['vs_selected_raw_cosine']['cosine_gram_relative_error']})
    lines.extend(['## 稀疏求解与实际余弦几何核查','',
        'Main 每个 lambda 的预算、可行性、原选解规则以及相对旧候选的原目标/几何保护已独立核验。正常训练量改善不保证异常性能，连续解误差已超容差时不能仅归咎于 max_features 截断。',''])
    if normal_rows:
        frame=pd.DataFrame(normal_rows).round(8)
        frame.to_csv(output/'normal_cosine_geometry.csv',index=False)
        lines.extend([markdown_table(frame,frame.columns.tolist()),'',
            '以上是相同 fit patches 上的只读诊断，不参与层/特征/lambda 选择。full/selected input 指当前归一化分支裁剪前的输入，并非一定是原始 backbone 数值。近邻比较仅排除自身，包含同图 patches 且对 ties 敏感，不等同于官方 memory/test 检测结果。',''])
    else:
        lines.extend(['本轮没有返回原始 fit patch 矩阵及 normal_geometry_audit.json，无法从逐图预测或 Q 恢复实际 concat/再归一化 cosine 误差；不将小型数学 fixture 的差异冒充真实 BMAD 诊断。下一版本新增这项正常训练诊断，输入和主目标保持原样。',''])
    variants=[]
    for diagnostic in data['diagnostics']:
        for point in diagnostic.get('representation_audit',[]):
            metric=df[(df.dataset==diagnostic['dataset'])&(df.method==point['method'])].iloc[0]
            variants.append({'dataset':diagnostic['dataset'],'method':point['method'],
                'layer_normalization':point['layer_normalization'],'layers':point['layers'],
                'feature_dim':metric.feature_dim,'image_auroc':metric.image_auroc,
                'image_auprc':metric.image_auprc,'aupro':metric.aupro,
                'unit_layer_vs_full_input_cosine_error':point['normal_cosine_geometry']['unit_layer_consensus_vs_raw_full_cosine_error']})
    if variants:
        comparison=pd.DataFrame(variants)
        comparison.to_csv(output/'representation_comparison.csv',index=False)
        lines.extend(['## 表示输入的同轮对照','',markdown_table(comparison.round(7),comparison.columns.tolist()),'',
            '输入分支复用相同正常图像/patches、扰动及 DINO forward，各自计算对应表示的 variability，全部方法采用同一分支内的一致输入和 memory 预算。归一化是预先固定的输入协议，不能按 test AUROC 从两个分支挑选主方法。',''])
    budget_rows=[]
    for diagnostic in data['diagnostics']:
        for record in diagnostic.get('representation_audit',[]):
            audit=record.get('budget_geometry_audit')
            if audit is not None:
                budget_rows.append({'dataset':diagnostic['dataset'],'method':record['method'],
                    'global_geometry_error_lower_bound':audit['geometry_error_lower_bound'],
                    'necessary_features_lower_bound':audit['necessary_features_lower_bound'],
                    'budget_ruled_out':audit['budget_ruled_out_at_working_precision']})
    if budget_rows:
        frame=pd.DataFrame(budget_rows); frame.to_csv(output/'global_budget_geometry.csv',index=False)
        lines.extend(['## 全局特征预算必要条件','',markdown_table(frame.round(8),frame.columns.tolist()),'',
            '该下界来自正常 H 的参考 Gram 对齐量，覆盖全部预算内支持集及 box 权重，区别于固定支持集下界。下界未超过容差不证明可行，必要维数不构造可行解；数值计算有报告 margin，不能当作区间算术证明。它不参与选解、不自动调整维数/容差/beta/lambda。',''])
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
    order=['main','layer_l2_main','raw_input_main','legacy_main','asls_prune_refit_uvarfs','asls_top_weights_uvarfs','asls_gate_prefix_uvarfs',
           'asls_raw','fixed4_raw','all_raw','randomk_raw','randomk_uvarfs','asls_random','asls_pca']
    order=[name for name in order if name in family.method_family.values]
    if 'asls_pooled_uvarfs' in family.method_family.values:
        order.insert(1,'asls_pooled_uvarfs')
    figure,axes=plt.subplots(1,2,figsize=(17,8),gridspec_kw={'width_ratios':[1.5,1]})
    values=family.pivot(index='method_family',columns='dataset',values='image_auroc_mean').loc[order,DATASETS]*100
    im=axes[0].imshow(values.to_numpy(),vmin=40,vmax=100,cmap='viridis',aspect='auto')
    axes[0].set_xticks(range(6),DATASETS,rotation=30,ha='right'); axes[0].set_yticks(range(len(order)),order)
    axes[0].set_title('Image AUROC (%) — all seeds included')
    for i in range(len(order)):
        for j in range(6):
            axes[0].text(j,i,f'{values.iloc[i,j]:.1f}',ha='center',va='center',color='white' if values.iloc[i,j]<78 else 'black')
    figure.colorbar(im,ax=axes[0],shrink=.7)
    methods=[name for name in ['main','legacy_main','fixed4_raw','all_raw'] if name in family.method_family.values]
    width=.8/len(methods)
    for j,method in enumerate(methods):
        data=family[family.method_family==method].set_index('dataset').loc[['brain','liver','resc']]
        axes[1].bar(np.arange(3)+(j-(len(methods)-1)/2)*width,data.aupro_mean*100,width=width,label=method)
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
    parser.add_argument('--completed-only',action='store_true',help='audit only fully evaluated datasets in a partial cosine/simplex return; never claim all-six performance')
    parser.add_argument('--fit-only',action='store_true',help='audit returned normal-fit artifacts without claiming anomaly performance')
    args=parser.parse_args()
    if args.results is not None and not args.fit_only and not args.plots_only:
        from uvarfs.objectives import LOCAL
        metadata_paths=sorted(args.results.glob('*/run_metadata.json'))
        if metadata_paths and any(json.loads(p.read_text(encoding='utf-8')).get('uvarfs_objective')==LOCAL for p in metadata_paths):
            if args.bootstrap<100:
                parser.error('at least 100 bootstrap draws are required')
            from scripts.analyze_local_results import audit as audit_local
            audit_local(args.results,args.out,args.completed_only,args.bootstrap)
            return
    if args.fit_only:
        if args.results is None or args.plots_only or args.completed_only:
            parser.error('--fit-only requires --results and cannot be combined with --plots-only/--completed-only')
        audit_fit_only(args.results,args.out)
        return
    if args.plots_only:
        if args.completed_only:
            parser.error('--completed-only cannot be combined with --plots-only')
        plot(args.out)
    else:
        if args.results is None or args.bootstrap<100:
            parser.error('--results and at least 100 bootstrap draws are required')
        audit(args.results,args.out,args.bootstrap,completed_only=args.completed_only)


if __name__=='__main__':
    main()
