"""Fixed normal precision gate first; only then a fixed all-six detector audit."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,queue,subprocess,sys,threading,time,traceback
from pathlib import Path
from scripts.fixed_support_precision import strict_fit,strict_evaluation,atomic,read,sha256
from uvarfs.precision_refit import VERSION,SETTINGS
from uvarfs.protocol import code_fingerprint

WORKER='scripts.fixed_support_precision';ANALYZER='scripts.analyze_fixed_support_precision'
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
def live(pid,tick):
    try:s=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    except FileNotFoundError:return False
    return s[0]!='Z' and s[19]==str(tick)
def tasks():return [(42,'liver')]+[(s,d) for s in [42,123,3407] for d in ['brain','liver','resc','xray','camelyon16','oct2017'] if (s,d)!=(42,'liver')]
def entrypoints(project):
    result={name:project/path for name,path in {'worker_sha256':'scripts/fixed_support_precision.py','solver_sha256':'uvarfs/precision_refit.py','scheduler_sha256':'scripts/run_fixed_support_precision.py','analyzer_sha256':'scripts/analyze_fixed_support_precision.py','shared_normal_audit_sha256':'scripts/evaluate_mechanism_validation.py'}.items()}
    if any(not p.is_file() for p in result.values()):raise ValueError('missing canonical precision entrypoint')
    return result
def global_gate(root):
    root=Path(root);records={};passed=0
    for seed,dataset in tasks():
        out=root/'results'/f'seed{seed}'/dataset;done=strict_fit(out);passed+=done['precision_pass_count']
        for name in ['normal-fit.json','representation-specs.json','normal-complete.json','normal-files-sha256.json','normal-role-split.json']:
            path=out/name;records[path.relative_to(root.parents[2]).as_posix()]=sha256(path)
    return dict(evaluation_version=VERSION,blocks=18,weighted_fits=252,precision_pass_count=passed,all_252_pass=passed==252,
        residual_tolerance=SETTINGS['residual_tolerance'],normal_fit_sha256=records,
        test_data_used_for_gate=False,heldout_feature_values_used_for_gate=False,
        fixed_support_current_nearest_branch_only=True,global_optimality_claimed=False)
def validate_paths(project,parent,cache,root):
    results=(project/'results').resolve()
    if (root.parent!=(results/'remote-experiments') or not root.name.startswith('precision-validation-') or root.exists() or
        not parent.is_relative_to(results) or not cache.is_relative_to(results) or parent==cache):raise ValueError('fresh project-local precision batch required')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['parent-batch','cache-batch','run-root','control-dir']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--expected-head',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    project=Path(__file__).resolve().parents[1];parent=a.parent_batch.resolve();cache=a.cache_batch.resolve();root=a.run_root.resolve();validate_paths(project,parent,cache,root)
    for source in [parent,cache]:
        state=read(source/'state.json')
        if state['phase']!='complete' or state['live_workers'] or state['error'] or len(state['jobs'])!=18:raise ValueError('immutable parent/cache not fully closed')
        if any(j['status']!='strict_complete' or j['exit_code']!=0 or live(j['pid'],j['start_ticks']) for j in state['jobs'].values()):raise ValueError('old workers not successfully closed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    if head!=a.expected_head or subprocess.check_output(['git','diff','--name-only','HEAD','--','uvarfs','scripts','configs'],cwd=project,text=True).strip():raise ValueError('clean frozen precision source required')
    paths=entrypoints(project)
    import fcntl
    locks=[]
    for gpu in range(4):
        handle=(a.control_dir/f'gpu{gpu}.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    def available(gpu):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        ids={line.split(',')[1].strip():int(line.split(',')[0]) for line in raw.splitlines()}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if gpu not in ids.values() or any(ids.get(line.split(',')[0].strip())==gpu for line in apps.splitlines()):raise ValueError(f'GPU{gpu} absent or occupied; do not preempt')
    for gpu in range(4):available(gpu)
    if a.preflight_only:
        print(json.dumps(dict(preflight_only=True,head=head,entrypoint_sha256={k:sha256(v) for k,v in paths.items()},old_batches_closed=True,gpus_free=[0,1,2,3],run_created=False)),flush=True);return
    root.mkdir();logroot=project/'logs/remote-experiments'/root.name;logroot.mkdir(parents=True,exist_ok=True);(logroot/'jobs').mkdir()
    reportroot=project/'reports'/root.name;reportroot.mkdir(exist_ok=False)
    config=root/'fixed-precision.yaml';config.write_bytes((project/'configs/fixed_support_precision.yaml').read_bytes())
    subprocess.run(['git','archive','--format=tar','--output',str(root/'frozen-source.tar'),head],cwd=project,check=True)
    start=dict(head=head,code_fingerprint=code_fingerprint(project),evaluation_version=VERSION,pid=os.getpid(),start_ticks=ticks(os.getpid()),started_at_utc=utc(),
        parent_relative=parent.relative_to(project).as_posix(),parent_state_sha256=sha256(parent/'state.json'),cache_relative=cache.relative_to(project).as_posix(),cache_state_sha256=sha256(cache/'state.json'),
        config_sha256=sha256(config),source_archive_sha256=sha256(root/'frozen-source.tar'),physical_gpus=[0,1,2,3],source_fit_seeds=[42,123,3407],normal_fits=252,conditional_detector_rows=306,
        **{k:sha256(v) for k,v in paths.items()})
    atomic(root/'startup.json',start)
    state=dict(evaluation_version=VERSION,startup=start,phase='normal_engineering_gate',jobs={},live_workers=0,error=None,normal_complete=0,evaluation_complete=0,normal_engineering_gate_complete=False)
    mutex=threading.Lock();stop=threading.Event()
    def update():state['updated_at_utc']=utc();atomic(root/'state.json',state)
    def pins():
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()!=head or code_fingerprint(project)!=start['code_fingerprint'] or
            any(sha256(v)!=start[k] for k,v in paths.items()) or sha256(config)!=start['config_sha256'] or
            sha256(parent/'state.json')!=start['parent_state_sha256'] or sha256(cache/'state.json')!=start['cache_state_sha256']):raise ValueError('frozen precision source/config/reference changed')
    def run_one(task,gpu,phase):
        seed,dataset=task;pins();available(gpu);key=f'{phase}-seed{seed}-{dataset}';out=root/'results'/f'seed{seed}'/dataset
        log=logroot/'jobs'/(key+'.log');exitfile=logroot/'jobs'/(key+'-exit.json')
        if log.exists() or exitfile.exists():raise ValueError('duplicate precision job')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(project),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        argv=[sys.executable,'-u','-m',WORKER,'--phase',phase,'--out',str(out)]
        if phase=='fit':argv+=['--parent',str(parent/'results'/f'seed{seed}'/dataset),'--cache',str(cache/'results'/f'seed{seed}'/dataset),'--config',str(config)]
        else:argv+=['--gate',str(root/'global-normal-gate.json')]
        with log.open('xb') as f:
            child=subprocess.Popen(argv,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            record=dict(phase=phase,seed=seed,dataset=dataset,physical_gpu=gpu,pid=child.pid,start_ticks=ticks(child.pid),log=str(log),exit_file=str(exitfile),output=str(out),status='running',started_at_utc=utc())
            with mutex:state['jobs'][key]=record;state['live_workers']+=1;update()
            code=child.wait()
        done=dict(exit_code=code,finished_at_utc=utc(),pid=child.pid,start_ticks=record['start_ticks']);atomic(exitfile,done)
        with mutex:record.update(done,status='exited');state['live_workers']-=1;update()
        if code:raise RuntimeError('precision worker failed: '+str(log))
        pins();checked=strict_fit(out) if phase=='fit' else strict_evaluation(out)
        if checked['fit_seed']!=seed or checked['dataset']!=dataset:raise ValueError('worker block identity differs')
        with mutex:record.update(status='strict_complete',audit=checked);state['normal_complete' if phase=='fit' else 'evaluation_complete']+=1;update()
    def parallel(todo,phase):
        work=queue.Queue()
        for t in todo:work.put(t)
        def worker(gpu):
            while not stop.is_set():
                try:task=work.get_nowait()
                except queue.Empty:return
                try:run_one(task,gpu,phase)
                except BaseException:stop.set();raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in concurrent.futures.as_completed([pool.submit(worker,gpu) for gpu in range(4)]):result.result()
    update()
    try:
        run_one(tasks()[0],0,'fit');state['normal_engineering_gate_complete']=True;state['phase']='normal_fitting';update();parallel(tasks()[1:],'fit')
        gate=global_gate(root);atomic(root/'global-normal-gate.json',gate);state['precision_pass_count']=gate['precision_pass_count'];state['global_normal_gate_passed']=gate['all_252_pass'];update()
        if gate['all_252_pass']:
            state['phase']='evaluation_engineering_gate';update();run_one(tasks()[0],0,'evaluate');state['phase']='evaluation';update();parallel(tasks()[1:],'evaluate')
        state['phase']='reporting';update()
        subprocess.run([sys.executable,'-u','-m',ANALYZER,'--batch',str(root),'--out',str(reportroot/'final-analysis')],cwd=project,check=True)
        state['phase']='complete' if gate['all_252_pass'] else 'precision_inconclusive';state['finished_at_utc']=utc();update()
    except BaseException:state['phase']='failed';state['error']=traceback.format_exc();update();raise
    finally:
        for handle in locks:handle.close()

if __name__=='__main__':main()
