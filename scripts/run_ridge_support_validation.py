#!/usr/bin/env python3
"""Four-GPU, 18-block paired continuation; new ridge support fitting, no refit of historical controls."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, queue, subprocess, sys, threading, time, traceback
from pathlib import Path
from scripts.evaluate_ridge_support import VERSION, atomic, read, sha256, strict_complete
from uvarfs.protocol import code_fingerprint, BMAD_DATASETS


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--expected-head',required=True)
    p.add_argument('--cache-batch',type=Path,required=True)
    p.add_argument('--control-dir',type=Path,required=True,help='external GPU scheduling locks only')
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    source=a.source_batch.resolve();root=a.run_root.resolve()
    if (not source.is_relative_to((project/'results').resolve()) or
        root.parent!=(project/'results/remote-experiments').resolve() or
        not root.name.startswith('ridge-support-validation-') or root.exists()):
        raise ValueError('fresh project-local batch and frozen source required')
    cache=a.cache_batch.resolve()
    if not cache.is_relative_to((project/'results').resolve()):raise ValueError('project-local cache batch required')
    cached=read(cache/'state.json')
    if cached['phase']!='complete' or cached['live_workers'] or cached['error'] or len(cached['jobs'])!=18:raise ValueError('complete frozen cache batch required')
    old=read(source/'state.json')
    if old['phase']!='complete' or old['live_workers'] or old['error'] or len(old['jobs'])!=18:
        raise ValueError('source all-six three-fit batch must have finished successfully')
    for job in old['jobs'].values():
        if job['status']!='strict_complete' or job['exit_code']!=0:
            raise ValueError('source batch has incomplete workers')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    if head!=a.expected_head or subprocess.check_output(['git','diff','--name-only','HEAD','--','uvarfs','scripts','configs'],cwd=project,text=True).strip():
        raise ValueError('approved clean source revision required')
    import fcntl
    locks=[]
    for gpu in range(4):
        handle=(a.control_dir/f'gpu{gpu}.lock').open('a+')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    def available(gpu):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        ids={line.split(',')[1].strip():int(line.split(',')[0]) for line in raw.splitlines()}
        if gpu not in ids.values():raise ValueError('authorized GPU missing')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if any(ids.get(line.split(',')[0].strip())==gpu for line in apps.splitlines()):
            raise ValueError(f'physical GPU{gpu} occupied; do not preempt')
    for gpu in range(4):available(gpu)
    root.mkdir();logroot=(project/'logs/remote-experiments'/root.name).resolve()
    reportroot=(project/'reports'/root.name).resolve()
    if not logroot.is_relative_to(project) or not reportroot.is_relative_to(project):raise ValueError('output escapes project')
    logroot.mkdir(parents=True,exist_ok=True);(logroot/'jobs').mkdir();reportroot.mkdir(exist_ok=False)
    cfgpath=root/'fixed-ridge.yaml';cfgpath.write_bytes((project/'configs/ridge_support.yaml').read_bytes())
    subprocess.run(['git','archive','--format=tar','--output',str(root/'frozen-evaluation-source.tar'),head],cwd=project,check=True)
    pinned={'head':head,'code_fingerprint':code_fingerprint(project),'config_sha256':sha256(cfgpath),
        'worker_sha256':sha256(project/'scripts/evaluate_ridge_support.py'),
        'fitter_sha256':sha256(project/'scripts/fit_ridge_support.py'),'scheduler_sha256':sha256(__file__),'analyzer_sha256':sha256(project/'scripts/analyze_ridge_support_validation.py')}
    startup={**pinned,'evaluation_version':VERSION,'started_at_utc':utc(),'pid':os.getpid(),'start_ticks':ticks(os.getpid()),
        'physical_gpus':[0,1,2,3],'source_batch':str(source),'source_state_sha256':sha256(source/'state.json'),
        'cache_batch':str(cache),'cache_state_sha256':sha256(cache/'state.json'),'fit_seeds':[42,123,3407],'new_evaluations_per_block':10,'retained_controls_per_block':352,
        'unique_new_representations_per_block':8,'frozen_source_sha256':sha256(root/'frozen-evaluation-source.tar')}
    atomic(root/'startup.json',startup)
    tasks=[(42,d) for d in ['liver','brain','resc','xray','camelyon16','oct2017']]
    tasks += [(s,d) for s in [123,3407] for d in ['xray','brain','liver','resc','camelyon16','oct2017']]
    state={'evaluation_version':VERSION,'phase':'running','startup':startup,'jobs':{},'live_workers':0,
           'error':None,'task_order':[{'seed':s,'dataset':d} for s,d in tasks]}
    mutex=threading.Lock();stop=threading.Event();work=queue.Queue()
    for task in tasks[4:]:work.put(task)
    def update():state['updated_at_utc']=utc();atomic(root/'state.json',state)
    def pins():
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()!=head or
            code_fingerprint(project)!=pinned['code_fingerprint'] or sha256(cfgpath)!=pinned['config_sha256'] or
            sha256(project/'scripts/evaluate_ridge_support.py')!=pinned['worker_sha256'] or
            sha256(__file__)!=pinned['scheduler_sha256'] or
            sha256(project/'scripts/fit_ridge_support.py')!=pinned['fitter_sha256'] or
            sha256(cache/'state.json')!=startup['cache_state_sha256'] or
            sha256(project/'scripts/analyze_ridge_support_validation.py')!=pinned['analyzer_sha256'] or
            sha256(source/'state.json')!=startup['source_state_sha256']):
            raise ValueError('frozen source/config/source-batch identity changed')
    def run_task(seed,dataset,gpu):
        pins();available(gpu);key=f'seed{seed}-{dataset}';out=root/'results'/f'seed{seed}'/dataset
        log=logroot/'jobs'/(key+'.log');exitfile=logroot/'jobs'/(key+'-exit.json')
        if log.exists() or exitfile.exists() or out.exists():raise ValueError('duplicate task refused: '+key)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(project),
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        argv=[sys.executable,'-u','-m','scripts.evaluate_ridge_support','--source',str(source/'results'/f'seed{seed}'/dataset),
              '--out',str(out),'--selector-config',str(cfgpath),'--cache-source',str(cache/'results'/f'seed{seed}'/dataset)]
        t=time.monotonic()
        with log.open('xb') as f:
            child=subprocess.Popen(argv,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            rec={'seed':seed,'dataset':dataset,'physical_gpu':gpu,'pid':child.pid,'start_ticks':ticks(child.pid),
                 'started_at_utc':utc(),'log':str(log),'exit_file':str(exitfile),'output':str(out),'status':'running'}
            with mutex:state['jobs'][key]=rec;state['live_workers']+=1;update()
            code=child.wait()
        exitrecord={'exit_code':code,'finished_at_utc':utc(),'wall_seconds':time.monotonic()-t,
                    'pid':child.pid,'start_ticks':rec['start_ticks']};atomic(exitfile,exitrecord)
        with mutex:rec.update(exitrecord,status='exited');state['live_workers']-=1;update()
        if code:raise RuntimeError(key+' failed; inspect '+str(log))
        pins();audit=strict_complete(out)
        with mutex:rec.update(status='strict_complete',result_audit=audit);update()
    def gpu_thread(gpu):
        task=tasks[gpu]
        while not stop.is_set():
            try:run_task(*task,gpu)
            except BaseException:
                stop.set()
                with mutex:state['error']=traceback.format_exc();update()
                raise
            try:task=work.get_nowait()
            except queue.Empty:return
    update()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in concurrent.futures.as_completed([pool.submit(gpu_thread,gpu) for gpu in range(4)]):result.result()
        pins()
        if len(state['jobs'])!=18 or any(r['status']!='strict_complete' for r in state['jobs'].values()):
            raise ValueError('all 18 predefined blocks required')
        state['phase']='reporting';update()
        subprocess.run([sys.executable,'-u','-m','scripts.analyze_ridge_support_validation','--batch',str(root),
                        '--out',str(reportroot/'final-analysis')],cwd=project,check=True)
        state['phase']='complete';state['finished_at_utc']=utc();update()
    except BaseException:
        state['phase']='failed';state['error']=state['error'] or traceback.format_exc();update();raise
    finally:
        for handle in locks:handle.close()


if __name__=='__main__':main()
