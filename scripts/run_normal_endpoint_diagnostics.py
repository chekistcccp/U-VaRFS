"""Six predefined seed42 normal endpoint blocks with a formal Liver gate."""
from __future__ import annotations
import argparse,concurrent.futures,os,queue,subprocess,sys,threading,time,traceback
from pathlib import Path
from scripts.normal_endpoint_diagnostics import strict_complete,read,atomic,sha256
from scripts.run_fixed_support_precision import live,ticks,utc
from uvarfs.endpoint_diagnostics import VERSION
from uvarfs.protocol import code_fingerprint

def tasks():return ['liver','brain','resc','xray','camelyon16','oct2017']
def entrypoints(p):return {n:p/f for n,f in {'worker_sha256':'scripts/normal_endpoint_diagnostics.py','diagnostic_sha256':'uvarfs/endpoint_diagnostics.py','scheduler_sha256':'scripts/run_normal_endpoint_diagnostics.py','analyzer_sha256':'scripts/analyze_normal_endpoint_diagnostics.py','unchanged_objective_sha256':'uvarfs/matching_uvarfs.py'}.items()}
def validate_paths(project,source,root):
    results=(project/'results').resolve()
    if not results.is_relative_to(project) or not source.is_relative_to(results) or root.parent!=results/'remote-experiments' or not root.name.startswith('endpoint-diagnostics-') or root.exists():raise ValueError('fresh project-local endpoint batch required')

def prepare_output_dirs(project,root):
    logs=(project/'logs/remote-experiments'/root.name).resolve();reports=(project/'reports'/root.name).resolve()
    if not logs.is_relative_to(project) or not reports.is_relative_to(project) or reports.exists():raise ValueError('physical project-local log/report required')
    # The nohup launcher creates its wrapper files before the coordinator starts.
    if logs.exists() and (logs/'jobs').exists():raise ValueError('old worker log directory exists')
    root.mkdir();logs.mkdir(parents=True,exist_ok=True);(logs/'jobs').mkdir();reports.mkdir(parents=True)
    return logs,reports

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source-batch','run-root','control-dir']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-head',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    project=Path(__file__).resolve().parents[1];source=a.source_batch.resolve();root=a.run_root.resolve();validate_paths(project,source,root)
    old=read(source/'state.json')
    if old['phase']!='precision_inconclusive' or old['normal_complete']!=18 or old['evaluation_complete']!=0 or old['live_workers'] or old['error']:raise ValueError('closed inconclusive precision source required')
    start=read(source/'startup.json')
    if live(start['pid'],start['start_ticks']) or any(j['status']!='strict_complete' or j['exit_code'] or live(j['pid'],j['start_ticks']) or read(Path(j['exit_file']))['exit_code'] for j in old['jobs'].values()):raise ValueError('original precision processes not closed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    if head!=a.expected_head or subprocess.check_output(['git','diff','--name-only','HEAD','--','uvarfs','scripts','configs'],cwd=project,text=True).strip():raise ValueError('clean frozen endpoint source required')
    import fcntl
    locks=[]
    for gpu in range(4):
        handle=(a.control_dir/f'gpu{gpu}.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    def available(gpu):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        ids={line.split(',')[1].strip():int(line.split(',')[0]) for line in raw.splitlines()}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if gpu not in ids.values() or any(ids.get(line.split(',')[0].strip())==gpu for line in apps.splitlines()):raise ValueError(f'GPU{gpu} occupied or missing; no preemption')
    for gpu in range(4):available(gpu)
    paths=entrypoints(project)
    if a.preflight_only:
        print(__import__('json').dumps({'preflight_only':True,'head':head,'source_closed':True,'run_created':False,'free_physical_gpus':[0,1,2,3],'entrypoint_sha256':{k:sha256(v) for k,v in paths.items()}}),flush=True);return
    logs,reports=prepare_output_dirs(project,root)
    config=root/'fixed-endpoint.yaml';config.write_bytes((project/'configs/normal_endpoint_diagnostics.yaml').read_bytes())
    subprocess.run(['git','archive','--format=tar','--output',str(root/'frozen-source.tar'),head],cwd=project,check=True)
    startup=dict(evaluation_version=VERSION,head=head,code_fingerprint=code_fingerprint(project),pid=os.getpid(),start_ticks=ticks(os.getpid()),started_at_utc=utc(),
      source_relative=source.relative_to(project).as_posix(),source_state_sha256=sha256(source/'state.json'),config_sha256=sha256(config),source_archive_sha256=sha256(root/'frozen-source.tar'),
      source_seed=42,datasets=tasks(),endpoints=30,physical_gpus=[0,1,2,3],**{k:sha256(v) for k,v in paths.items()});atomic(root/'startup.json',startup)
    state={'phase':'normal_engineering_gate','jobs':{},'live_workers':0,'error':None,'complete_blocks':0,'planned_blocks':6,'started_at_utc':utc()};mutex=threading.Lock();stop=threading.Event()
    def update():state['updated_at_utc']=utc();atomic(root/'state.json',state)
    def pins():
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()!=head or code_fingerprint(project)!=startup['code_fingerprint'] or any(sha256(v)!=startup[k] for k,v in paths.items()) or sha256(config)!=startup['config_sha256'] or sha256(source/'state.json')!=startup['source_state_sha256']:raise ValueError('endpoint frozen source/config changed')
    def one(dataset,gpu):
        pins();available(gpu);out=root/'results'/dataset;log=logs/'jobs'/(dataset+'.log');exitfile=logs/'jobs'/(dataset+'-exit.json')
        if out.exists() or log.exists() or exitfile.exists():raise ValueError('duplicate endpoint task')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(project),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        args=[sys.executable,'-u','-m','scripts.normal_endpoint_diagnostics','--source',str(source/'results/seed42'/dataset),'--out',str(out),'--config',str(config)]
        with log.open('xb') as f:
            child=subprocess.Popen(args,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            rec=dict(dataset=dataset,physical_gpu=gpu,pid=child.pid,start_ticks=ticks(child.pid),log=str(log),exit_file=str(exitfile),output=str(out),status='running',started_at_utc=utc())
            with mutex:state['jobs'][dataset]=rec;state['live_workers']+=1;update()
            code=child.wait()
        done=dict(exit_code=code,finished_at_utc=utc(),pid=child.pid,start_ticks=rec['start_ticks']);atomic(exitfile,done)
        with mutex:rec.update(done,status='exited');state['live_workers']-=1;update()
        if code:raise RuntimeError('endpoint worker failed: '+str(log))
        pins();checked=strict_complete(out)
        if checked['dataset']!=dataset:raise ValueError('block identity differs')
        with mutex:rec.update(status='strict_complete',audit=checked);state['complete_blocks']+=1;update()
    def parallel():
        work=queue.Queue()
        for d in tasks()[1:]:work.put(d)
        def loop(gpu):
            while not stop.is_set():
                try:d=work.get_nowait()
                except queue.Empty:return
                try:one(d,gpu)
                except BaseException:stop.set();raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for r in concurrent.futures.as_completed([pool.submit(loop,gpu) for gpu in range(4)]):r.result()
    update()
    try:
        one('liver',0);state['phase']='normal_diagnostics';update();parallel();state['phase']='reporting';update();pins()
        subprocess.run([sys.executable,'-u','-m','scripts.analyze_normal_endpoint_diagnostics','--batch',str(root),'--out',str(reports/'final-analysis')],cwd=project,check=True)
        state['phase']='complete';state['finished_at_utc']=utc();update()
    except BaseException:state['phase']='failed';state['error']=traceback.format_exc();update();raise
    finally:
        for handle in locks:handle.close()

if __name__=='__main__':main()
