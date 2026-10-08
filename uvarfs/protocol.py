from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from .representation import normalization_modes, normalization_prefix
from .objectives import experiment_branches

EXPERIMENT_VERSION = 'gpu-eval-v14-fixed-support-polish'
BMAD_DATASETS = {'brain', 'liver', 'resc', 'oct2017', 'xray', 'camelyon16'}
RANDOM_FAMILIES = {'asls_random', 'random4_uvarfs', 'randomk_raw', 'randomk_uvarfs'}


def expected_method_names(cfg):
    names = []
    for method in cfg['methods']:
        if method in RANDOM_FAMILIES:
            names.extend(f'{method}_seed{seed}' for seed in cfg.get('random_baseline_seeds',[cfg['seed']]))
        else:
            names.append(method)
    if len(names) != len(set(names)):
        raise ValueError('duplicate methods or random baseline seeds')
    base=names.copy()
    names=[branch['prefix']+name for branch in experiment_branches(cfg) for name in base]
    if len(names)!=len(set(names)):
        raise ValueError('normalization ablation method names collide')
    return sorted(names)


def config_fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def code_fingerprint(root: Path):
    digest=hashlib.sha256()
    for path in sorted((root/'uvarfs').glob('*.py'))+[root/'scripts'/'run_all.py']:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return digest.hexdigest()


def validate_results_version(root: Path):
    """Refuse cross-version overwrites, even with force/resume disabled."""
    # A returned run may have finished fitting without any final metric CSV.
    # Its recorded version still protects that historical directory.
    for path in root.glob('*/run_metadata.json'):
        try:
            metadata=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError) as error:
            raise ValueError(f'Cannot establish results version at {path}; use a fresh version directory.') from error
        if not isinstance(metadata,dict) or metadata.get('experiment_version')!=EXPERIMENT_VERSION:
            raise ValueError(f'Results version differs at {path}; set results_dir to a fresh version directory. Historical results must remain intact.')
    import pandas as pd
    paths=[root/'all_metrics.csv',root/'all_metrics.partial.csv']
    paths.extend(root/name/'metrics.csv' for name in BMAD_DATASETS)
    for path in paths:
        if not path.is_file():
            continue
        try:
            frame=pd.read_csv(path)
        except (pd.errors.EmptyDataError,pd.errors.ParserError):
            continue  # Incomplete same-directory files can be rerun.
        if len(frame) and ('experiment_version' not in frame or
                          not frame.experiment_version.eq(EXPERIMENT_VERSION).all()):
            raise ValueError(f'Results version differs at {path}; set results_dir to a fresh version directory. Historical results must remain intact.')


def data_fingerprint(train, test):
    """Detect changes to sampled protocol inputs without hashing full archives.

    File identity, size and nanosecond mtime cover all train/test images and
    evaluation masks. Masks only participate in provenance, never fitting.
    """
    digest=hashlib.sha256()
    for sample in [*train,*test]:
        digest.update(json.dumps([sample.dataset,sample.split,sample.label]).encode())
        for path in (sample.image,sample.mask):
            if path is None:
                digest.update(b'null')
            else:
                stat=path.stat()
                digest.update(json.dumps([str(path.resolve()),stat.st_size,stat.st_mtime_ns]).encode())
    return digest.hexdigest()


def completed_result_compatible(frame, metadata, progress, cfg, code_hash, pixel_required, data_hash):
    if not isinstance(metadata,dict) or not isinstance(progress,dict):
        return False
    if frame.empty or not {'method','experiment_version','image_auroc','image_auprc'} <= set(frame.columns):
        return False
    methods=expected_method_names(cfg)
    if frame['method'].duplicated().any() or sorted(frame['method']) != methods:
        return False
    if not frame['experiment_version'].astype(str).eq(EXPERIMENT_VERSION).all():
        return False
    if metadata.get('config_fingerprint') != config_fingerprint(cfg) or metadata.get('code_fingerprint') != code_hash:
        return False
    if metadata.get('data_fingerprint')!=data_hash:
        return False
    if progress.get('stage')!='complete' or progress.get('methods')!=len(methods):
        return False
    required=['image_auroc','image_auprc']+(['pixel_auroc','pixel_auprc','aupro'] if pixel_required else [])
    return all(column in frame and frame[column].map(
        lambda value:isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1
    ).all() for column in required)
