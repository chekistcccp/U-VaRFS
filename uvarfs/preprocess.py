from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
ARCHIVE_EXTS = ('.zip', '.tar', '.tar.gz', '.tgz', '.tar.xz', '.txz', '.7z')


@dataclass
class PrepResult:
    dataset: str
    status: str
    train: int = 0
    valid: int = 0
    test: int = 0
    message: str = ''


def _safe_rel(member: str) -> bool:
    p = Path(member)
    return not p.is_absolute() and '..' not in p.parts


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(ext) for ext in ARCHIVE_EXTS)


def _archive_stem(path: Path) -> str:
    name = path.name
    for suffix in sorted(ARCHIVE_EXTS, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)] or 'archive'
    return path.stem


def _extract_one_archive(archive: Path, dst_root: Path) -> Path:
    stat = archive.stat()
    fingerprint = f'{archive.resolve()}|{stat.st_size}|{stat.st_mtime_ns}'
    tag = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
    out = dst_root / f'{_archive_stem(archive)}_{tag}'
    marker = out / '.uvarfs_extracted.json'
    if marker.exists():
        return out

    tmp = dst_root / f'.tmp_{_archive_stem(archive)}_{tag}'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        low = archive.name.lower()
        if low.endswith('.zip'):
            with zipfile.ZipFile(archive) as z:
                bad = [m.filename for m in z.infolist() if not _safe_rel(m.filename)]
                if bad:
                    raise RuntimeError(f'unsafe paths in {archive}: {bad[:3]}')
                z.extractall(tmp)
        elif low.endswith(('.tar', '.tar.gz', '.tgz', '.tar.xz', '.txz')):
            with tarfile.open(archive) as t:
                members = t.getmembers()
                bad = [m.name for m in members if (not _safe_rel(m.name)) or m.issym() or m.islnk()]
                if bad:
                    raise RuntimeError(f'unsafe paths/links in {archive}: {bad[:3]}')
                t.extractall(tmp)
        elif low.endswith('.7z'):
            try:
                import py7zr
            except ImportError as e:
                raise RuntimeError('py7zr is required because a .7z archive was found') from e
            with py7zr.SevenZipFile(archive, mode='r') as z:
                names = z.getnames()
                bad = [name for name in names if not _safe_rel(name)]
                if bad:
                    raise RuntimeError(f'unsafe paths in {archive}: {bad[:3]}')
                z.extractall(tmp)
        else:
            raise RuntimeError(f'unsupported archive type: {archive}')

        marker_payload = {
            'source': str(archive.resolve()),
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        }
        (tmp / '.uvarfs_extracted.json').write_text(
            json.dumps(marker_payload, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
        return out
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def extract_archives(data_root: Path, workers: int = 4, max_depth: int = 5) -> list[Path]:
    """Extract all user-provided archives from data/archives into data/_extracted.

    The archive directory is the *only* raw-data entry point. Extraction is cached
    using path + size + mtime and supports nested archives for up to max_depth
    rounds. Original archives are never modified or deleted.
    """
    del workers  # reserved for future parallel archive extraction
    data_root = Path(data_root)
    archive_root = data_root / 'archives'
    dst_root = data_root / '_extracted'
    archive_root.mkdir(parents=True, exist_ok=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    queue: list[tuple[Path, int]] = [(p, 0) for p in sorted(archive_root.rglob('*')) if _is_archive(p)]
    outputs: list[Path] = []
    seen: set[str] = set()

    if not queue:
        cached = [p for p in dst_root.iterdir() if p.is_dir() and (p / '.uvarfs_extracted.json').exists()]
        if cached:
            print(f'[extract] no archives currently in {archive_root}; reusing {len(cached)} cached extraction roots')
            return sorted(cached)
        print(f'[extract] no supported archives found in fixed input directory: {archive_root}')
        return []

    print(f'[extract] found {len(queue)} top-level archive(s) in {archive_root}')
    while queue:
        archive, depth = queue.pop(0)
        try:
            key = str(archive.resolve())
        except OSError:
            key = str(archive)
        if key in seen:
            continue
        seen.add(key)
        if depth > max_depth:
            raise RuntimeError(f'nested archive depth exceeded {max_depth}: {archive}')

        out = _extract_one_archive(archive, dst_root)
        outputs.append(out)
        print(f'[extract] depth={depth} {archive.name} -> {out.name}')

        if depth < max_depth:
            nested = [p for p in sorted(out.rglob('*')) if _is_archive(p)]
            queue.extend((p, depth + 1) for p in nested)

    return sorted(set(outputs))


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _save_gray(arr: np.ndarray, dst: Path) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        u8 = np.zeros(arr.shape, np.uint8)
    else:
        vals = arr[finite]
        lo, hi = np.percentile(vals, [0.5, 99.5])
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            u8 = np.zeros(arr.shape, np.uint8)
        else:
            u8 = np.clip((arr - lo) / (hi - lo), 0, 1)
            u8 = (u8 * 255).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8, mode='L').save(dst)


def _save_mask(arr: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(arr) > 0).astype(np.uint8) * 255, mode='L').save(dst)


def _hist_equalization(img_array: np.ndarray) -> np.ndarray:
    img = np.asarray(img_array, dtype=np.uint8)
    hist = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
    hist[0] = 0
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    total = hist.sum()
    if total <= 0:
        return img
    hist /= total
    cdf = np.cumsum(hist)
    transform = np.floor(255 * cdf).astype(np.uint8)
    mid = (transform > 0) & (transform < 150)
    transform[mid] = np.floor(transform[mid].astype(np.float32) / 150 * 120 + 30).astype(np.uint8)
    return transform[img]


def _count_split(root: Path, split: str) -> int:
    p = root / split
    if not p.exists():
        return 0
    count = 0
    for cls in ('good', 'Ungood', 'ungood', 'bad', 'normal'):
        croot = p / cls
        if not croot.exists():
            continue
        iroot = croot / 'img' if (croot / 'img').is_dir() else croot
        count += sum(1 for x in iroot.rglob('*') if x.is_file() and x.suffix.lower() in IMG_EXTS and 'anomaly_mask' not in x.parts and 'label' not in x.parts)
    return count


def _looks_processed_dataset(root: Path) -> bool:
    return (root / 'train').is_dir() and (root / 'test').is_dir() and any(
        (root / 'train' / name).exists() for name in ('good', 'normal')
    )


def normalize_existing_processed(raw_root: Path, out_root: Path) -> list[PrepResult]:
    aliases = {
        'brain': 'Brain', 'brats': 'Brain', 'liver': 'liver', 'resc': 'RESC',
        'oct2017': 'OCT2017', 'rsna': 'xray', 'xray': 'xray', 'chest': 'xray',
        'camelyon16': 'camelyon16', 'camelyon': 'camelyon16',
    }
    results = []
    for p in sorted(raw_root.rglob('*')):
        if not p.is_dir() or out_root in p.parents or not _looks_processed_dataset(p):
            continue
        lname = p.name.lower().replace('-', '').replace('_', '')
        canonical = None
        for key, val in aliases.items():
            if key.replace('_', '') in lname:
                canonical = val
                break
        if not canonical:
            continue
        dst = out_root / canonical
        if dst.exists() and any(dst.rglob('*.png')):
            continue
        for f in p.rglob('*'):
            if f.is_file():
                rel = f.relative_to(p)
                _link_or_copy(f, dst / rel)
        results.append(PrepResult(canonical, 'reused', _count_split(dst,'train'), _count_split(dst,'valid'), _count_split(dst,'test'), f'normalized from {p}'))
    return results


def _nifti_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob('*') if p.is_file() and (p.name.endswith('.nii') or p.name.endswith('.nii.gz'))])


def _case_id_from_brats(path: Path) -> str:
    m = re.search(r'BraTS2021_(\d+)_flair', path.name)
    if m:
        return m.group(1)
    name = path.name.replace('.nii.gz','').replace('.nii','')
    name = re.sub(r'(_|-)?flair$', '', name, flags=re.I)
    m = re.search(r'(\d{3,})', name)
    return m.group(1) if m else name


def prepare_brain(raw_root: Path, out_root: Path, metadata_root: Path, force: bool=False) -> PrepResult:
    dst = out_root / 'Brain'
    if not force and (dst/'train').exists() and _count_split(dst,'train') > 100:
        return PrepResult('Brain','ready',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))
    flairs = [p for p in _nifti_files(raw_root) if re.search(r'flair\.nii(\.gz)?$', p.name, re.I)]
    cases = {}
    for f in flairs:
        seg_name = re.sub(r'flair(?=\.nii(?:\.gz)?$)', 'seg', f.name, flags=re.I)
        seg = f.with_name(seg_name)
        if seg.exists():
            cases[_case_id_from_brats(f)] = (f, seg)
    if len(cases) < 100:
        return PrepResult('Brain','missing',message='BraTS flair/seg NIfTI pairs not found')
    import nibabel as nib
    split_meta = json.loads((metadata_root/'brats_splits.json').read_text(encoding='utf-8'))
    available = set(cases)
    official_hits = sum(x in available for x in split_meta['train_id'])
    if official_hits >= min(300, len(split_meta['train_id'])):
        train_ids = [x for x in split_meta['train_id'] if x in available]
        valid_norm = [x for x in split_meta['valid_normal_id'] if x in available]
        test_norm = [x for x in split_meta['test_normal_id'] if x in available]
    else:
        ids = sorted(available)
        train_ids, valid_norm, test_norm = ids[:424], ids[424:463], ids[463:625]
    abnormal = sorted(available - set(train_ids) - set(valid_norm) - set(test_norm))
    valid_abn, test_abn = abnormal[:11], abnormal[11:]
    if force and dst.exists(): shutil.rmtree(dst)

    def emit(ids: Sequence[str], split: str, abnormal_mode: bool, step: int):
        for cid in ids:
            flair_p, seg_p = cases[cid]
            flair = nib.load(str(flair_p)).get_fdata(dtype=np.float32)
            seg = nib.load(str(seg_p)).get_fdata(dtype=np.float32)
            maxz = min(flair.shape[-1], seg.shape[-1])
            for z in range(60, min(100,maxz), step):
                has = bool(np.max(seg[:,:,z]) >= 1)
                if not abnormal_mode and has:
                    continue
                cls = 'Ungood' if abnormal_mode else 'good'
                name = f'{cid}_{z}.png'
                _save_gray(flair[:,:,z], dst/split/cls/'img'/name)
                if abnormal_mode:
                    _save_mask(seg[:,:,z], dst/split/cls/'anomaly_mask'/name)
    emit(train_ids, 'train', False, 1)
    emit(valid_norm, 'valid', False, 1)
    emit(valid_abn, 'valid', True, 10)
    emit(test_norm, 'test', False, 1)
    emit(test_abn, 'test', True, 8)
    manifest = {'source':'BraTS2021','train_ids':train_ids,'valid_normal_ids':valid_norm,'valid_abnormal_ids':valid_abn,'test_normal_ids':test_norm,'test_abnormal_ids':test_abn}
    (dst/'preprocess_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return PrepResult('Brain','processed',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))


def _find_lits_pairs(raw_root: Path) -> list[tuple[Path,Path]]:
    pairs=[]
    vols=[p for p in _nifti_files(raw_root) if re.match(r'volume-\d+\.nii(?:\.gz)?$',p.name,re.I)]
    for v in vols:
        s=v.with_name(re.sub(r'^volume-', 'segmentation-', v.name, flags=re.I))
        if s.exists(): pairs.append((v,s))
    return pairs


def _find_btcv_pairs(raw_root: Path) -> list[tuple[Path,Path]]:
    pairs=[]
    imgs=[p for p in _nifti_files(raw_root) if re.match(r'img\d+\.nii(?:\.gz)?$',p.name,re.I)]
    alln=_nifti_files(raw_root)
    lookup={p.name:p for p in alln}
    for im in imgs:
        label_name=re.sub(r'^img','label',im.name,flags=re.I)
        candidates=[im.with_name(label_name), im.parent.parent/'label'/label_name, im.parent.parent/'labels'/label_name]
        lab=next((x for x in candidates if x.exists()),None) or lookup.get(label_name)
        if lab: pairs.append((im,lab))
    return pairs


def prepare_liver(raw_root: Path, out_root: Path, force: bool=False) -> PrepResult:
    dst=out_root/'liver'
    if not force and _count_split(dst,'train')>100:
        return PrepResult('liver','ready',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))
    atlas=_find_btcv_pairs(raw_root); lits=_find_lits_pairs(raw_root)
    if not atlas or not lits:
        return PrepResult('liver','missing',message=f'need BTCV/ATLAS and LiTS NIfTI pairs; found atlas={len(atlas)}, lits={len(lits)}')
    import nibabel as nib
    if force and dst.exists(): shutil.rmtree(dst)
    train_count=0
    for im_p,lab_p in sorted(atlas):
        im=nib.load(str(im_p)).get_fdata(dtype=np.float32); lab=nib.load(str(lab_p)).get_fdata(dtype=np.float32)
        zmax=min(im.shape[-1],lab.shape[-1]); case=re.sub(r'\D','',im_p.stem) or im_p.stem
        for z in range(zmax):
            liver=(lab[:,:,z]==6)
            if not liver.any(): continue
            # BMAD's liver preprocessing masks the liver and applies histogram equalization.
            sl=_to_u8_window(im[:,:,z], None)
            sl=np.flipud(sl*liver.astype(np.uint8)); sl=_hist_equalization(sl)
            out=dst/'train'/'good'/'img'/f'atlas_{case}_{z}.png'; out.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(sl).save(out); train_count+=1
    rows=[]
    for im_p,seg_p in sorted(lits):
        im=nib.load(str(im_p)).get_fdata(dtype=np.float32); seg=nib.load(str(seg_p)).get_fdata(dtype=np.float32)
        zmax=min(im.shape[-1],seg.shape[-1]); case=re.search(r'(\d+)',im_p.name).group(1)
        for z in range(zmax):
            lab=seg[:,:,z]; liver=lab>0
            if not liver.any(): continue
            tumor=lab==2
            sl=_to_u8_window(im[:,:,z],(-200,250)); sl=np.flipud(sl*liver.astype(np.uint8)); sl=_hist_equalization(sl)
            tumor=np.flipud(tumor)
            rows.append((f'lits_{case}_{z}.png',sl,tumor,bool(tumor.any())))
    good=[r for r in rows if not r[3]]; bad=[r for r in rows if r[3]]
    # Published BMAD counts: validation 93 normal + 73 abnormal, remaining LiTS slices test.
    valid_good, test_good = good[:93], good[93:]
    valid_bad, test_bad = bad[:73], bad[73:]
    def emit(records,split,cls):
        for name,sl,mask,_ in records:
            p=dst/split/cls/'img'/name; p.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(sl).save(p)
            if cls=='Ungood': _save_mask(mask,dst/split/cls/'anomaly_mask'/name)
    emit(valid_good,'valid','good'); emit(valid_bad,'valid','Ungood'); emit(test_good,'test','good'); emit(test_bad,'test','Ungood')
    manifest={'source':'BTCV/ATLAS + LiTS','atlas_pairs':len(atlas),'lits_pairs':len(lits),'lits_good':len(good),'lits_bad':len(bad),'validation_rule':'first 93 normal + first 73 abnormal after deterministic filename/slice sort'}
    (dst/'preprocess_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return PrepResult('liver','processed',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))


def _to_u8_window(arr: np.ndarray, window: tuple[float,float]|None) -> np.ndarray:
    x=np.asarray(arr,dtype=np.float32)
    if window:
        lo,hi=window; x=np.clip(x,lo,hi)
    else:
        finite=x[np.isfinite(x)]
        lo,hi=(np.percentile(finite,[0.5,99.5]) if finite.size else (0,1))
    if hi<=lo: return np.zeros(x.shape,np.uint8)
    return (np.clip((x-lo)/(hi-lo),0,1)*255).astype(np.uint8)


def _collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _find_named_dir(raw_root: Path, parts: Sequence[str]) -> Path|None:
    target=[x.lower() for x in parts]
    for p in raw_root.rglob('*'):
        if not p.is_dir(): continue
        names=[x.lower() for x in p.parts]
        if len(names)>=len(target) and names[-len(target):]==target:
            return p
    return None


def prepare_resc(raw_root: Path, out_root: Path, force: bool=False) -> PrepResult:
    dst=out_root/'RESC'
    if not force and _count_split(dst,'train')>100:
        return PrepResult('RESC','ready',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))
    train_images=_find_named_dir(raw_root,['train','images'])
    test_abn=_find_named_dir(raw_root,['test','images'])
    test_norm=_find_named_dir(raw_root,['test','normal_images'])
    lesion=_find_named_dir(raw_root,['test','lesion_mask'])
    if not all([train_images,test_abn,test_norm,lesion]):
        return PrepResult('RESC','missing',message='expected RESC train/images, test/images, test/normal_images, test/lesion_mask')
    if force and dst.exists(): shutil.rmtree(dst)
    train=_collect_images(train_images); norms=_collect_images(test_norm); abns=_collect_images(test_abn)
    for i,src in enumerate(train): _link_or_copy(src,dst/'train'/'good'/'img'/f'{i:06d}{src.suffix.lower()}')
    # BMAD validation has 115 samples out of 1920 RESC evaluation images. Preserve class ratio deterministically.
    nvg=min(65,len(norms)); nvb=min(50,len(abns)); valid_norm,test_norms=norms[:nvg],norms[nvg:]; valid_abn,test_abns=abns[:nvb],abns[nvb:]
    mask_index={str(p.relative_to(lesion)):p for p in _collect_images(lesion)}
    mask_by_name={p.name:p for p in _collect_images(lesion)}
    def emit_normal(seq,split):
        for i,src in enumerate(seq): _link_or_copy(src,dst/split/'good'/'img'/f'{i:06d}{src.suffix.lower()}')
    def emit_bad(seq,split):
        for i,src in enumerate(seq):
            name=f'{i:06d}{src.suffix.lower()}'; _link_or_copy(src,dst/split/'Ungood'/'img'/name)
            try: rel=str(src.relative_to(test_abn))
            except ValueError: rel=src.name
            m=mask_index.get(rel) or mask_by_name.get(src.name)
            if m: _link_or_copy(m,dst/split/'Ungood'/'anomaly_mask'/name)
    emit_normal(valid_norm,'valid'); emit_bad(valid_abn,'valid'); emit_normal(test_norms,'test'); emit_bad(test_abns,'test')
    (dst/'preprocess_manifest.json').write_text(json.dumps({'source':'RESC','valid_normal':nvg,'valid_abnormal':nvb,'rule':'sorted deterministic split'},indent=2),encoding='utf-8')
    return PrepResult('RESC','processed',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))


def _find_oct_root(raw_root: Path) -> Path|None:
    for p in raw_root.rglob('*'):
        if p.is_dir() and (p/'train'/'NORMAL').is_dir() and (p/'test'/'NORMAL').is_dir(): return p
    return None


def prepare_oct2017(raw_root: Path, out_root: Path, force: bool=False) -> PrepResult:
    dst=out_root/'OCT2017'
    if not force and _count_split(dst,'train')>100:
        return PrepResult('OCT2017','ready',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))
    root=_find_oct_root(raw_root)
    if not root: return PrepResult('OCT2017','missing',message='expected OCT2017 train/test with NORMAL,CNV,DME,DRUSEN')
    if force and dst.exists(): shutil.rmtree(dst)
    normal_train=_collect_images(root/'train'/'NORMAL')
    for i,src in enumerate(normal_train): _link_or_copy(src,dst/'train'/'good'/'img'/f'{i:06d}{src.suffix.lower()}')
    normal_test=_collect_images(root/'test'/'NORMAL'); diseases={k:_collect_images(root/'test'/k) for k in ('CNV','DME','DRUSEN')}
    # BMAD: val = 8 normal + 24 abnormal (8 from each disease); test = remaining 242 + 726 for standard 1000-image OCT test set.
    vg,tg=normal_test[:8],normal_test[8:]; vb=[]; tb=[]
    for k in ('CNV','DME','DRUSEN'): vb+=diseases[k][:8]; tb+=diseases[k][8:]
    for split,cls,seq in [('valid','good',vg),('valid','Ungood',vb),('test','good',tg),('test','Ungood',tb)]:
        for i,src in enumerate(seq): _link_or_copy(src,dst/split/cls/'img'/f'{i:06d}{src.suffix.lower()}')
    (dst/'preprocess_manifest.json').write_text(json.dumps({'source':'OCT2017/Kermany','validation':'8 NORMAL + 8 each CNV/DME/DRUSEN'},indent=2),encoding='utf-8')
    return PrepResult('OCT2017','processed',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))


def _dicom_worker(args):
    src,dst=args
    import pydicom
    ds=pydicom.dcmread(str(src)); arr=ds.pixel_array.astype(np.float32)
    if getattr(ds,'PhotometricInterpretation','')=='MONOCHROME1': arr=arr.max()-arr
    u8=_to_u8_window(arr,None); dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(u8).save(dst); return 1


def prepare_xray(raw_root: Path, out_root: Path, force: bool=False, workers: int=8) -> PrepResult:
    dst=out_root/'xray'
    if not force and _count_split(dst,'train')>100:
        return PrepResult('xray','ready',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))
    csvs=[p for p in raw_root.rglob('stage_2_detailed_class_info.csv')]
    if not csvs: return PrepResult('xray','missing',message='stage_2_detailed_class_info.csv not found')
    meta=csvs[0]; dcm_dirs=[p for p in raw_root.rglob('stage_2_train_images') if p.is_dir()]
    if not dcm_dirs: return PrepResult('xray','missing',message='stage_2_train_images not found')
    dcm_dir=dcm_dirs[0]
    labels={}
    with open(meta,newline='',encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row or row[0].lower() in ('patientid','patient id'): continue
            labels[row[0]]=row[1]
    normal=sorted([pid for pid,c in labels.items() if c=='Normal' and (dcm_dir/f'{pid}.dcm').exists()])
    abnormal=sorted([pid for pid,c in labels.items() if c!='Normal' and (dcm_dir/f'{pid}.dcm').exists()])
    if len(normal)<8000: return PrepResult('xray','missing',message=f'only {len(normal)} normal RSNA DICOMs found; expected >=8000')
    if force and dst.exists(): shutil.rmtree(dst)
    train=normal[:8000]; rem_n=normal[8000:]; rem_a=abnormal
    total=len(rem_n)+len(rem_a); val_total=min(1490,total)
    val_n=int(round(val_total*len(rem_n)/max(total,1))); val_n=min(val_n,len(rem_n)); val_a=min(val_total-val_n,len(rem_a))
    valid_n,test_n=rem_n[:val_n],rem_n[val_n:]; valid_a,test_a=rem_a[:val_a],rem_a[val_a:]
    jobs=[]
    for split,cls,ids in [('train','good',train),('valid','good',valid_n),('valid','Ungood',valid_a),('test','good',test_n),('test','Ungood',test_a)]:
        for pid in ids: jobs.append((dcm_dir/f'{pid}.dcm', dst/split/cls/'img'/f'{pid}.png'))
    with ProcessPoolExecutor(max_workers=max(1,workers)) as ex:
        fut=[ex.submit(_dicom_worker,j) for j in jobs]
        for f in as_completed(fut): f.result()
    manifest={'source':'RSNA stage 2','train_normal':len(train),'valid_normal':len(valid_n),'valid_abnormal':len(valid_a),'test_normal':len(test_n),'test_abnormal':len(test_a),'rule':'sorted deterministic; 8000 normal train; 1490 stratified validation from remainder'}
    (dst/'preprocess_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return PrepResult('xray','processed',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))


def _read_coords(path: Path) -> dict[str,list[tuple[int,int,int]]]:
    out={}
    for i,line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        if not line.strip(): continue
        pid,x,y=line.split(','); out.setdefault(pid,[]).append((i,int(x),int(y)))
    return out


def _camelyon_slide_task(args):
    slide_path, items, outdir, patch_size, level = args
    import openslide
    slide=openslide.OpenSlide(str(slide_path)); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    for idx,x_center,y_center in items:
        x=int(x_center-patch_size/2); y=int(y_center-patch_size/2)
        img=slide.read_region((x,y),level,(patch_size,patch_size)).convert('RGB')
        img.save(outdir/f'{idx:06d}.png')
    slide.close(); return len(items)


def prepare_camelyon(raw_root: Path, out_root: Path, metadata_root: Path, force: bool=False, workers: int=8) -> PrepResult:
    dst=out_root/'camelyon16'
    if not force and _count_split(dst,'train')>100:
        return PrepResult('camelyon16','ready',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))
    tifs=[p for p in raw_root.rglob('*') if p.is_file() and p.suffix.lower() in ('.tif','.tiff')]
    if len(tifs)<100: return PrepResult('camelyon16','missing',message=f'Camelyon16 WSI .tif files not found (found {len(tifs)})')
    slides={p.stem.lower():p for p in tifs}
    coord_map={
        'train_good.txt':('train','good'), 'valid_good.txt':('valid','good'), 'valid_bad.txt':('valid','Ungood'),
        'test_good.txt':('test','good'), 'test_bad.txt':('test','Ungood'),
    }
    if force and dst.exists(): shutil.rmtree(dst)
    jobs=[]; missing=[]
    for fn,(split,cls) in coord_map.items():
        coord=metadata_root/'camelyon16'/fn
        groups=_read_coords(coord)
        for pid,items in groups.items():
            slide=slides.get(pid.lower())
            if slide is None:
                # tolerate prefixes/case differences
                slide=next((p for key,p in slides.items() if key==pid.lower() or key.endswith(pid.lower())),None)
            if slide is None: missing.append(pid); continue
            jobs.append((slide,items,dst/split/cls,256,0))
    if missing and len(missing)>10:
        return PrepResult('camelyon16','missing',message=f'{len(set(missing))} coordinate-referenced slides missing; e.g. {sorted(set(missing))[:5]}')
    try:
        import openslide  # noqa
    except Exception as e:
        return PrepResult('camelyon16','missing',message=f'openslide-python/system libopenslide required: {e}')
    with ProcessPoolExecutor(max_workers=max(1,workers)) as ex:
        fut=[ex.submit(_camelyon_slide_task,j) for j in jobs]
        for f in as_completed(fut): f.result()
    (dst/'preprocess_manifest.json').write_text(json.dumps({'source':'Camelyon16','coords':'BMAD official data_processing/histopathology/coords','patch_size':256,'level':0},indent=2),encoding='utf-8')
    return PrepResult('camelyon16','processed',_count_split(dst,'train'),_count_split(dst,'valid'),_count_split(dst,'test'))


def prepare_all(data_root: Path, out_root: Path, metadata_root: Path, datasets: Sequence[str]|None=None, force: bool=False, workers: int=8, extract: bool=True) -> list[PrepResult]:
    data_root=Path(data_root).resolve()
    out_root=Path(out_root).resolve()
    metadata_root=Path(metadata_root).resolve()
    out_root.mkdir(parents=True,exist_ok=True)

    archive_root=data_root/'archives'
    extracted_root=data_root/'_extracted'
    archive_root.mkdir(parents=True,exist_ok=True)
    extracted_root.mkdir(parents=True,exist_ok=True)

    if extract:
        extract_archives(data_root,workers)

    # Intentionally scan ONLY the extraction cache. This guarantees that users can
    # keep every original BMAD download untouched in one fixed directory.
    scan_root=extracted_root
    if not any(p.is_file() for p in scan_root.rglob('*')):
        result=PrepResult('all','missing',message=f'No extracted raw data found. Put BMAD archives in {archive_root} and rerun.')
        (out_root/'preprocess_summary.json').write_text(
            json.dumps([result.__dict__],indent=2,ensure_ascii=False),encoding='utf-8'
        )
        return [result]

    reused=normalize_existing_processed(scan_root,out_root)
    wanted=set(x.lower() for x in datasets) if datasets else None
    funcs=[
        ('brain', lambda:prepare_brain(scan_root,out_root,metadata_root,force)),
        ('liver', lambda:prepare_liver(scan_root,out_root,force)),
        ('resc', lambda:prepare_resc(scan_root,out_root,force)),
        ('oct2017', lambda:prepare_oct2017(scan_root,out_root,force)),
        ('xray', lambda:prepare_xray(scan_root,out_root,force,workers)),
        ('camelyon16', lambda:prepare_camelyon(scan_root,out_root,metadata_root,force,workers)),
    ]
    results=list(reused)
    existing={r.dataset.lower() for r in results}
    for name,fn in funcs:
        if wanted and name not in wanted:
            continue
        if name in existing and not force:
            continue
        results.append(fn())

    summary=[r.__dict__ for r in results]
    (out_root/'preprocess_summary.json').write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8'
    )
    return results

