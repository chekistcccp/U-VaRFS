from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from uvarfs.preprocess import prepare_all


def _img(path: Path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), value, np.uint8)).save(path)


def _zip_dir(src: Path, archive: Path, arc_prefix: str = ''):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
        for p in sorted(src.rglob('*')):
            if p.is_file():
                rel = p.relative_to(src)
                name = str(Path(arc_prefix) / rel) if arc_prefix else str(rel)
                z.write(p, name)


def test_oct2017_archive_to_bmad(tmp_path):
    payload = tmp_path / 'payload' / 'OCT2017'
    for split in ('train', 'test'):
        for cls in ('NORMAL', 'CNV', 'DME', 'DRUSEN'):
            n = 3 if split == 'train' else 10
            for i in range(n):
                _img(payload / split / cls / f'{cls}_{i}.png', i)

    data_root = tmp_path / 'data'
    _zip_dir(payload, data_root / 'archives' / 'oct2017.zip', 'OCT2017')

    out = data_root / 'processed' / 'BMAD'
    results = prepare_all(data_root, out, tmp_path / 'metadata', datasets=['oct2017'])
    r = next(x for x in results if x.dataset == 'OCT2017')

    assert r.status == 'processed'
    assert r.train == 3
    assert r.valid == 32
    assert r.test == 8
    assert (data_root / '_extracted').exists()
    assert (out / 'OCT2017' / 'preprocess_manifest.json').exists()


def test_resc_nested_archive_to_bmad(tmp_path):
    payload = tmp_path / 'payload' / 'RESC'
    for i in range(4):
        _img(payload / 'train' / 'images' / 'c1' / f'{i}.png', i)
    for i in range(70):
        _img(payload / 'test' / 'normal_images' / 'c1' / f'{i}.png', i)
    for i in range(55):
        _img(payload / 'test' / 'images' / 'c1' / f'{i}.png', i)
        _img(payload / 'test' / 'lesion_mask' / 'c1' / f'{i}.png', 255)

    inner = tmp_path / 'resc_inner.zip'
    _zip_dir(payload, inner, 'RESC')

    data_root = tmp_path / 'data'
    outer = data_root / 'archives' / 'resc_download.zip'
    outer.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer, 'w', compression=zipfile.ZIP_STORED) as z:
        z.write(inner, 'nested/resc_inner.zip')

    out = data_root / 'processed' / 'BMAD'
    results = prepare_all(data_root, out, tmp_path / 'metadata', datasets=['resc'])
    r = next(x for x in results if x.dataset == 'RESC')

    assert r.status == 'processed'
    assert r.train == 4
    assert r.valid == 115
    assert r.test == 10
    assert (out / 'RESC' / 'test' / 'Ungood' / 'anomaly_mask').exists()


def test_empty_archive_directory_reports_missing(tmp_path):
    data_root = tmp_path / 'data'
    out = data_root / 'processed' / 'BMAD'
    results = prepare_all(data_root, out, tmp_path / 'metadata', datasets=['oct2017'])
    assert len(results) == 1
    assert results[0].status == 'missing'
    assert 'archives' in results[0].message


def test_replaced_archive_prunes_stale_cache(tmp_path):
    data_root = tmp_path / 'data'
    archive = data_root / 'archives' / 'oct.zip'

    payload1 = tmp_path / 'p1' / 'OCT2017'
    _img(payload1 / 'train' / 'NORMAL' / 'a.png', 1)
    _img(payload1 / 'test' / 'NORMAL' / 'a.png', 1)
    for cls in ('CNV', 'DME', 'DRUSEN'):
        _img(payload1 / 'test' / cls / 'a.png', 1)
    _zip_dir(payload1, archive, 'OCT2017')

    out = data_root / 'processed' / 'BMAD'
    prepare_all(data_root, out, tmp_path / 'metadata', datasets=['oct2017'])
    first = [p for p in (data_root / '_extracted').iterdir() if p.is_dir() and (p / '.uvarfs_extracted.json').exists()]
    assert len(first) == 1

    payload2 = tmp_path / 'p2' / 'OCT2017'
    _img(payload2 / 'train' / 'NORMAL' / 'a.png', 2)
    _img(payload2 / 'train' / 'NORMAL' / 'b.png', 2)
    _img(payload2 / 'test' / 'NORMAL' / 'a.png', 2)
    for cls in ('CNV', 'DME', 'DRUSEN'):
        _img(payload2 / 'test' / cls / 'a.png', 2)
    _zip_dir(payload2, archive, 'OCT2017')

    prepare_all(data_root, out, tmp_path / 'metadata', datasets=['oct2017'], force=True)
    second = [p for p in (data_root / '_extracted').iterdir() if p.is_dir() and (p / '.uvarfs_extracted.json').exists()]
    assert len(second) == 1
    assert second[0] != first[0]


def test_released_liver_hist_diy_archive(tmp_path):
    payload = tmp_path / 'payload' / 'Liver' / 'Train' / 'hist_DIY'

    for i in range(6):
        _img(payload / 'train' / 'good' / f'train_{i}.png', i)

    for split in ('valid', 'test'):
        for i in range(3):
            _img(payload / split / 'img' / 'good' / f'g_{i}.png', i)
        for i in range(4):
            _img(payload / split / 'img' / 'Ungood' / f'b_{i}.png', i)
            _img(payload / split / 'label' / 'Ungood' / f'b_{i}.png', 255)

    data_root = tmp_path / 'data'
    _zip_dir(payload, data_root / 'archives' / 'Liver_AD.zip', 'Liver/Train/hist_DIY')

    out = data_root / 'processed' / 'BMAD'
    results = prepare_all(data_root, out, tmp_path / 'metadata', datasets=['liver'])
    r = next(x for x in results if x.dataset == 'liver')

    assert r.status == 'reused'
    assert r.train == 6
    assert r.valid == 7
    assert r.test == 7
    assert len(list((out / 'liver' / 'test' / 'Ungood' / 'anomaly_mask').glob('*.png'))) == 4


def test_released_val_alias_is_normalized_to_valid(tmp_path):
    payload = tmp_path / 'payload' / 'Chest-RSNA'
    for i in range(5):
        _img(payload / 'train' / 'good' / f't_{i}.png', i)
    for i in range(2):
        _img(payload / 'val' / 'good' / f'vg_{i}.png', i)
        _img(payload / 'val' / 'Ungood' / f'vb_{i}.png', i)
        _img(payload / 'test' / 'good' / f'tg_{i}.png', i)
        _img(payload / 'test' / 'Ungood' / f'tb_{i}.png', i)

    data_root = tmp_path / 'data'
    _zip_dir(payload, data_root / 'archives' / 'Chest-AD.zip', 'Chest-RSNA')

    out = data_root / 'processed' / 'BMAD'
    results = prepare_all(data_root, out, tmp_path / 'metadata', datasets=['xray'])
    r = next(x for x in results if x.dataset == 'xray')

    assert r.status == 'reused'
    assert r.train == 5
    assert r.valid == 4
    assert r.test == 4
    assert (out / 'xray' / 'valid' / 'good' / 'img').is_dir()
