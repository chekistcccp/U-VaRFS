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
