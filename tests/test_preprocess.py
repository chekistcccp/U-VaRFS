from pathlib import Path
import numpy as np
from PIL import Image
from uvarfs.preprocess import prepare_oct2017, prepare_resc


def _img(path: Path, value=1):
    path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(np.full((8,8),value,np.uint8)).save(path)


def test_oct2017_raw_to_bmad(tmp_path):
    root=tmp_path/'OCT2017'
    for split in ('train','test'):
        for cls in ('NORMAL','CNV','DME','DRUSEN'):
            n=3 if split=='train' else 10
            for i in range(n): _img(root/split/cls/f'{cls}_{i}.png',i)
    r=prepare_oct2017(tmp_path,tmp_path/'out')
    assert r.train==3 and r.valid==32 and r.test==8


def test_resc_raw_to_bmad(tmp_path):
    root=tmp_path/'RESC'
    for i in range(4): _img(root/'train/images/c1'/f'{i}.png',i)
    for i in range(70): _img(root/'test/normal_images/c1'/f'{i}.png',i)
    for i in range(55):
        _img(root/'test/images/c1'/f'{i}.png',i); _img(root/'test/lesion_mask/c1'/f'{i}.png',255)
    r=prepare_resc(tmp_path,tmp_path/'out')
    assert r.train==4 and r.valid==115 and r.test==10
