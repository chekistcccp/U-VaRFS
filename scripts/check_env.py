#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, sys

REQUIRED=['torch','keras','keras_hub','modelscope','numpy','pandas','PIL','yaml','sklearn','scipy','faiss','nibabel','pydicom','cv2']
OPTIONAL=['py7zr','openslide']
missing=[m for m in REQUIRED if importlib.util.find_spec(m) is None]
if sys.version_info < (3,11): missing.append('Python>=3.11')
if missing:
    raise SystemExit('Environment is incomplete. Missing: '+', '.join(missing)+'\nCreate/activate your conda environment and install requirements.txt; run.sh will not install packages.')
print('[env] required packages OK')
for m in OPTIONAL:
    if importlib.util.find_spec(m) is None: print(f'[env] optional package missing: {m} (needed only for .7z or Camelyon16 respectively)')
