"""Package a private, second-split RealMLP robustness run. Never submits."""
from pathlib import Path
import json
import py_compile

root=Path(__file__).resolve().parent
out=root/'kaggle-realmlp-confirm'
out.mkdir(exist_ok=True)
source="import sys\nsys.argv.extend(['--seed','2026'])\n"+(root/'train_realmlp.py').read_text()
(out/'train.py').write_text(source)
py_compile.compile(str(out/'train.py'),doraise=True)
meta={
    'id':'wasidyms/s6e10-realmlp-confirm-seed2026',
    'title':'S6E10 RealMLP Confirm Seed2026',
    'code_file':'train.py','language':'python','kernel_type':'script',
    'is_private':True,'enable_gpu':True,'enable_internet':True,
    'machine_shape':'NvidiaTeslaT4','dataset_sources':[],
    'competition_sources':['playground-series-s6e10'],
    'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(out)
