"""Bundle source-only auxiliary rating models for private Kaggle CUDA execution."""
from pathlib import Path
import json
import py_compile

root=Path(__file__).resolve().parent
out=root/'kaggle-original-auxiliary'
out.mkdir(exist_ok=True)
sources={name:(root/name).read_text() for name in ('external_teacher.py','external_auxiliary.py')}
script='''import subprocess,sys
from pathlib import Path
subprocess.run(['nvidia-smi'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','xgboost==3.2.0'],check=True)
code=Path('/kaggle/working/reproducible-source');code.mkdir()
'''
script+='sources='+repr(sources)+'\n'
script+='''for name,source in sources.items():
    (code/name).write_text(source)
data=next(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv')).parent
original=next(Path('/kaggle/input').rglob('binary-aviation-satisfaction-129k/data.csv'))
subprocess.run([sys.executable,str(code/'external_auxiliary.py'),'--data',str(data),'--original',str(original),'--output','/kaggle/working/original-auxiliary','--device','cuda','--rounds','400'],check=True)
print('All 13 original-only auxiliary CUDA models completed',flush=True)
'''
(out/'train.py').write_text(script)
py_compile.compile(str(out/'train.py'),doraise=True)
meta={
    'id':'wasidyms/s6e10-batch5-original-auxiliary',
    'title':'S6E10 Batch5 Original Auxiliary',
    'code_file':'train.py','language':'python','kernel_type':'script',
    'is_private':True,'enable_gpu':True,'enable_internet':True,
    'machine_shape':'NvidiaTeslaT4',
    'dataset_sources':['arseniyshutko/binary-aviation-satisfaction-129k'],
    'competition_sources':['playground-series-s6e10'],
    'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(out)
