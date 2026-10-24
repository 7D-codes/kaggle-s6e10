"""Bundle exact sources for one teacher-enhanced RealMLP ablation. No submission."""
from pathlib import Path
import json
import py_compile

root=Path(__file__).resolve().parent
out=root/'kaggle-teacher-realmlp'
out.mkdir(exist_ok=True)
sources={name:(root/name).read_text() for name in ('features.py','external_teacher.py','train_baselines.py','train_realmlp.py')}
script='''import subprocess,sys,json
from pathlib import Path
subprocess.run(['nvidia-smi'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','pytabkit==1.7.3','cloudpickle==3.1.2','xgboost==3.2.0'],check=True)
code=Path('/kaggle/working/reproducible-source');code.mkdir()
'''
script+='sources='+repr(sources)+'\n'
script+='''for name,source in sources.items():
    (code/name).write_text(source)
data=next(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv')).parent
original=next(Path('/kaggle/input').rglob('binary-aviation-satisfaction-129k/data.csv'))
teacher=Path('/kaggle/working/original-teacher')
subprocess.run([sys.executable,str(code/'external_teacher.py'),'--data',str(data),'--original',str(original),'--output',str(teacher),'--device','cuda','--rounds','600'],check=True)
print('Original-only CUDA teacher complete',flush=True)
subprocess.run([sys.executable,str(code/'train_realmlp.py'),'--data',str(data),'--output','/kaggle/working','--seed','42','--teacher-csv',str(teacher/'teacher-features.csv')],check=True)
'''
(out/'train.py').write_text(script)
py_compile.compile(str(out/'train.py'),doraise=True)
meta={
    'id':'wasidyms/s6e10-batch4-teacher-realmlp',
    'title':'S6E10 Batch4 Teacher RealMLP',
    'code_file':'train.py','language':'python','kernel_type':'script',
    'is_private':True,'enable_gpu':True,'enable_internet':True,
    'machine_shape':'NvidiaTeslaT4',
    'dataset_sources':['arseniyshutko/binary-aviation-satisfaction-129k'],
    'competition_sources':['playground-series-s6e10'],
    'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(out)
