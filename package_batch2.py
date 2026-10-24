"""Package the exact local feature implementation in a single Kaggle script."""
from pathlib import Path
import py_compile

root=Path(__file__).resolve().parent
source=(root/'train_baselines.py').read_text()
features=(root/'features.py').read_text()
prefix='''import sys, types, subprocess
from pathlib import Path
subprocess.run(['nvidia-smi'],check=True)
import xgboost as xgb
print('xgboost',xgb.__version__,flush=True)
smoke=xgb.XGBClassifier(n_estimators=2,device='cuda',tree_method='hist')
smoke.fit([[0],[1],[2],[3]],[0,0,1,1])
assert 'cuda' in smoke.get_booster().save_config(), 'GPU training not activated'
print('GPU training smoke test passed',flush=True)
data=next(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv')).parent
original=next(Path('/kaggle/input').rglob('binary-aviation-satisfaction-129k/data.csv'))
sys.argv=['train.py','--data',str(data),'--original',str(original),'--models','xgboost','--gpu','--features','rich','--rounds','3500']
'''
prefix += "module=types.ModuleType('features')\nsys.modules['features']=module\nexec("+repr(features)+",module.__dict__)\n"
p=root/'kaggle-batch2/train.py'
p.write_text(prefix+source)
py_compile.compile(str(p),doraise=True)
print('Packaged and syntax-checked GPU script:',p)
