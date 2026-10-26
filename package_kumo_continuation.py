"""Build the private bounded Kumo continuation package. Never launches or submits."""
import json
from pathlib import Path
import py_compile

import package_kumo_pair as original


def continuation_caps(source):
    for before, after in (('FIRST_PAIR_SECONDS = 3600', 'FIRST_PAIR_SECONDS = 5400'),
                          ('TOTAL_BUDGET_SECONDS = 28800', 'TOTAL_BUDGET_SECONDS = 24900')):
        if source.count(before) != 1:
            raise ValueError('Expected exactly one pair of original budget constants')
        source = source.replace(before, after)
    return source


SUPERVISOR = continuation_caps(original.SUPERVISOR)


def package(output=None):
    root = Path(__file__).resolve().parent
    output = Path(output) if output is not None else root / 'kaggle-kumo-continuation'
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: (root / name).read_text() for name in ('train_kumo_pair.py', 'train_tabpfn.py')}
    sources['train_kumo_pair.py'] = continuation_caps(sources['train_kumo_pair.py'])
    worker = '\n'.join(f'{name} = {value!r}' for name, value in dict(SOURCES=sources, INPUT_HASHES=original.INPUT_HASHES,
        SOURCE_COMMIT=original.SOURCE_COMMIT, CHECKPOINT_COMMIT=original.CHECKPOINT_COMMIT,
        CHECKPOINT_SHA=original.CHECKPOINT_SHA).items()) + '\n' + original.WORKER
    source = 'RUNNER_SOURCE = ' + repr(sources['train_kumo_pair.py']) + '\nWORKER_SOURCE = ' + repr(worker) + '\n' + SUPERVISOR
    (output / 'train.py').write_text(source)
    py_compile.compile(str(output / 'train.py'), doraise=True)
    metadata = dict(id='wasidyms/s6e10-codex-kumo-continuation-v1', title='S6E10 Codex Kumo Continuation v1',
        code_file='train.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4',
        competition_sources=['playground-series-s6e10'], dataset_sources=[], kernel_sources=[], model_sources=[])
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    return output


if __name__ == '__main__':
    print(package())
