"""Check that the documentation matches the code.

1. Every ``autoclass``, ``autofunction``, ``:class:``, ``:func:``, ``:meth:`` and ``:mod:`` target in the docs must import.
2. Every Python code block in the docs, README and examples must run.

Code blocks in the same file share a namespace and run in order.
Training is capped at a few epochs, so this checks that the code runs, not that models fit well.

Usage: ``python docs/check_snippets.py [files ...]``. Without arguments, all docs, README and example files are checked.
"""
import glob
import importlib
import os
import re
import sys
import time
import traceback
import warnings

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS_SOURCE = os.path.join(REPO_ROOT, 'docs', 'source')
MAX_EPOCHS = int(os.environ.get('DOCS_MAX_EPOCHS', 3))

# Schematic snippets that reference undefined names on purpose
SKIP_MARKERS = ['z, log_det_forward = b.forward(x)\n']


def find_reference_targets(rst_files):
    """Collect Python object paths referenced by Sphinx directives and roles.

    :param list rst_files: paths to rst files.
    :rtype: set.
    :return: dotted object paths.
    """
    targets = set()
    for path in rst_files:
        text = open(path, encoding='utf-8').read()
        targets |= set(re.findall(r'(?:autoclass|autofunction):: ([\w.]+)', text))
        targets |= set(re.findall(r':(?:class|func|meth):`[^`<]*<~?(torchflows[\w.]+)>`', text))
        targets |= set(re.findall(r':(?:class|func|meth|mod):`~?(torchflows[\w.]*)`', text))
    return targets


def resolve(path):
    """Import the object at a dotted path.

    :param str path: dotted path, e.g. ``torchflows.flows.Flow``.
    :rtype: bool.
    :return: True if the object exists.
    """
    parts = path.split('.')
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
        except ImportError:
            continue
        try:
            for part in parts[i:]:
                obj = getattr(obj, part)
            return True
        except AttributeError:
            return False
    return False


def extract_code_blocks(path):
    """Extract Python code blocks from an rst or markdown file.

    :param str path: file path.
    :rtype: list.
    :return: list of code strings.
    """
    text = open(path, encoding='utf-8').read()
    if path.endswith('.rst'):
        blocks = re.findall(r'\.\. code-block:: python\n\n((?:(?: {4}.*)?\n)+)', text + '\n')
        return ['\n'.join(line[4:] for line in block.split('\n')) for block in blocks]
    return re.findall(r'```python\n(.*?)```', text, re.S)


def cap_training_epochs():
    """Limit the number of epochs in all flow fitting methods."""
    from torchflows.flows import BaseFlow

    for name in ['fit', 'variational_fit', 'fit_kl_p_to_q']:
        original = getattr(BaseFlow, name)

        def capped(self, *args, _original=original, **kwargs):
            kwargs['n_epochs'] = min(kwargs.get('n_epochs', 500), MAX_EPOCHS)
            kwargs['show_progress'] = False
            return _original(self, *args, **kwargs)

        setattr(BaseFlow, name, capped)


def main(files):
    warnings.filterwarnings('ignore')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import torch
    import torchflows

    print(f'torchflows {torchflows.__version__}, torch {torch.__version__}, python {sys.version.split()[0]}')
    rst_files = sorted(glob.glob(os.path.join(DOCS_SOURCE, '**', '*.rst'), recursive=True))
    if not files:
        files = rst_files + [os.path.join(REPO_ROOT, 'README.md')] + sorted(
            glob.glob(os.path.join(REPO_ROOT, 'examples', '*.md'))
        )

    targets = find_reference_targets(rst_files)
    broken = sorted(t for t in targets if not resolve(t))
    print(f'References: {len(targets)} checked, {len(broken)} broken')
    for target in broken:
        print(f'  BROKEN {target}')

    cap_training_epochs()
    n_blocks = 0
    failures = []
    for path in files:
        namespace = {}
        name = os.path.relpath(path, REPO_ROOT)
        for i, code in enumerate(extract_code_blocks(path)):
            if any(marker in code for marker in SKIP_MARKERS):
                continue
            n_blocks += 1
            t0 = time.time()
            torch.manual_seed(0)
            try:
                exec(compile(code, f'{name}[block {i}]', 'exec'), namespace)
                print(f'ok   {name} block {i} ({time.time() - t0:.1f}s)', flush=True)
            except Exception:
                failures.append(f'{name} block {i}')
                print(f'FAIL {name} block {i}\n{traceback.format_exc()}', flush=True)

    print(f'Code blocks: {n_blocks} run, {len(failures)} failed')
    for failure in failures:
        print(f'  FAILED {failure}')
    return 1 if broken or failures else 0


if __name__ == '__main__':
    sys.exit(main([os.path.abspath(f) for f in sys.argv[1:]]))
