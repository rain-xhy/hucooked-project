"""Read saved checkpoints without running any additional environment episodes."""
import argparse
import json
from pathlib import Path
import torch


def tensors(value, path='root'):
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items(): yield from tensors(item, path + '.' + str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value): yield from tensors(item, path + '.' + str(i))


def verify(run):
    run = Path(run)
    latest = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
    initial = torch.load(run / 'initial.pt', map_location='cpu', weights_only=False)
    before, after = dict(tensors(initial)), dict(tensors(latest))
    # CoMeDi moves the first live policy into the frozen convention list after
    # its training rounds; compare that same policy across the naming change.
    if initial.get('current') is not None and latest.get('conventions'):
        for key, value in list(before.items()):
            if key.startswith('root.current.'):
                before[key.replace('root.current.', 'root.conventions.0.', 1)] = value
    if not after: raise AssertionError('Checkpoint contains no tensors')
    changed, delta = [], 0.
    for key, value in after.items():
        if not torch.isfinite(value).all(): raise AssertionError('Nonfinite tensor: ' + key)
        if key in before and before[key].shape == value.shape and value.is_floating_point():
            difference = float((value - before[key]).abs().sum())
            if difference: changed.append(key); delta += difference
    status = json.loads((run / 'status.json').read_text())
    return {'status': {k: v for k, v in status.items() if not isinstance(v, (dict, list))},
            'finite_tensors': len(after), 'matching_tensors_changed': len(changed),
            'l1_change': delta, 'changed_examples': changed[:8]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run'); parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = verify(args.run)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output: Path(args.output).write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__': main()
