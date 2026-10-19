import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def behavior(condition):
    match = re.match(r'^(.+?)_f[0-9]', str(condition))
    if match is None:
        raise ValueError(condition)
    result = match.group(1)
    if result not in ['arm_flapping', 'body_rocking', 'hand_flapping', 'head_banging', 'head_shaking']:
        raise ValueError(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, default=Path('input_data/prism_multiview'))
    parser.add_argument('--output-root', type=Path, default=Path('output/multiview_inputs'))
    args = parser.parse_args()
    if args.input_root.resolve() == args.output_root.resolve():
        raise ValueError('Use a separate output directory')
    target = args.output_root / 'trajectories'
    target.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.input_root / 'trajectories').glob('*.npz'))
    if not paths:
        raise ValueError('No input trajectories')
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        condition = str(data.get('condition_id', path.stem))
        data['behavior'] = np.array(behavior(condition))
        np.savez_compressed(target / path.name, **data)
    rows = {}
    for source, name in [('construct_recovery_rows.csv', 'construct_recovery_rows.csv'),
                         ('regularity/regularity_rows.csv', 'regularity_rows.csv')]:
        path = args.input_root / source
        if path.exists():
            frame = pd.read_csv(path)
            frame['behavior'] = frame.condition_id.map(behavior)
            frame.to_csv(args.output_root / name, index=False)
            rows[name] = len(frame)
    report = dict(trajectories=len(paths), tables=rows,
                  operation='Recover behavior metadata from the known generating condition IDs; original arrays and source archives are preserved.')
    (args.output_root / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
