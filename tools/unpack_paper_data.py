import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['benchmark', 'multiview', 'all'], default='all')
    parser.add_argument('--output-root', type=Path, default=Path('input_data'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output_root.resolve()
    names = ['benchmark', 'multiview'] if args.dataset == 'all' else [args.dataset]
    counts = {}
    for name in names:
        count = 0
        with ZipFile(root / 'data/archives' / ('prism_' + name + '.zip')) as archive:
            for info in archive.infolist():
                relative = Path(info.filename)
                if info.is_dir() or any(part.startswith('.') or part in ('__MACOSX', '__pycache__') for part in relative.parts):
                    continue
                path = (target / relative).resolve()
                if not path.is_relative_to(target):
                    raise ValueError(info.filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(info))
                count += 1
        counts[name] = count
    print(json.dumps(dict(output_root=str(target), extracted_files=counts), indent=2))


if __name__ == '__main__':
    main()
