"""Register an already installed isolated msbuddy 1.0.0 runtime on this computer."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
from standard_lipidomics.msbuddy_runtime import configuration_path, freeze_runtime
from output_io import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--libomp', default='')
    parser.add_argument('--config', type=Path, default=None)
    args = parser.parse_args()
    config = {'schema_version': 1, 'python': str(Path(args.python).expanduser().absolute()),
              'data': str(Path(args.data).expanduser().resolve()),
              'libomp': str(Path(args.libomp).expanduser().resolve()) if args.libomp else ''}
    frozen = freeze_runtime(config)
    if not frozen['available']:
        raise ValueError(frozen['reason'])
    target = args.config or configuration_path()
    atomic_write_json(target, config)
    print(json.dumps({'configured': True, 'version': frozen['version'], 'configuration': str(target)}))


if __name__ == '__main__':
    main()
