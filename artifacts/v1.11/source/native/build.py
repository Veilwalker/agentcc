"""Build against this Python interpreter; no installation or network access."""
import argparse
from pathlib import Path
import subprocess
import sysconfig

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent / 'src')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / ('_capacity_native' + sysconfig.get_config_var('EXT_SUFFIX'))
    temporary = target.with_suffix(target.suffix + '.building')
    command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-ffp-contract=off',
               '-Wall', '-Wextra', '-Wno-missing-field-initializers',
               '-I' + sysconfig.get_path('include'), str(ROOT / 'capacity.cpp'),
               '-o', str(temporary)]
    try:
        subprocess.run(command, check=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(target.resolve())
