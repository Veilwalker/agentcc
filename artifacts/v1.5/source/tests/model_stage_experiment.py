"""Explicit SDK experiment: compare model placement without hidden extra calls."""
import argparse
import asyncio
from functools import partial
import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

from tests import compare_versions as compare
import src.service as service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['seed', 'residual', 'guided', 'anchored'], required=True)
    parser.add_argument('--samples', nargs='+', default=['cloudmatrix/ag', 'cloudmatrix/ar', 'mesh2d/ar', 'nvlink_like/ar'])
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = compare.check_output(args.output_dir)
    versions = compare.resolve_versions(['1.4', '1.5'])
    samples = compare.load_samples(args.samples)
    compare.configure_environment()
    limits = compare.apply_limits()
    limits['model_stage_experiment'] = args.stage
    archive = compare.PROJECT / 'artifacts/v1.5/iterations/revision-02-pilots/latest-src'
    name = 'agentcc_placement_prototype'
    spec = importlib.util.spec_from_file_location(name, archive / '__init__.py', submodule_search_locations=[str(archive)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    prototype = importlib.import_module(name + '.service')
    limits['prototype_source'] = str(archive.relative_to(compare.PROJECT))
    limits['prototype_hashes'] = compare.source_hashes([compare.Version('prototype', archive, False)])
    with patch.object(service, 'solve_input', partial(prototype.solve_input, model_stage=args.stage)):
        result = asyncio.run(compare.run(versions, samples, args.repeat, output, limits))
    raise SystemExit(result)

if __name__ == '__main__':
    main()
