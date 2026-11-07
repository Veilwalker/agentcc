"""Real SDK/model checks after changing actual graphs but keeping task IDs.

This is a deliberately small integration check, not a score benchmark. Its
measurements remain separate from the unmodified public-sample comparison.
"""
import argparse
import asyncio

from tests.compare_versions import (Sample, apply_limits, check_output,
    configure_environment, load_samples, resolve_versions, run, write_json)
from tests.test_regressions_v11 import perturb_topology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = check_output(args.output_dir)
    originals = load_samples(['cloudmatrix/ag', 'cloudmatrix/ar', 'mesh2d/ag', 'sparse_fabric/ar'])
    samples, changes = [], []
    for original in originals:
        graph, changed = perturb_topology(original.topology, 211, preserve_rank_connectivity=True)
        samples.append(Sample(original.label, original.task, graph))
        changes.append({'sample': original.label, **changed})
    configure_environment()
    limits = apply_limits()
    code = asyncio.run(run(resolve_versions(['current']), samples, 1, output, limits))
    write_json(output / 'mutations.json', {
        'scope': 'changed graph, original task/instance IDs; real API, not baseline inputs',
        'changes': changes})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
