"""Opt-in real-model SDK checks on changed graphs and binding round budgets.

This is separate from the paired release benchmark and makes four real model
requests (up to one feedback repair each). No synthetic reply is supplied.
"""
import argparse
import asyncio
from copy import deepcopy

from tests.compare_versions import (Sample, apply_limits, check_output, configure_environment,
                                    load_samples, resolve_versions, run)
from tests.test_regressions_v11 import perturb_topology
from tests.test_solver import make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = check_output(args.output_dir)
    samples = []
    for sample in load_samples(['cloudmatrix/ag', 'mesh2d/ar']):
        topology, _changes = perturb_topology(sample.topology, 593, preserve_rank_connectivity=True)
        samples.append(Sample('changed-' + sample.label, deepcopy(sample.task), topology))
    for operator, suffix, limit in [('all-gather', 'ag', 2), ('all-reduce', 'ar', 4)]:
        task, topology = make_case(16, operator)
        task.update(task_id='binding-' + suffix, instance_id='binding-' + suffix, max_nonempty_rounds=limit)
        samples.append(Sample('binding/' + suffix, task, topology))
    configure_environment()
    return asyncio.run(run(resolve_versions(['current']), samples, 1, output, apply_limits()))


if __name__ == '__main__':
    raise SystemExit(main())
