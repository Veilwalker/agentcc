"""Offline split ablation: charge preview, materialization, Replay and JSON.

The dynamic program is an offline reference only; production still executes
model-authored code. No model calls or official score estimates are made here.
"""
import argparse
import json
from pathlib import Path
import time

from src.refined_search import RefinedSearchSpace
from src.replay import replay
from src.topology import Topology
from tests.compare_versions import apply_limits, load_samples


def best_output(space):
    choices = []
    for profile, table in enumerate(space.costs):
        states = {(0, 0): (0., [])}
        for i, row in enumerate(table):
            for k in range(space.limit):
                if (i, k) not in states:
                    continue
                cost, path = states[i, k]
                for width, step in enumerate(row, 1):
                    key = i + width, k + 1
                    proposal = cost + step, path + [width]
                    if key not in states or proposal < states[key]:
                        states[key] = proposal
        for (i, _k), (cost, path) in states.items():
            if i == space.dimensions:
                choices.append((cost, profile, path))
    cost, profile, widths = min(choices)
    return [profile, widths], cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('retain previous experiment output')
    apply_limits()
    rows = []
    for sample in load_samples(None, all_public=True):
        if sample.task['operator'] != 'all-gather':
            continue
        for split in (1, 2, 4):
            if split not in sample.task['split_policy']['candidates']:
                continue
            started = time.process_time()
            graph = Topology.from_data(sample.topology)
            space = RefinedSearchSpace(sample.task, graph, split)
            output, cost = best_output(space)
            preview_s = time.process_time() - started
            solution, predicted = space.compile(output)
            materialized_s = time.process_time() - started - preview_s
            checked = replay(sample.task, graph, solution)
            encoded = json.dumps(solution, separators=(',', ':'))
            total_s = time.process_time() - started
            assert checked['valid'], checked['errors']
            assert abs(checked['total_time_us'] - predicted) < 1e-4
            assert abs(cost - predicted) < 1e-4
            row = {'sample': sample.label, 'split': split, 'time_us': cost,
                   'preview_s': preview_s, 'materialize_s': materialized_s,
                   'cpu_s': total_s, 'response_bytes': len(encoded.encode()),
                   'widths': output[1], 'valid': True}
            rows.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
