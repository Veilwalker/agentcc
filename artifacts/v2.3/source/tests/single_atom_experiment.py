"""Test whole-round route rebalancing even when a movement has one atom.

One atom cannot be split, but its complete path can still be changed. The old
screen excluded both operations. This experiment separates them explicitly.
"""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from src.refined_search import RefinedProfile, RefinedSearchSpace
from src.replay import replay
from src.topology import Topology
from tests.compare_versions import apply_limits, load_samples
from tests.split_experiment import best_output


def rebalance_transition(self, i, w, phase=None):
    key = i, w, phase
    if key not in self.selected:
        variants = [self.plain, self.striped]
        if self.holders is not None and i >= 3 and w == 1:
            variants.append(self.holders)
        def cost(space):
            value = space.transition(i, w)[2]
            return value if phase is None else space.phase_times[i, w][phase]
        self.selected[key] = min(variants, key=cost)
    return self.selected[key].transition(i, w)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('retain previous experiment output')
    apply_limits()
    original = RefinedProfile.transition
    rows = []
    for sample in load_samples(None, all_public=True):
        for mode in ('baseline', 'rebalance-single-atom'):
            with patch.object(RefinedProfile, 'transition', original if mode == 'baseline' else rebalance_transition):
                started = time.process_time()
                graph = Topology.from_data(sample.topology)
                splits = sample.task['split_policy']['candidates']
                split = min(splits) if sample.task['operator'] == 'all-gather' else max(splits)
                space = RefinedSearchSpace(sample.task, graph, split)
                output, cost = best_output(space)
                preview = time.process_time() - started
                solution, predicted = space.compile(output)
                checked = replay(sample.task, graph, solution)
                encoded = json.dumps(solution, separators=(',', ':'))
                total = time.process_time() - started
                assert checked['valid'], checked['errors']
                assert abs(checked['total_time_us'] - predicted) < 1e-4
                row = {'sample': sample.label, 'mode': mode, 'time_us': cost,
                       'preview_s': preview, 'cpu_s': total, 'bytes': len(encoded),
                       'widths': output[1], 'valid': True}
                rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(rows, indent=2) + '\n')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
