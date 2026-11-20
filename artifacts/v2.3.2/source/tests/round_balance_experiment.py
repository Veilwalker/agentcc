"""Offline route ablations, with independent Replay; no model usage or score estimate."""
import argparse
import json
from pathlib import Path
import time

from src.balanced_stages import BalancedStageSpace
from src.collectives import locality_order
from src.fabric_routes import RouteCatalog
from src.stage_program import StageSpace
from src.replay import replay
from src.topology import Topology
from tests.compare_versions import load_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('keep previous measurements; choose a new output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in load_samples(args.samples, all_public=not args.samples):
        task, graph = sample.task, Topology.from_data(sample.topology)
        split = task['split_policy']['candidates'][0 if task['operator'] == 'all-gather' else -1]
        catalog = RouteCatalog(graph, task['data_size']['per_rank_bytes'] // split, branch_ports=True)
        for order_name, order in [('input', list(graph.ranks)), ('locality', locality_order(graph))]:
            variants = [('plain', StageSpace, {}),
                        ('small_packets', BalancedStageSpace, {'stripes': 1, 'passes': 0}),
                        ('striped', BalancedStageSpace, {'stripes': 4, 'passes': 0}),
                        ('balanced', BalancedStageSpace, {'stripes': 4, 'passes': 2}),
                        ('balanced8', BalancedStageSpace, {'stripes': 8, 'passes': 2})]
            for mode, cls, options in variants:
                started = time.process_time()
                space = cls(task, graph, split, order=order, catalog=catalog, **options)
                cost, widths = space.oracle()
                preview_cpu = time.process_time() - started
                solution, _ = space.compile(widths)
                checked = replay(task, graph, solution)
                row = dict(sample=sample.label, order=order_name, mode=mode, time_us=cost,
                           preview_cpu_s=preview_cpu, widths=widths, valid=checked['valid'],
                           replay_us=checked['total_time_us'], errors=checked['errors'],
                           transfers=sum(len(r['transfers']) for r in solution['rounds']))
                if not checked['valid'] or abs(cost-checked['total_time_us']) > 1e-4:
                    raise AssertionError(row)
                rows.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
