"""Offline fanout sweep with full Replay; never used to select by instance ID."""
import json
from pathlib import Path
import time

from src.collectives import locality_order
from src.fabric_routes import RouteCatalog
from src.replay import replay
from src.stage_program import StageSpace
from src.topology import Topology
from tests.compare_versions import load_samples


def main():
    output = Path('artifacts/v1.6/analysis/fanout-ablation.json')
    rows = []
    for sample in load_samples(all_public=True):
        task, graph = sample.task, Topology.from_data(sample.topology)
        split = task['split_policy']['candidates'][0 if task['operator'] == 'all-gather' else -1]
        catalog = RouteCatalog(graph, task['data_size']['per_rank_bytes'] // split, branch_ports=True)
        spaces = [StageSpace(task, graph, split, order=order, catalog=catalog, max_width=8)
                  for order in (list(graph.ranks), locality_order(graph))]
        space = min(spaces, key=lambda s: sum(s.transition(i, 1)[2] for i in range(s.dimensions)))
        for width in (2, 3, 4, 5, 6, 7, 8):
            started = time.process_time()
            d = space.dimensions
            program = [width] * (d // width) + ([d % width] if d % width else [])
            if task['operator'] == 'all-gather':
                program.reverse()
            solution, cost = space.compile(program)
            checked = replay(task, graph, solution)
            row = dict(sample=sample.label, width=width, program=program,
                       cpu_s=time.process_time() - started, time_us=cost,
                       valid=checked['valid'], transfers=sum(len(r['transfers']) for r in solution['rounds']),
                       bytes=len(json.dumps(solution)))
            rows.append(row)
            print(json.dumps(row), flush=True)
            output.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
