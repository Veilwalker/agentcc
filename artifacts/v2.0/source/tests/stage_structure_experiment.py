"""Offline structural ablation. No model calls or production answer cache."""
import json
from pathlib import Path
import time

from src.collectives import locality_order
from src.fabric_routes import RouteCatalog
from src.replay import replay
from src.stage_program import StageSpace
from src.topology import Topology
from tests.compare_versions import load_samples
from tests.collective_structure_experiment import dimension_order


def pressure_order(space):
    """Place expensive physical dimensions at low-volume exchange positions."""
    order, scores = space.order, []
    for bit in range(space.dimensions):
        moves = {('copy', source, order[i ^ (1 << bit)]): [f'{source}_c0']
                 for i, source in enumerate(order)}
        _, cost = space._route(moves)
        scores.append(cost)
    bits = sorted(range(space.dimensions), key=lambda bit: (scores[bit], bit))
    return dimension_order(order, bits), scores


def main():
    output = Path('artifacts/v1.6/analysis/structure-ablation.json')
    rows = []
    for sample in load_samples(all_public=True):
        task, graph = sample.task, Topology.from_data(sample.topology)
        split = task['split_policy']['candidates'][0 if task['operator'] == 'all-gather' else -1]
        for branches in (False, True):
            catalog = RouteCatalog(graph, task['data_size']['per_rank_bytes'] // split, branch_ports=branches)
            base = StageSpace(task, graph, split, catalog=catalog)
            pressure, scores = pressure_order(base)
            orders = [('input', list(graph.ranks)), ('locality', locality_order(graph)), ('pressure', pressure)]
            for name, order in orders:
                start = time.process_time()
                space = StageSpace(task, graph, split, order=order, catalog=catalog)
                cost, widths = space.oracle()
                solution, _ = space.compile(widths)
                checked = replay(task, graph, solution)
                row = dict(sample=sample.label, order=name, branch_ports=branches,
                           cpu_s=time.process_time() - start, widths=widths,
                           time_us=cost, valid=checked['valid'], pressure_us=scores)
                rows.append(row)
                print(json.dumps(row), flush=True)
                output.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
