"""Spend the model wait on one reusable, certified flow refinement."""
from time import monotonic

from . import congestion_routes, relay_exchange, threshold_routes
from .capacity_backend import native
from .cut_bounds import gather_bound


class FlowSearch:
    def __init__(self, task, graph, catalog):
        self.task, self.graph, self.catalog = task, graph, catalog
        self.parent = None
        self.result = None

    def finish(self, plan, report):
        if plan is self.parent and self.result is not None:
            return self.result
        started = monotonic()
        original = plan
        audit = {'parent_us': report['total_time_us']}
        if native is not None and 64 <= len(self.graph.ranks) <= 256 and report.get('valid'):
            args = self.task, self.graph
            try:
                if self.task['operator'] == 'all-reduce':
                    plan, report, threshold = threshold_routes.improve(*args, plan, report, self.catalog)
                    if threshold:
                        audit['threshold_rounds'] = threshold
                    # Placement is useful when receivers have unequal access to
                    # the fabric, or when the schedule crosses many barriers.
                    if len(plan['rounds']) >= 8 or self._irregular_access():
                        plan, report = relay_exchange.improve(*args, plan, report, self.catalog, trials=1024)
                        audit['placement_us'] = report['total_time_us']
                bound = gather_bound(*args) if self.task['operator'] == 'all-gather' else 0
                if bound:
                    audit['cut_bound_us'] = bound
                if report['total_time_us'] > bound * 1.025:
                    plan, report, rounds = congestion_routes.improve(*args, plan, report, self.catalog)
                    audit['changed_rounds'] = rounds
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
                # A limited route catalog or unsupported retirement structure
                # cannot invalidate a schedule certified before refinement.
                audit['skipped'] = str(error)[:180]
        audit.update(final_us=report['total_time_us'], elapsed_s=round(monotonic() - started, 6))
        self.parent, self.result = original, (plan, report, audit)
        return self.result

    def _irregular_access(self):
        graph = self.graph
        ranks = set(graph.ranks)
        return any(pools and (u in ranks or v in ranks)
                   for (u, v), pools in graph.edge_resources.items())


def record(search, report, audit):
    search['flow_search'] = audit
    search['model_edit']['final_time_us'] = report['total_time_us']
    for candidate in search['candidates']:
        if candidate.get('selected'):
            candidate['total_time_us'] = report['total_time_us']
