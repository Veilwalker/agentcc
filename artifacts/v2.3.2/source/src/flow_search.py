"""Spend the model wait on one reusable, certified flow refinement."""
from time import monotonic

from . import central_exchange, congestion_routes, domain_fanout, relay_exchange, slack_schedule, shard_pipeline, threshold_routes
from .replay import certify
from .solver import validate_task
from .capacity_backend import native
from .cut_bounds import gather_bound


class FlowSearch:
    def __init__(self, task, graph, catalog, pending=None):
        self.task, self.graph, self.catalog = task, graph, catalog
        self.pending = pending
        self.slack_options = {}
        self.parent = None
        self.result = None
        self.slack_data = None

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
                    try:
                        plan, report, exchange = central_exchange.improve(*args, plan, report, self.catalog)
                        if exchange:
                            audit['central_exchange_round'] = exchange
                    except (ValueError, KeyError, IndexError) as error:
                        audit['central_exchange_skipped'] = type(error).__name__
                    # Placement is useful when receivers have unequal access to
                    # the fabric, or when the schedule crosses many barriers.
                    if len(plan['rounds']) >= 8 or self._irregular_access():
                        plan, report = relay_exchange.improve(*args, plan, report, self.catalog, trials=1024)
                        audit['placement_us'] = report['total_time_us']
                bound = gather_bound(*args) if self.task['operator'] == 'all-gather' else 0
                if bound:
                    audit['cut_bound_us'] = bound
                if bound and report['total_time_us'] > bound*1.50:
                    allowed = validate_task(*args)
                    try:
                        option = domain_fanout.price(*args,self.catalog,allowed)
                    except (ValueError, KeyError, IndexError) as error:
                        option = None
                        audit['domain_fanout_skipped'] = type(error).__name__
                    if option and option[0] < report['total_time_us']*.985:
                        candidate = shard_pipeline.materialize(*args,{'split':1},self.catalog,option[1])
                        verified = certify(*args,candidate)
                        if verified['valid'] and abs(verified['total_time_us']-option[0]) < max(1e-4,option[0]*1e-9):
                            plan,report = candidate,verified
                            audit['domain_fanout_us'] = option[0]
                if report['total_time_us'] > bound * 1.025:
                    plan, report, rounds = congestion_routes.improve(*args, plan, report, self.catalog)
                    audit['changed_rounds'] = rounds
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
                # A limited route catalog or unsupported retirement structure
                # cannot invalidate a schedule certified before refinement.
                audit['skipped'] = str(error)[:180]
        self.slack_data = None
        self.slack_options = {}
        if native is not None and 64 <= len(self.graph.ranks) <= 256 and report.get('valid'):
            try:
                self.slack_data = slack_schedule.prepare(self.task,self.graph,plan,report,self.catalog)
                for relay in (False, True):
                    if self.slack_data is None or self.pending is None or self.pending.done():
                        break
                    trial = slack_schedule.improve(self.task,self.graph,plan,report,self.catalog,
                                                   prepared=self.slack_data,prefer_relay=relay)
                    self.slack_options[relay] = trial
                    if trial[2] and trial[2]['elapsed_s'] > .25:
                        break
            except (ValueError, IndexError, TypeError, KeyError, OverflowError) as error:
                audit['slack_prepare_error'] = type(error).__name__
        audit.update(final_us=report['total_time_us'], elapsed_s=round(monotonic() - started, 6))
        self.parent, self.result = original, (plan, report, audit)
        return self.result

    def apply_focus(self, plan, report, audit, probes, interactions):
        if self.slack_data is not None:
            choice = interactions[-1].get('focus_id') if interactions else None
            probe = probes[choice] if type(choice) is int and 0 <= choice < len(probes) else None
            key = probe['key'] if probe else None
            policy = interactions[-1].get('repair_policy') if probe else None
            try:
                relay = policy == 'R'
                option = self.slack_options.get(relay)
                binding = []
                if option is not None and key is not None:
                    kind = 'pool' if isinstance(key,str) else 'edge'
                    binding = [row['round'] for row in option[2]['rounds']
                               if row['peak_resource'] == (kind,key) and row['ready_earlier']]
                # The model can select a prepared intervention. Reprice its
                # specific physical resource if the generic intervention stalls
                # and the measured final schedule confirms the hypothesis.
                focused = option is None or (binding and not option[2]['accepted'])
                if focused:
                    option = slack_schedule.improve(self.task,self.graph,plan,report,self.catalog,
                                                    focus=key,prepared=self.slack_data,prefer_relay=relay)
                plan, report, slack = option
                slack.update(reused_during_model_wait=not focused, matched_binding_rounds=binding)
                self.slack_data = None
                self.slack_options.clear()
                if slack:
                    if probe:
                        slack['model_selection'] = dict(round=choice+1,resource=key,observation=probe['row'],policy=policy)
                    audit = dict(audit, round_slack=slack, final_us=report['total_time_us'])
            except (ValueError, IndexError, TypeError, KeyError, OverflowError) as error:
                audit = dict(audit, slack_search_error=type(error).__name__)
        return plan, report, audit

    def _irregular_access(self):
        graph = self.graph
        ranks = set(graph.ranks)
        return any(pools and (u in ranks or v in ranks)
                   for (u, v), pools in graph.edge_resources.items())


def record(search, report, audit):
    search['flow_search'] = audit
    if search['model_edit'].get('semantics') == 'model_round_operator_v1':
        slack = audit.get('round_slack',{})
        selected = bool(slack.get('model_selection') and slack.get('accepted'))
        search['model_edit'].update(selected=selected,accepted=selected,choice=slack.get('model_selection'))
    search['model_edit']['final_time_us'] = report['total_time_us']
    for candidate in search['candidates']:
        if candidate.get('selected'):
            candidate['total_time_us'] = report['total_time_us']
