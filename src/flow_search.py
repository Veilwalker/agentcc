"""Spend the model wait on one reusable, certified flow refinement."""
from time import monotonic

from . import central_exchange, congestion_routes, domain_cycles, domain_fanout, domain_reduce, local_bypass, local_flatten, local_tail, paced_pipeline, pod_fold, relay_exchange, slack_schedule, shard_pipeline, threshold_routes
from .replay import certify
from .solver import validate_task
from .capacity_backend import native
from .cut_bounds import gather_bound


class FlowSearch:
    def __init__(self, task, graph, catalog, pending=None):
        self.task, self.graph, self.catalog = task, graph, catalog
        self.pending = pending
        self.slack_options = {}
        self.focus_previews = {}
        self.parent = None
        self.result = None
        self.deferred_tree = None
        self.slack_data = None

    def finish(self, plan, report):
        if plan is self.parent and self.result is not None:
            return self.result
        started = monotonic()
        original = plan
        self.deferred_tree = None
        audit = {'parent_us': report['total_time_us']}
        if native is not None and 64 <= len(self.graph.ranks) <= 256 and report.get('valid'):
            args = self.task, self.graph
            try:
                reducing = self.task['operator'] == 'all-reduce'
                irregular = reducing and self._irregular_access()
                allowed = validate_task(*args)
                tree_won = False
                # Compare shallow physical-domain trees before costly generic repairs.
                # A modest gain retains both candidates until the repaired plan is known.
                if not irregular and (reducing or plan['split'] == 1):
                    family = domain_reduce if reducing else domain_cycles
                    candidate, checked, detail = family.improve(*args, plan, report, self.catalog, allowed)
                    if detail:
                        audit['domain_reduce' if reducing else 'domain_cycles'] = detail
                    if candidate is not plan:
                        if checked['total_time_us'] < .90 * report['total_time_us']:
                            plan, report = candidate, checked
                            tree_won = True
                            audit['early_domain_tree'] = True
                        else:
                            self.deferred_tree = candidate, checked
                if reducing and not tree_won:
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
                    if len(plan['rounds']) >= 8 or irregular:
                        plan, report = relay_exchange.improve(*args, plan, report, self.catalog, trials=1024)
                        audit['placement_us'] = report['total_time_us']
                bound = gather_bound(*args) if self.task['operator'] == 'all-gather' else 0
                if bound:
                    audit['cut_bound_us'] = bound
                if bound and not tree_won and report['total_time_us'] > bound*1.50:
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
                before_routing = report['total_time_us']
                if not tree_won and before_routing > bound * 1.025:
                    plan, report, rounds = congestion_routes.improve(*args, plan, report, self.catalog)
                    audit['changed_rounds'] = rounds
                if self.task['operator'] == 'all-reduce' and not tree_won:
                    plan, report, merged = local_flatten.improve(*args, plan, report, self.catalog)
                    if merged:
                        audit['flattened_local_rounds'] = merged
                    plan, report, bypassed = local_bypass.improve(*args, plan, report, self.catalog)
                    if bypassed:
                        audit['parallel_local_rounds'] = bypassed
                    else:
                        plan, report, folded = pod_fold.improve(*args, plan, report, self.catalog, allowed)
                        if folded:
                            audit['gateway_fold'] = folded
                if irregular:
                    candidate, checked, detail = domain_reduce.improve(*args, plan, report, self.catalog, allowed)
                    if detail:
                        audit['domain_reduce'] = detail
                    if candidate is not plan:
                        self.deferred_tree = candidate, checked
                # Once per-shard route and parent changes have removed a large
                # part of the critical path, cloning shard zero's tree across
                # the other shards discards that diversity and costs a full scan.
                if bound and report['total_time_us'] > max(bound*1.06, before_routing*.80):
                    plan,report,pacing = paced_pipeline.improve(*args,plan,report,self.catalog)
                    if pacing:
                        audit['paced_pipeline'] = pacing
                if bound and plan['split'] == 1 and report['total_time_us'] > bound*1.025:
                    plan, report, tail = local_tail.improve(*args, plan, report, self.catalog)
                    if tail:
                        audit['local_tail'] = tail
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
                # A limited route catalog or unsupported retirement structure
                # cannot invalidate a schedule certified before refinement.
                audit['skipped'] = str(error)[:180]
        self.slack_data = None
        self.slack_options = {}
        self.focus_previews = {}
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

    def prepare_focus(self, probes):
        if self.slack_data is None or self.pending is None or self.result is None:
            return
        plan, report, _ = self.result
        for relay, option in self.slack_options.items():
            data = option[2] or {}
            if data.get('accepted'):
                continue
            active = {tuple(row['peak_resource']) for row in data.get('rounds', ()) if row['ready_earlier']}
            for probe in probes:
                key = probe['key']
                tagged = ('pool' if isinstance(key,str) else 'edge', key)
                if tagged not in active or (relay,key) in self.focus_previews:
                    continue
                if self.pending.done() or len(self.focus_previews) >= 6:
                    return
                trial = slack_schedule.improve(self.task,self.graph,plan,report,self.catalog,
                                               focus=key,prepared=self.slack_data,prefer_relay=relay)
                self.focus_previews[relay,key] = trial
                if trial[2] and trial[2].get('accepted'):
                    return

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
                cached = self.focus_previews.get((relay,key))
                if focused and cached is not None:
                    option = cached
                    focused = False
                if focused:
                    option = slack_schedule.improve(self.task,self.graph,plan,report,self.catalog,
                                                    focus=key,prepared=self.slack_data,prefer_relay=relay)
                model_winner = True
                chosen_policy = relay
                for candidate_policy, other in self.slack_options.items():
                    if other[1]['total_time_us'] < option[1]['total_time_us'] - 1e-6:
                        option = other
                        model_winner = False
                        chosen_policy = candidate_policy
                plan, report, slack = option
                slack['model_preferred_winner'] = bool(policy in ('E', 'R') and model_winner)
                slack['applied_policy'] = 'R' if chosen_policy else 'E'
                slack.update(reused_during_model_wait=not focused or not model_winner, matched_binding_rounds=binding)
                self.slack_data = None
                self.slack_options.clear()
                if slack:
                    if probe:
                        slack['model_selection'] = dict(round=choice+1,resource=key,observation=probe['row'],policy=policy)
                    audit = dict(audit, round_slack=slack, final_us=report['total_time_us'])
            except (ValueError, IndexError, TypeError, KeyError, OverflowError) as error:
                audit = dict(audit, slack_search_error=type(error).__name__)
        if self.deferred_tree is not None:
            candidate, checked = self.deferred_tree
            self.deferred_tree = None
            if checked['total_time_us'] < report['total_time_us'] - 1e-6:
                plan, report = candidate, checked
                audit = dict(audit, deferred_tree_won=True, final_us=report['total_time_us'])
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
        selected = bool(not audit.get('deferred_tree_won') and slack.get('model_selection') and slack.get('accepted') and slack.get('model_preferred_winner'))
        search['model_edit'].update(selected=selected,accepted=selected,choice=slack.get('model_selection'))
    search['model_edit']['final_time_us'] = report['total_time_us']
    for candidate in search['candidates']:
        if candidate.get('selected'):
            candidate['total_time_us'] = report['total_time_us']
