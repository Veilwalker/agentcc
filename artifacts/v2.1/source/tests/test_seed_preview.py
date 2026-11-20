"""Exact seed pricing and deferred-validation failure paths."""
import unittest
from threading import Event
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.seed_preview import SeedPreview
from src.topology import Topology
from src.generated_service import solve_input
from src.generated_service import validate_guard
from src.code_policy import assemble_frontier
from src.llm import LLMError
from tests.test_solver import make_case


class SeedPreviewTests(unittest.TestCase):
    def test_guard_validation_overlaps_wait_without_skipping_model_execution(self):
        task, data = make_case(8, 'all-gather')
        ready = Event()

        def model(*_args):
            if not ready.wait(5):
                raise AssertionError('guard was not validated while waiting')
            return assemble_frontier('min((v+t,p+[w]) for v,p,w,t in a)'), {
                'called': True, 'program_protocol': 'incoming_v1', 'usage': {}}

        def validate(*args):
            result = validate_guard(*args)
            ready.set()
            return result

        with patch('src.generated_service.load_settings'), \
             patch('src.generated_service.propose', side_effect=model) as called, \
             patch('src.generated_service.validate_guard', side_effect=validate):
            output = solve_input({'task': task, 'topology': data})
        evidence = output['diagnostics']['search']['model_edit']
        self.assertEqual(called.call_count, 1)
        self.assertTrue(evidence['guard']['prepared_during_model_wait'])
        self.assertTrue(evidence['executions'][0]['valid_program'])
        self.assertTrue(replay(task, data, output['answer'])['valid'])

    def test_failed_model_still_materializes_and_replays_guard(self):
        task, data = make_case(8, 'all-gather')
        with patch('src.generated_service.load_settings'), \
             patch('src.generated_service.propose', side_effect=LLMError('offline transport failure')):
            output = solve_input({'task': task, 'topology': data})
        self.assertTrue(output['diagnostics']['local_replay']['valid'])
        self.assertTrue(replay(task, data, output['answer'])['valid'])
        self.assertEqual(output['diagnostics']['search']['model_edit']['stop_reason'],
                         'model_transport_failed')

    def test_final_answer_never_uses_unverified_capacity_preview(self):
        task, data = make_case(8, 'all-reduce')
        with patch('src.generated_service.replay', wraps=replay) as checked:
            output = solve_input({'task': task, 'topology': data}, use_llm=False)
        self.assertTrue(output['diagnostics']['local_replay']['valid'])
        self.assertTrue(replay(task, data, output['answer'])['valid'])
        # Legacy construction may validate internally. Any deferred path must
        # instead call this service's independent checker before returning.
        evidence = output['diagnostics']['search']
        if not evidence['initial_candidate_materialized']:
            self.assertGreater(checked.call_count, 0)

    def test_matches_original_constructor_with_overlapping_capacity_pools(self):
        for operator in ('all-gather', 'all-reduce'):
            for n in (4, 8, 16):
                task, data = make_case(n, operator)
                data['resources']=[{'id':'shared','capacity':1e8,'edges':data['edges'][::3]}]
                graph=Topology.from_data(data)
                for split in (1, 4):
                    preview=SeedPreview(task,graph,split)
                    expected=build(task,graph,split,kind='reverse_doubling' if operator=='all-gather' else 'xor_tree',
                                   order=list(graph.ranks),resource_weight=4.,packets=2)
                    actual, cost=preview.compile([1]*preview.dimensions)
                    before, after=replay(task,graph,expected),replay(task,graph,actual)
                    self.assertTrue(before['valid'],before['errors'])
                    self.assertTrue(after['valid'],after['errors'])
                    self.assertAlmostEqual(cost,before['total_time_us'],places=6)
                    self.assertAlmostEqual(cost,after['total_time_us'],places=6)
                    self.assertAlmostEqual(cost,preview.binary_cost(),places=6)


if __name__=='__main__':
    unittest.main()
