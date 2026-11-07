"""New production orchestration: the model's exact program reaches Replay."""
from unittest.mock import patch
import unittest

from src.config import Settings
from src.forward_service import solve_input
from tests.test_solver import make_case


class ForwardServiceTests(unittest.TestCase):
    def test_reply_is_applied_to_final_control_or_explicitly_rejected(self):
        task, topology = make_case(8)
        settings = Settings('http://example.test/v1/chat/completions', 'test-key', 'test-model')
        calls = []
        def model(task, problem, **kwargs):
            counts = [2] * len(problem['edits'])
            calls.append(counts)
            return {'counts':counts}, {'usage': {'prompt_tokens':20, 'completion_tokens':10}}
        with patch('src.forward_service.load_settings', return_value=settings), patch('src.forward_service.edit_retiming', side_effect=model):
            result = solve_input({'task':task, 'topology':topology})
        self.assertEqual(len(calls), 1)
        diag = result['diagnostics']
        self.assertTrue(diag['local_replay']['valid'])
        edit = diag['search']['model_edit']
        self.assertEqual(edit['requested'], {'counts':calls[0]})
        self.assertEqual(edit['compiled_counts'], calls[0])
        self.assertLessEqual(edit['final_time_us'], edit['control_time_us'])
        self.assertNotIn('strategy', result['answer'])
        self.assertTrue(diag['llm']['called'])

    def test_transport_failure_does_not_fabricate_model_use(self):
        from src.llm import LLMError
        task, topology = make_case(4)
        settings = Settings('http://example.test/v1/chat/completions', 'test-key', 'test-model')
        with patch('src.forward_service.load_settings', return_value=settings), patch('src.forward_service.edit_retiming', side_effect=LLMError('mock failure')):
            with self.assertRaises(LLMError):
                solve_input({'task':task,'topology':topology})

    def test_invalid_response_keeps_verified_control(self):
        task, topology = make_case(4)
        settings = Settings('http://example.test/v1/chat/completions', 'test-key', 'test-model')
        with patch('src.forward_service.load_settings', return_value=settings), patch('src.forward_service.edit_retiming', return_value=(None, {})):
            result = solve_input({'task':task,'topology':topology})
        self.assertTrue(result['diagnostics']['local_replay']['valid'])
        self.assertFalse(result['diagnostics']['search']['model_edit']['selected'])
        self.assertEqual(result['diagnostics']['search']['model_edit']['reason'], 'rejected')

    def test_offline_control_does_not_call_or_claim_model(self):
        task, topology = make_case(4)
        with patch('src.forward_service.edit_retiming', side_effect=AssertionError('offline network')):
            result = solve_input({'task':task,'topology':topology}, use_llm=False)
        self.assertFalse(result['diagnostics']['llm']['called'])
        self.assertTrue(result['diagnostics']['local_replay']['valid'])
