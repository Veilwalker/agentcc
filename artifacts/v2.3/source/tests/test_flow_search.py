from pathlib import Path
import json
import unittest
from unittest.mock import patch

from src import generated_service as service
from src.capacity_backend import native
from src.cut_bounds import gather_bound
from src.flow_search import FlowSearch
from src.fabric_routes import RouteCatalog
from src.reduction_parents import eligibility, regenerate
from src.replay import replay
from src.topology import Topology
from tests.test_solver import make_case


class FlowTests(unittest.TestCase):
    def test_recovery_uses_real_model_evidence_once(self):
        task, topology = make_case(8, 'all-reduce')
        evidence = dict(called=True, usage={'total_tokens': 19})
        with patch.object(service, 'load_settings'), patch.object(service, 'propose',
                return_value=('', evidence)) as model, patch.object(
                service, 'SearchSpace', side_effect=IndexError('injected constructor failure')):
            result = service.solve_input(dict(task=task, topology=topology))
        self.assertTrue(replay(task, topology, result['answer'])['valid'])
        self.assertEqual(model.call_count, 1)
        self.assertEqual(result['diagnostics']['llm']['interactions'], [evidence])
        self.assertEqual(result['diagnostics']['search']['model_edit']['stop_reason'], 'recovery_seed')

    def test_multi_root_reparenting_reads_round_start(self):
        ranks = ['r0','r1','r2','r3']
        def move(a,b,token):
            return dict(action='reduce',src=ranks[a],dst=ranks[b],chunk_ranges=[token])
        rows=[dict(transfers=[move(0,2,'r0_c0'),move(1,3,'r1_c0')]),
              dict(transfers=[move(2,3,'r2_c0+r0_c0'),move(3,2,'r3_c0+r1_c0')])]
        _, masks=eligibility(dict(split=1,rounds=rows),dict(zip(ranks,range(4))))
        self.assertEqual(masks,{(0,0):12})
        rows[0]['transfers'][0]['dst']='r3'
        changed=regenerate(rows,ranks,1)
        self.assertEqual(changed[1]['transfers'][0]['chunk_ranges'],['r2_c0'])
        self.assertEqual(set(changed[1]['transfers'][1]['chunk_ranges'][0].split('+')),
                         {'r0_c0','r1_c0','r3_c0'})

    @unittest.skipIf(native is None, 'C++ capacity backend is optional in development')
    def test_cut_bound_and_optional_flow_failure_keep_valid_answer(self):
        task, raw=make_case(64)
        graph=Topology.from_data(raw)
        result=service.solve_input(dict(task=task,topology=raw),use_llm=False)
        answer=result['answer']
        report=replay(task,graph,answer)
        self.assertTrue(report['valid'])
        self.assertLessEqual(gather_bound(task,graph),report['total_time_us']+1e-6)
        search=FlowSearch(task,graph,RouteCatalog(graph,1024))
        with patch('src.flow_search.gather_bound',return_value=0), patch(
                    'src.congestion_routes.improve',side_effect=ValueError('route incidence limit')):
            selected,checked,audit=search.finish(answer,report)
            self.assertIs(search.finish(answer,report)[0],selected)
        self.assertIs(answer,selected)
        self.assertEqual(checked['total_time_us'],report['total_time_us'])
        self.assertIn('route incidence limit',audit['skipped'])
