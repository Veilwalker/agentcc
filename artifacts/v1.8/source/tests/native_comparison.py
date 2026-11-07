"""Paired SDK comparison of Python/native releases using fixed model replies.

No network or credentials. Replays original v1.7 model output byte-for-byte;
simulated waits are explicitly separate from real-model release measurements.
"""
import argparse
import asyncio
from contextlib import AsyncExitStack, ExitStack
import gc
import hashlib
import importlib
import json
import socket
import statistics
import time
from unittest.mock import patch

from tests import compare_versions as compare


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def no_network(*_args, **_kwargs):
    raise AssertionError('network and credentials are forbidden in the offline comparison')


def recordings(folder, samples):
    data = compare.read_object(folder / 'summary.json')
    result = {}
    for sample in samples:
        if compare.read_object(folder / 'inputs' / (sample.label.replace('/', '-') + '.json')) != sample.request():
            raise ValueError('recording input differs from current public input')
        row = next(r for r in data['rows'] if r['version'] == '1.7'
                   and r['repeat'] == 1 and r['sample'] == sample.label)
        reply = compare.read_object(folder / row['response_file'])['output']['diagnostics']['llm']['interactions'][0]
        raw = reply['response_evidence']['content']
        if reply.get('synthetic') or reply['response_evidence'].get('credential_redacted'):
            raise ValueError('expected an unchanged real model reply')
        if hashlib.sha256(raw.encode()).hexdigest() != reply['response_sha256']:
            raise ValueError('recorded reply hash mismatch')
        result[sample.label] = reply
    return result


def summary(report):
    groups, agreements = [], []
    for sample in report['samples']:
        for delay in report['delays_s']:
            rows = [r for r in report['rows'] if r['sample'] == sample and r['delay_s'] == delay]
            complete = len(rows) == len(report['versions']) * report['repeat']
            agreements.append({'sample': sample, 'delay_s': delay, 'complete': complete,
                'same_schedule': len({r['schedule_hash'] for r in rows}) == 1,
                'same_prompt': len({r['prompt_hash'] for r in rows}) == 1})
            for version in report['versions']:
                matched = [r for r in rows if r['version'] == version]
                if matched:
                    groups.append({'sample': sample, 'delay_s': delay, 'version': version,
                        **{key: statistics.median(r[key] for r in matched)
                           for key in ('request_s', 'cpu_s', 'communication_us', 'prepare_s')}})
    return {'groups': groups, 'agreements': agreements,
            'all_schedules_equal': all(r['complete'] and r['same_schedule'] for r in agreements),
            'all_prompts_equal': all(r['complete'] and r['same_prompt'] for r in agreements)}


async def run(args):
    import httpx
    from agenthack import AgentRuntime
    from agenthack.telemetry import Telemetry
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    versions = compare.resolve_versions(args.versions)
    samples = compare.load_samples(args.samples, all_public=not args.samples)
    output = compare.check_output(args.output_dir)
    saved = recordings(compare.within_project(args.recordings), samples)
    report = {'scope': 'fixed original model replies; no network; official SDK ASGI requests',
              'actual_model_calls': 0, 'actual_tokens': 0,
              'versions': [v.label for v in versions], 'samples': [s.label for s in samples],
              'delays_s': args.delays, 'repeat': args.repeat, 'limits': compare.apply_limits(),
              'source_before': compare.source_hashes(versions), 'rows': []}
    async with AsyncExitStack() as stack:
        for name in ('connect', 'connect_ex'):
            stack.enter_context(patch.object(socket.socket, name, side_effect=no_network))
        for name in ('create_connection', 'getaddrinfo'):
            stack.enter_context(patch.object(socket, name, side_effect=no_network))
        clients, modules = {}, {}
        provider = TracerProvider(resource=Resource({'service.name': 'agentcc-native-comparison'}))
        stack.callback(provider.shutdown)
        for version in versions:
            agent = compare.load_agent(version)
            package = agent.__class__.__module__.rsplit('.', 1)[0]
            modules[version.label] = {name: importlib.import_module(package + '.' + name)
                                     for name in ('generated_service', 'code_policy', 'config')}
            stack.enter_context(patch.object(modules[version.label]['config'], 'read_env_file', side_effect=no_network))
            runtime = AgentRuntime(agent, telemetry=Telemetry(provider=provider))
            await stack.enter_async_context(runtime.app.router.lifespan_context(runtime.app))
            client = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=runtime.app), base_url='http://official-sdk-test'))
            assert (await client.get('/health')).status_code == 200
            clients[version.label] = client
        for repeat in range(args.repeat):
            for index, sample in enumerate(samples):
                original = saved[sample.label]
                for delay in args.delays:
                    order = compare.rotating_order(versions, index, repeat)
                    for version in order:
                        observed = []

                        def complete(system, user, settings, **_kwargs):
                            observed.append([system, user])
                            started = time.monotonic()
                            if delay:
                                time.sleep(delay)
                            return original['response_evidence']['content'], {
                                'called': True, 'synthetic': True, 'usage': {},
                                'model': original['model'], 'finish_reason': original['finish_reason'],
                                'duration_ms': 1000 * (time.monotonic() - started)}

                        selected = modules[version.label]
                        settings = selected['config'].Settings('http://unused.invalid', '', original['model'])
                        with ExitStack() as request:
                            request.enter_context(patch.object(selected['generated_service'], 'load_settings', return_value=settings))
                            request.enter_context(patch.object(selected['code_policy'], '_chat_completion', side_effect=complete))
                            gc.collect()
                            begin, cpu = time.monotonic(), time.process_time()
                            response = await clients[version.label].post('/solve', json=sample.request())
                            data = response.json()
                            elapsed, cpu_elapsed = time.monotonic() - begin, time.process_time() - cpu
                        measured = compare.response_measurements(sample, data, response.status_code)
                        path = f'v{version.label}/{sample.label.replace("/", "-")}-r{repeat+1:02d}-delay{delay:g}.json'
                        compare.write_json(output / path, data)
                        if not measured['valid']:
                            raise RuntimeError(f'failed SDK request: {path}')
                        answer = data['output']['answer']
                        diagnostics = data['output']['diagnostics']
                        row = {'sample': sample.label, 'repeat': repeat+1, 'version': version.label,
                               'delay_s': delay, 'request_s': elapsed, 'cpu_s': cpu_elapsed,
                               'communication_us': measured['time_us'], 'valid': True,
                               'prepare_s': diagnostics['phases_s']['capacity_space_s'],
                               'capacity_backend': diagnostics.get('capacity_backend', 'python'),
                               'schedule_hash': digest(answer), 'prompt_hash': digest(observed),
                               'simulated_calls': len(observed), 'response_file': path}
                        report['rows'].append(row)
                        compare.write_json(output / 'summary.json', report)
                        print(json.dumps({k: row[k] for k in ('sample', 'version', 'delay_s', 'request_s', 'cpu_s', 'capacity_backend')}), flush=True)
    report['source_after'] = compare.source_hashes(versions)
    report['source_unchanged'] = report['source_before'] == report['source_after']
    report['complete'] = len(report['rows']) == len(samples)*len(versions)*len(args.delays)*args.repeat
    report.update(summary(report))
    compare.write_json(output / 'summary.json', report)
    return 0 if report['complete'] and report['source_unchanged'] and report['all_schedules_equal'] and report['all_prompts_equal'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--versions', nargs='+', default=['1.7', 'current'])
    parser.add_argument('--samples', nargs='+')
    parser.add_argument('--delays', type=float, nargs='+', default=[0, 5])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--recordings', default='artifacts/v1.7/runs/release-sdk')
    parser.add_argument('--output-dir', default='artifacts/v1.8/runs/fixed-model')
    args = parser.parse_args()
    if args.repeat < 1 or any(not 0 <= delay <= 60 for delay in args.delays):
        parser.error('positive repeat count and finite delays between 0 and 60 required')
    raise SystemExit(asyncio.run(run(args)))
