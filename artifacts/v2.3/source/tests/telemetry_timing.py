"""Measure the official SDK's response barrier with a controlled local exporter.

This is a sensitivity experiment, NOT a reconstruction of platform telemetry
latency. No SDK code, environment flush switch, network, or credentials change.
"""
import asyncio
import json
from pathlib import Path
import time


async def main():
    import httpx
    from agenthack import AgentRuntime, BaseAgent
    from agenthack.telemetry import Telemetry
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

    class ProbeAgent(BaseAgent):
        async def solve(self, task_input):
            return {'probe': True}

    class ControlledExporter(SpanExporter):
        def __init__(self, delay):
            self.delay, self.calls = delay, []

        def export(self, spans):
            started = time.monotonic()
            time.sleep(self.delay)
            self.calls.append({'spans': len(spans), 'export_s': time.monotonic()-started})
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    rows = []
    for delay in (0, .2, 6):
        exporter = ControlledExporter(delay)
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=500))
        runtime = AgentRuntime(ProbeAgent(), telemetry=Telemetry(provider=provider))
        async with runtime.app.router.lifespan_context(runtime.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url='http://sdk-test') as client:
                begin = time.monotonic()
                response = await client.post('/solve', json={'task_id': 'flush-probe', 'task_type': 'agentcc',
                    'input': {}, 'constraints': {'timeout_ms': 360000}})
                elapsed = time.monotonic()-begin
                row = {'synthetic_export_delay_s': delay, 'request_s': elapsed,
                       'status': response.status_code, 'metadata': response.json().get('metadata', {})}
                rows.append(row)
        provider.shutdown()
        row['exports'] = exporter.calls
    target = Path('artifacts/v1.11/analysis/telemetry-timing.json')
    if target.exists():
        raise ValueError('keep prior evidence')
    target.write_text(json.dumps({'scope': __doc__, 'rows': rows}, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
