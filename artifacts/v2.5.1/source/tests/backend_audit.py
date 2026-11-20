"""Same-source backend isolation, using application files from exported images.

Fixed replies separate compute from model variability. Live mode measures both;
neither claims to reproduce a platform collector or platform CPU quota.
"""
import argparse
import asyncio
from contextlib import AsyncExitStack, ExitStack
import gc
import importlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from tests import compare_versions as compare
from tests.native_comparison import digest, recordings


async def run(args):
    import httpx
    from agenthack import AgentRuntime
    samples=compare.load_samples(args.samples,all_public=not args.samples)
    output=compare.check_output(args.output)
    if args.live:
        compare.configure_environment()
    saved={v: recordings(compare.PROJECT/f'artifacts/v{v}/runs'/folder,samples,v)
           for v,folder in [('1.9','live-sdk'),('1.10','bounded-output-sdk')]} if not args.live else {}
    report={'scope':'exported app on extracted official runtime; no container CPU quota or remote collector',
            'mode':'live' if args.live else 'fixed-original-replies',
            'limits':compare.apply_limits(),'rows':[],'repeat':args.repeat}
    variants=[('1.6','python'),('1.9','python'),('1.9','cpp'),('1.10','python'),('1.10','cpp')]
    if args.versions:
        variants=[x for x in variants if x[0] in args.versions]
    async with AsyncExitStack() as stack:
        cases=[]
        for index,(v,backend) in enumerate(variants):
            source=compare.PROJECT/f'artifacts/v1.11/analysis/images/image-{v}/app/src'
            with patch.dict(os.environ,AGENTCC_CAPACITY_BACKEND=backend):
                agent=compare.load_agent(compare.Version(f'{v}.{901+index}',source,False),
                                         enforce_release_backend=False)
            package=type(agent).__module__.rsplit('.',1)[0]
            module=importlib.import_module(type(agent).__module__)
            service=importlib.import_module(module.solve_async.__module__)
            transport=importlib.import_module(package+('.code_policy' if v=='1.10' else '.llm'))
            config=importlib.import_module(package+'.config')
            runtime=AgentRuntime(agent)
            await stack.enter_async_context(runtime.app.router.lifespan_context(runtime.app))
            client=await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=runtime.app),base_url='http://sdk-test'))
            cases.append((v,backend,client,service,transport,config,runtime))
        for repeat in range(args.repeat):
            for index,sample in enumerate(samples):
                offset=(index+repeat)%len(cases)
                for v,backend,client,service,transport,config,runtime in cases[offset:]+cases[:offset]:
                    recorded=saved['1.10' if v=='1.10' else '1.9'][sample.label] if not args.live else None
                    prompts=[]
                    def complete(system,user,settings,**kwargs):
                        prompts.append([system,user])
                        return recorded['response_evidence']['content'],{
                            'called':True,'synthetic':True,'usage':{},'model':recorded['model'],
                            'finish_reason':recorded['finish_reason'],'duration_ms':0}
                    with ExitStack() as request:
                        if not args.live:
                            settings=config.Settings('http://unused.invalid','',recorded['model'])
                            request.enter_context(patch.object(service,'load_settings',return_value=settings))
                            request.enter_context(patch.object(transport,'_chat_completion',side_effect=complete))
                        gc.collect()
                        begin,cpu=time.monotonic(),time.process_time()
                        response=await client.post('/solve',json=sample.request())
                        data=response.json()
                        elapsed,compute=time.monotonic()-begin,time.process_time()-cpu
                    row=compare.response_measurements(sample,data,response.status_code)
                    row.update(version=v,backend=backend,sample=sample.label,repeat=repeat+1,
                        request_s=elapsed,cpu_s=compute,response_bytes=len(response.content),
                        sdk_session_s=data.get('metadata',{}).get('duration_ms',0)/1000,
                        service_s=data.get('output',{}).get('diagnostics',{}).get('elapsed_s'),
                        prompt_hash=digest(prompts),schedule_hash=digest(data.get('output',{}).get('answer')))
                    file=f'{v}-{backend}/{sample.label.replace("/","-")}-{repeat+1}.json'
                    compare.write_json(output/file,data)
                    row['response_file']=file
                    report['rows'].append(row)
                    compare.write_json(output/'summary.json',report)
                    print(json.dumps({k:row.get(k) for k in ('version','backend','sample','valid','request_s','cpu_s','llm_s','time_us')}),flush=True)
    report['complete']=len(report['rows'])==args.repeat*len(samples)*len(cases)
    compare.write_json(output/'summary.json',report)
    return 0 if report['complete'] and all(r['valid'] for r in report['rows']) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live',action='store_true')
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--samples',nargs='+')
    p.add_argument('--versions',nargs='+')
    p.add_argument('--output',default='artifacts/v1.11/runs/backend-fixed')
    args=p.parse_args()
    raise SystemExit(asyncio.run(run(args)))
