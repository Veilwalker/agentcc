"""Change the physical placement of a slot's logical reduction tree."""
from collections import defaultdict
from .capacity_backend import native
from .congestion_routes import graph_data
from .replay import certify


def improve(task,graph,answer,checked,catalog,trials=1536):
    if task['operator']!='all-reduce' or len(graph.ranks)>256:
        return answer,checked
    split=answer['split']; size=task['data_size']['per_rank_bytes']//split
    edges,rates,pairs,nodes=graph_data(graph,size)
    edge_ids={edge:i for i,edge in enumerate(pairs)}
    paths={};moves=[];copy=[];slots={}
    rounds=len(answer['rounds'])
    for r,row in enumerate(answer['rounds']):
        for tr in row['transfers']:
            path=tuple(tr['path']);p=paths.setdefault(path,len(paths))
            for token in tr['chunk_ranges']:
                if token not in slots:
                    slots[token]=int(token[1:-4]) if token.endswith('_sum') else int(token.rsplit('_c',1)[1])
                moves.append((r,slots[token],p))
                copy.append(tr['action']=='copy')
    encoded=[tuple(edge_ids[e] for e in zip(path,path[1:])) for path in paths]
    result=native.exchange_relays(len(graph.ranks),nodes,rounds,split,edges,rates,
                                 graph.startup_us,graph.switch_overhead_us,encoded,moves,trials,checked['total_time_us'])
    if result is None:return answer,checked
    predicted,physical,selection=result
    actual={};state=[[f'{r}_c{s}' for r in graph.ranks] for s in range(split)]
    rows=[];batches=defaultdict(list);arrivals=defaultdict(list)
    def finish():
        for (s,dst),seq in arrivals.items():
            state[s][dst]='+'.join([state[s][dst],*seq])
        arrivals.clear()
        rows.append(batches.copy());batches.clear()
    for (r,s,_),iscopy,j in zip(moves,copy,selection):
        if r!=len(rows):finish()
        if j not in actual:
            ids=physical[j];actual[j]=(pairs[ids[0]][0],*(pairs[e][1] for e in ids))
        path=actual[j]
        if iscopy:token=f'c{s}_sum'
        else:
            token=state[s][catalog.rank_index[path[0]]]
            arrivals[s,catalog.rank_index[path[-1]]].append(token)
        batches[path,iscopy].append(token)
    finish()
    schedule=[]
    for row in rows:
        loads=defaultdict(float);latency=serial=0.
        for (p,_),ts in row.items():
            terms,lat,inv,_=catalog.capacity_costs.compile(p)
            latency=max(latency,lat);serial=max(serial,lat+inv*size)
            for k,rate in terms:loads[k]+=rate*size*len(ts)
        ceiling=max(serial,latency+max(loads.values(),default=0))
        transfers=[]
        for (p,iscopy),ts in row.items():
            _,lat,inv,_=catalog.capacity_costs.paths[p]
            width=max(1,int((ceiling-lat)/(size*inv)+1e-10))
            for k in range(0,len(ts),width):
                transfers.append(dict(src=p[0],dst=p[-1],path=list(p),action='copy' if iscopy else 'reduce',chunk_ranges=ts[k:k+width]))
        schedule.append(dict(round=len(schedule)+1,transfers=transfers))
    candidate=dict(answer,rounds=schedule)
    report=certify(task,graph,candidate)
    if report['valid'] and abs(report['total_time_us']-predicted)<max(1e-4,predicted*1e-9):
        return candidate,report
    return answer,checked
