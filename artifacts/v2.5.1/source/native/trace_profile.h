// Price the complete recursive-doubling / reduction-tree trace in a batch.
// Loads account for repeated visits to a shared pool. Resource encounter
// order is retained for equal peaks and for the public trace's rounding.
namespace trace_profile {
struct Edge {
    size_t id;
    std::vector<size_t> pools;
};
struct Topology {
    Owned edge_ids{PyDict_New()}, pool_ids{PyDict_New()}, names{PyList_New(0)};
    PyObject *links, *bindings;
    double scale;
    std::vector<double> rate;
    std::vector<Edge> cache;
    Topology(PyObject* edges,PyObject* members,double bytes)
        : links(edges),bindings(members),scale(bytes*1e6) {}

    bool add_rate(PyObject* name,PyObject* capacity) {
        double bw=PyFloat_AsDouble(capacity);
        if(PyErr_Occurred()) return false;
        if(!std::isfinite(bw) || bw<=0 || !std::isfinite(scale/bw) || !(scale/bw>0)) {
            PyErr_SetString(PyExc_ValueError,"invalid trace bandwidth");return false;
        }
        if(PyList_Append(names.p,name)<0) return false;
        rate.push_back(scale/bw);
        return true;
    }
    bool init(PyObject* capacities) {
        if(!edge_ids.p || !pool_ids.p || !names.p) return false;
        Py_ssize_t off=0;PyObject *name,*bw;
        while(PyDict_Next(capacities,&off,&name,&bw)) {
            Owned id(PyLong_FromSize_t(rate.size()));
            if(!id.p || PyDict_SetItem(pool_ids.p,name,id.p)<0 || !add_rate(name,bw)) return false;
        }
        return true;
    }
    const Edge* edge(PyObject* u,PyObject* v) {
        Owned key(PyTuple_Pack(2,u,v));
        if(!key.p) return nullptr;
        PyObject* old=PyDict_GetItemWithError(edge_ids.p,key.p);
        if(old) return &cache[PyLong_AsSize_t(old)];
        if(PyErr_Occurred()) return nullptr;
        PyObject* bw=PyDict_GetItemWithError(links,key.p);
        PyObject* pools=PyDict_GetItemWithError(bindings,key.p);
        if(!bw || !pools || !PyTuple_Check(pools)) {
            if(!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,"unknown trace edge");
            return nullptr;
        }
        Edge entry{rate.size(),{}};
        for(int i=0;i<PyTuple_GET_SIZE(pools);++i) {
            PyObject* pool=PyDict_GetItemWithError(pool_ids.p,PyTuple_GET_ITEM(pools,i));
            if(!pool) {if(!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,"unknown trace pool");return nullptr;}
            entry.pools.push_back(PyLong_AsSize_t(pool));
        }
        Owned id(PyLong_FromSize_t(cache.size()));
        if(!id.p || PyDict_SetItem(edge_ids.p,key.p,id.p)<0 || !add_rate(key.p,bw)) return nullptr;
        cache.push_back(std::move(entry));
        return &cache.back();
    }
};
PyObject* price(PyObject*,PyObject* args) {
    int ranks,split,gather; double bytes,startup,switching;
    PyObject *edges,*bindings,*pools,*rows;
    if(!PyArg_ParseTuple(args,"iipdddO!O!O!O!",&ranks,&split,&gather,&bytes,&startup,&switching,
                         &PyDict_Type,&edges,&PyDict_Type,&bindings,&PyDict_Type,&pools,&PyList_Type,&rows)) return nullptr;
    int levels=0;
    while(levels<9 && (1<<levels)<ranks) ++levels;
    const int count=levels*(gather?1:2);
    Py_ssize_t resources=PyDict_Size(edges)+PyDict_Size(pools);
    if(ranks<2 || ranks>256 || (1<<levels)!=ranks || split<1 || split>65536 || !std::isfinite(bytes)
       || bytes<=0 || PyList_GET_SIZE(rows)!=count || resources<1 || resources>2000000) {
        PyErr_SetString(PyExc_ValueError,"invalid trace size");return nullptr;
    }
    try {
        Topology graph(edges,bindings,bytes);
        if(!graph.init(pools)) return nullptr;
        std::vector<double> loads(resources),room(resources);
        std::vector<int> last_seen(resources,-1),ready(ranks);
        std::vector<size_t> active;
        std::vector<std::pair<size_t,double>> terms;
        std::vector<std::pair<size_t,int>> multiplicity;
        ready[0]=levels;
        double last_duration=0,last_delay=0,last_peak=0;
        Owned out(PyList_New(count));
        if(!out.p) return nullptr;
        for(int step=0;step<count;++step) {
            active.clear();
            const bool reduce=!gather && step<levels;
            const int level=(gather||reduce)?step:2*levels-1-step;
            const int stride=1<<level;
            PyObject* paths=PyList_GET_ITEM(rows,step);
            if(!PyList_Check(paths) || PyList_GET_SIZE(paths)!=(gather?ranks:ranks/(2*stride))) {
                PyErr_SetString(PyExc_ValueError,"invalid trace round");return nullptr;
            }
            int64_t fit=0,deliveries=0;
            double delay=0,serial=0;
            for(int j=0;j<PyList_GET_SIZE(paths);++j) {
                int base=gather?j:j*2*stride;
                int atoms=split*(gather?stride:1);
                int eligible=gather?(step?atoms/2:0):(!reduce && ready[base]<step?split:0);
                PyObject* path=PyList_GET_ITEM(paths,j);
                if(!PyList_Check(path) || PyList_GET_SIZE(path)<2) {
                    PyErr_SetString(PyExc_ValueError,"invalid trace path");return nullptr;
                }
                terms.clear();multiplicity.clear();
                double serial_time=0;
                for(int k=1;k<PyList_GET_SIZE(path);++k) {
                    const Edge* edge=graph.edge(PyList_GET_ITEM(path,k-1),PyList_GET_ITEM(path,k));
                    if(!edge) return nullptr;
                    terms.emplace_back(edge->id,graph.rate[edge->id]);
                    serial_time+=graph.rate[edge->id];
                    for(size_t pool:edge->pools) {
                        auto it=std::find_if(multiplicity.begin(),multiplicity.end(),
                                             [&](const auto& entry){return entry.first==pool;});
                        if(it==multiplicity.end()) multiplicity.emplace_back(pool,1);
                        else ++it->second;
                    }
                }
                for(auto [id,times]:multiplicity) terms.emplace_back(id,graph.rate[id]*times);
                double latency=startup+(PyList_GET_SIZE(path)-2)*switching;
                serial_time+=latency;
                delay=std::max(delay,latency);
                serial=std::max(serial,serial_time);
                for(auto [id,unit]:terms) {
                    if(last_seen[id]!=step) {
                        last_seen[id]=step;active.push_back(id);loads[id]=0.;
                    }
                    loads[id]+=atoms*unit;
                }
                deliveries+=atoms;
                if(eligible && serial_time<=last_duration) {
                    double budget=last_duration-std::max(last_delay,latency);
                    int take=eligible;
                    if(last_peak>budget+1e-9) take=0;
                    else for(auto [id,unit]:terms) {
                        double free=(budget-room[id])/unit+1e-9;
                        if(free<take) take=free<=0 ? 0 : static_cast<int>(free);
                    }
                    if(take) {
                        fit+=take;
                        last_delay=std::max(last_delay,latency);
                        for(auto [id,unit]:terms) {
                            room[id]+=take*unit;
                            last_peak=std::max(last_peak,room[id]);
                        }
                    }
                }
                if(!gather && !reduce) ready[base+stride]=step+1;
            }
            double peak=-1,sum=0;size_t hot=0;
            std::fill(room.begin(),room.end(),0.);
            for(size_t id:active) {
                if(loads[id]>peak) {hot=id;peak=loads[id];}
                sum+=loads[id];
                room[id]=loads[id];
            }
            last_peak=peak;last_delay=delay;last_duration=std::max(serial,peak+delay);
            double available=active.size()*(last_duration-delay);
            if(!(available>0)) {PyErr_SetString(PyExc_ValueError,"trace duration overflow");return nullptr;}
            PyObject* row=Py_BuildValue("(Oddd)",PyList_GET_ITEM(graph.names.p,hot),last_duration,
                                        100*(1-sum/available),100.*fit/deliveries);
            if(!row) return nullptr;
            PyList_SET_ITEM(out.p,step,row);
        }
        return out.release();
    } catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
}
