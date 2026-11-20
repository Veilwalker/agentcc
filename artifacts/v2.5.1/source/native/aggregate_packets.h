// A copy delivered to the same receiver in the same round leaves every
// round-boundary state invariant, provided its new sender already owns it.
namespace aggregate_packets {
using Terms=std::vector<std::pair<int,int>>;
struct Route {
    int src,dst;
    PyObject* path;  // first sorted route key is retained by the Python dictionary
    double delay, serial;
    Terms terms;
};
struct Group {
    int route;
    int width;
    PyObject* prototype;
    std::vector<int> payload;
};
struct Packing {
    CapacityGraph& graph;
    double size;
    int nranks, ntokens;
    PyObject* rank_ids;
    Owned tokens{PyDict_New()}, paths{PyDict_New()};
    std::vector<PyObject*> names;
    std::vector<unsigned char> ready;
    std::vector<Route> routes;
    std::vector<double> rate;
    long long moved=0, saved=0;
    Owned payload_key{PyUnicode_InternFromString("chunk_ranges")}, path_key{PyUnicode_InternFromString("path")}, transfer_key{PyUnicode_InternFromString("transfers")};
    Packing(CapacityGraph& graph,double size,int nranks,int ntokens,PyObject* ranks)
        :graph(graph),size(size),nranks(nranks),ntokens(ntokens),rank_ids(ranks),ready(nranks*ntokens) {}
    int index(PyObject* d,PyObject* k) {
        auto val=k?PyDict_GetItemWithError(d,k):nullptr;
        if (!val) throw std::invalid_argument("unknown scheduling key");
        long id=PyLong_AsLong(val);
        if (id<0 || PyErr_Occurred()) throw std::invalid_argument("invalid index");
        return id;
    }
    void init(PyObject* initial) {
        if (!tokens.p || !paths.p || !payload_key.p || !path_key.p || !transfer_key.p) throw std::bad_alloc();
        rate.resize(graph.edges.size()+graph.pools.size());
        for (auto& entry:graph.edges) {
            const auto& e=entry.second;
            rate.at(PyLong_AsLong(PyTuple_GET_ITEM(e.term,0)))=size*e.inverse;
        }
        for (auto [id,bw]:graph.pools) rate.at(id)=size*1e6/bw;
        Py_ssize_t pos=0;PyObject *r,*parts;
        while(PyDict_Next(initial,&pos,&r,&parts)) {
            int rank=index(rank_ids,r);
            if (!PyList_Check(parts) || rank>=nranks) throw std::invalid_argument("invalid possession");
            for(Py_ssize_t j=0;j<PyList_GET_SIZE(parts);++j) {
                auto t=PyList_GET_ITEM(parts,j);int id=names.size();
                if(id>=ntokens) throw std::invalid_argument("too many atoms");
                Owned atom(PyLong_FromLong(id));
                if(!atom.p || PyDict_SetItem(tokens.p,t,atom.p)<0)throw std::bad_alloc();
                names.push_back(t);ready[rank*ntokens+id]=1;
            }
        }
        if(int(names.size())!=ntokens)throw std::invalid_argument("missing atoms");
    }
    int route(PyObject* path) {
        Owned key(PySequence_Tuple(path));
        if(!key.p || PyTuple_GET_SIZE(key.p)<2)throw std::invalid_argument("bad route");
        auto v=PyDict_GetItemWithError(paths.p,key.p);
        if(v)return PyLong_AsLong(v);
        int id=routes.size();
        Owned rid(PyLong_FromLong(id));
        if(!rid.p || PyDict_SetItem(paths.p,key.p,rid.p)<0)throw std::bad_alloc();
        int n=PyTuple_GET_SIZE(key.p);
        int s=index(rank_ids,PyTuple_GET_ITEM(key.p,0)),d=index(rank_ids,PyTuple_GET_ITEM(key.p,n-1));
        Route r{s,d,key.p,graph.startup+(n-2)*graph.switching,0.,{}};
        std::vector<int> keys;uint64_t u;
        if(!capacity_node(graph,PyTuple_GET_ITEM(key.p,0),u,false))throw std::invalid_argument("unknown route start");
        for(int j=1;j<n;++j) {
            uint64_t w;
            if(!capacity_node(graph,PyTuple_GET_ITEM(key.p,j),w,false))throw std::invalid_argument("unknown route node");
            auto it=graph.edges.find((u<<32)|w);if(it==graph.edges.end())throw std::invalid_argument("undeclared hop");
            const auto& edge=it->second;r.serial+=edge.inverse;
            keys.push_back(PyLong_AsLong(PyTuple_GET_ITEM(edge.term,0)));
            for(auto k:edge.pools)keys.push_back(graph.pools[k].first);
            u=w;
        }
        std::sort(keys.begin(),keys.end());
        for (auto k:keys) {
            if(r.terms.empty() || r.terms.back().first!=k)r.terms.emplace_back(k,1);
            else ++r.terms.back().second;
        }
        routes.push_back(std::move(r));return id;
    }
    int room(const Route& a,const Route& b,const std::vector<int>& cap,
             const std::vector<int>& load,int limit) {
        size_t i=0;
        for(auto [key,mult]:b.terms) {
            while(i<a.terms.size() && a.terms[i].first<key)++i;
            if(i<a.terms.size() && a.terms[i].first==key)mult-=a.terms[i].second;
            if(mult>0) {
                limit=std::min(limit,(cap[key]-load[key])/mult);
                if(limit<=0)return 0;
            }
        }
        return limit;
    }
    PyObject* compress(PyObject* row,double duration) {
        std::vector<Group> groups;
        std::vector<std::vector<int>> by_dest(nranks);
        std::vector<int> load(rate.size()),cap(rate.size());
        std::unordered_map<int,int> table;
        double max_delay=0;
        auto transfers=PyDict_GetItemWithError(row,transfer_key.p);
        if(!transfers || !PyList_Check(transfers))throw std::invalid_argument("missing transfers");
        for(Py_ssize_t i=0;i<PyList_GET_SIZE(transfers);++i) {
            auto tr=PyList_GET_ITEM(transfers,i);
            auto path=PyDict_GetItemWithError(tr,path_key.p);
            auto atoms=PyDict_GetItemWithError(tr,payload_key.p);
            if(!path || !atoms || !PyList_Check(atoms))throw std::invalid_argument("invalid gather move");
            int pid=route(path);const auto& r=routes[pid];
            auto [where,newly]=table.emplace(pid,groups.size());
            if(newly) {
                int width=std::max(1,int(std::min(double(ntokens),std::floor((duration-r.delay)/(size*r.serial)+1e-8))));
                by_dest[r.dst].push_back(groups.size());
                groups.push_back({pid,width,tr,{}});
                max_delay=std::max(max_delay,r.delay);
            }
            auto& g=groups[where->second];
            int n=PyList_GET_SIZE(atoms);
            for(int j=0;j<n;++j)g.payload.push_back(index(tokens.p,PyList_GET_ITEM(atoms,j)));
            for(auto [k,m]:r.terms)load[k]+=n*m;
        }
        for(size_t i=0;i<rate.size();++i) {
            const double bound=std::floor((duration-max_delay)/rate[i]+1e-7);
            cap[i]=std::max(load[i],int(std::clamp(bound,0.,1e9)));
        }
        for(auto& possible:by_dest) {
            std::stable_sort(possible.begin(),possible.end(),[&](int i,int j){
                return groups[i].payload.size()<groups[j].payload.size();});
            for(size_t i=0;i<possible.size();++i) {
                auto& a=groups[possible[i]];const auto& srcpath=routes[a.route];
                for(size_t j=possible.size();j-->i+1 && !a.payload.empty();) {
                    auto& b=groups[possible[j]];
                    auto count=b.payload.size();
                    if(!count || count%b.width==0)continue;
                    const auto& dstpath=routes[b.route];
                    const auto* held=ready.data()+dstpath.src*ntokens;
                    if (std::none_of(a.payload.begin(),a.payload.end(),[&](int token){return held[token]!=0;})) continue;
                    int space=room(srcpath,dstpath,cap,load,b.width-count%b.width);
                    if(!space)continue;
                    auto& data=a.payload;
                    size_t keep=0;int take=0;
                    for(int token:data) {
                        if(take<space && held[token]) {
                            b.payload.push_back(token);++take;
                        } else data[keep++]=token;
                    }
                    if(!take)continue;
                    data.resize(keep);
                    for(auto [k,m]:srcpath.terms)load[k]-=take*m;
                    for(auto [k,m]:dstpath.terms)load[k]+=take*m;
                    moved+=take;
                }
            }
        }
        size_t packet_count=0;
        for(auto& g:groups) {
            auto& r=routes[g.route];
            for(int token:g.payload)ready[r.dst*ntokens+token]=1;
            packet_count+=(g.payload.size()+g.width-1)/g.width;
        }
        if(packet_count>=size_t(PyList_GET_SIZE(transfers)))return Py_NewRef(row);
        saved+=PyList_GET_SIZE(transfers)-packet_count;
        Owned emitted(PyList_New(packet_count));if(!emitted.p)throw std::bad_alloc();
        size_t ix=0;
        for(auto& g:groups) {
            for(size_t first=0;first<g.payload.size();first+=g.width) {
                size_t n=std::min<size_t>(g.width,g.payload.size()-first);
                Owned payload(PyList_New(n));
                if(!payload.p)throw std::bad_alloc();
                for(size_t j=0;j<n;++j)PyList_SET_ITEM(payload.p,j,Py_NewRef(names[g.payload[first+j]]));
                Owned t(PyDict_Copy(g.prototype));
                if(!t.p || PyDict_SetItem(t.p,payload_key.p,payload.p)<0)throw std::bad_alloc();
                PyList_SET_ITEM(emitted.p,ix++,t.release());
            }
        }
        Owned fresh(PyDict_Copy(row));if(!fresh.p || PyDict_SetItemString(fresh.p,"transfers",emitted.p)<0)throw std::bad_alloc();
        return fresh.release();
    }
};
PyObject* run(PyObject*,PyObject* args) {
    PyObject *handle,*initial,*ranks,*rounds,*times;int split;double size;
    if(!PyArg_ParseTuple(args,"OO!O!O!O!id",&handle,&PyDict_Type,&initial,&PyDict_Type,&ranks,
        &PyList_Type,&rounds,&PyList_Type,&times,&split,&size))return nullptr;
    auto g=static_cast<CapacityGraph*>(PyCapsule_GetPointer(handle,capacity_graph_name));if(!g)return nullptr;
    int n=PyDict_Size(ranks);if(n<2||n>512||split<1||split>128||n*n*split>3000000||size<=0||
       PyList_GET_SIZE(rounds)!=PyList_GET_SIZE(times)){PyErr_SetString(PyExc_ValueError,"invalid packing dimensions");return nullptr;}
    try {
        Packing pack(*g,size,n,n*split,ranks);pack.init(initial);
        Owned result(PyList_New(PyList_GET_SIZE(rounds)));if(!result.p)return nullptr;
        for(Py_ssize_t i=0;i<PyList_GET_SIZE(rounds);++i) {
            auto row=PyList_GET_ITEM(rounds,i);
            auto next=pack.compress(row,PyFloat_AsDouble(PyList_GET_ITEM(times,i)));
            if(!next)return nullptr;
            PyList_SET_ITEM(result.p,i,next);
        }
        return Py_BuildValue("NLL",result.release(),pack.moved,pack.saved);
    } catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
}
