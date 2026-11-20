// Local search on the embedding of a validated reduction DAG. Each slot may
// use a different permutation of physical ranks. Swapping two vertices changes
// only their incident transmissions; the logical contributor DAG is preserved.
namespace flow_routes {
struct Transfer { int round,slot,u,v,path; };
struct Exchange {
    Optimizer paths;
    const Network& graph;
    int rounds,slots;
    std::vector<Transfer> moves;
    std::vector<std::vector<int>> incident;
    std::vector<std::vector<double>> loads;
    std::vector<double> scale;
    std::vector<std::vector<int>> next_edge;
    std::vector<std::unordered_map<int,int>> new_paths;
    std::vector<std::unordered_map<int,std::vector<int>>> existing;
    std::vector<std::vector<double>> edge_weight;
    unsigned seed=1729;
    Exchange(const Network& graph,int rounds,int slots): paths(graph), graph(graph),
        rounds(rounds),slots(slots), incident(slots*graph.ranks),
        loads(rounds,std::vector<double>(graph.rate.size())),scale(rounds),
        next_edge(rounds*graph.ranks),new_paths(rounds),existing(rounds),edge_weight(rounds) {}
    unsigned random() {seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
    void change(int round,int id,int sign) {
        for(auto [k,c]:paths.routes[id].terms) loads[round][k]+=sign*c*graph.rate[k];
    }
    void refresh() {
        for(int r=0;r<rounds;++r) {
            scale[r]=1/std::max(1e-9,*std::max_element(loads[r].begin(),loads[r].end()));
            auto& weights=edge_weight[r];weights.clear();
            for(auto& e:graph.edges) {
                double v=0;
                for(int k:e.ids) v+=graph.rate[k]*(.007+Optimizer::power(loads[r][k]*scale[r]));
                weights.push_back(v+.02*graph.rate[e.ids.front()]);
            }
            new_paths[r].clear();
        }
        for(auto& row:next_edge)row.clear();
    }
    double cost(int r,int id) const {
        const auto& route=paths.routes[id];
        double value=0;
        for(auto [k,c]:route.terms) {
            double volume=c*graph.rate[k], normalized=(loads[r][k]+volume*.5)*scale[r];
            value+=volume*(.002+Optimizer::power(std::max(0.,normalized)));
        }
        double penalty=(route.overhead+route.serial)*scale[r];
        return value+.12*route.serial*(1+Optimizer::power(penalty));
    }
    int pick(int r,int u,int v) {
        int pair=u*graph.ranks+v;
        auto& cache=new_paths[r]; auto got=cache.find(pair);
        int fresh=-1;
        if(got!=cache.end()) fresh=got->second;
        else {
            auto& tree=next_edge[r*graph.ranks+v];
            if(tree.empty()) {
                paths.weights=edge_weight[r];paths.shortest(v);
                tree=paths.pred;
                for(int node=0;node<graph.nodes;++node)
                    if(!std::isfinite(paths.dist[node]))tree[node]=-1;
            }
            std::vector<int> p;
            for(int node=u;node!=v;) {
                int e=tree[node];
                if(e<0 || p.size()>=64) {p.clear();break;}
                p.push_back(e);node=graph.edges[e].v;
            }
            if(!p.empty()) fresh=paths.intern(p);
            cache[pair]=fresh;
        }
        auto old=existing[r].find(pair);
        if(old==existing[r].end())return fresh;
        double best=fresh<0 ? std::numeric_limits<double>::infinity() : cost(r,fresh);
        for(int id:old->second) {
            double v=cost(r,id);
            if(v<best){best=v;fresh=id;}
        }
        return fresh;
    }
    double duration() const {
        std::vector<double> h(rounds),serial(rounds);
        for(auto& m:moves) {
            auto& p=paths.routes[m.path];
            h[m.round]=std::max(h[m.round],p.overhead);
            serial[m.round]=std::max(serial[m.round],p.overhead+p.serial);
        }
        double sum=0;
        for(int r=0;r<rounds;++r) sum+=std::max(serial[r],h[r]+*std::max_element(loads[r].begin(),loads[r].end()));
        return sum;
    }
    void trial(int slot,int a,int b) {
        auto key=[&](int u){return slot*graph.ranks+u;};
        auto swap=[&](int u){return u==a?b:u==b?a:u;};
        auto ids=incident[key(a)];
        ids.insert(ids.end(),incident[key(b)].begin(),incident[key(b)].end());
        std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        if(ids.empty()||ids.size()>128)return;
        for(int j:ids){auto& m=moves[j];change(m.round,m.path,-1);}
        double before=0,after=0;
        std::vector<int> replacements;replacements.reserve(ids.size());
        for(int j:ids) {
            auto& m=moves[j];int id=pick(m.round,swap(m.u),swap(m.v));
            if(id<0){after=std::numeric_limits<double>::infinity();break;}
            before+=cost(m.round,m.path);
            after+=cost(m.round,id);
            change(m.round,id,1);
            replacements.push_back(id);
        }
        if(after<before*(1.-1e-5)) {
            for(size_t i=0;i<ids.size();++i) {
                auto& m=moves[ids[i]];
                m.path=replacements[i];m.u=swap(m.u);m.v=swap(m.v);
            }
            std::swap(incident[key(a)],incident[key(b)]);
        } else {
            for(size_t i=0;i<replacements.size();++i)change(moves[ids[i]].round,replacements[i],-1);
            for(int j:ids){auto& m=moves[j];change(m.round,m.path,1);}
        }
    }
    std::pair<double,std::vector<Transfer>> run(double baseline,int trials) {
        double best=baseline;std::vector<Transfer> saved;
        for(int pass=0;pass<2;++pass) {
            refresh();
            for(int i=0;i<trials;++i) {
                auto& m=moves[random()%moves.size()];
                int a=(i%3)?m.v:m.u;
                int width=std::min(graph.ranks,1<<(2+random()%6));
                int b=a^(1+int(random()%(width-1)));
                if(b>=graph.ranks)continue;
                trial(m.slot,a,b);
                if(i%256==255||i+1==trials) {
                    double value=duration();
                    if(value<best-1e-7){best=value;saved=moves;}
                }
            }
        }
        return {best,std::move(saved)};
    }
};

PyObject* exchange(PyObject*,PyObject* args) {
    int n,nodes,rounds,slots,trials;double startup,sw,baseline;
    PyObject *edges,*rates,*paths,*moves;
    if(!PyArg_ParseTuple(args,"iiiiO!O!ddO!O!id",&n,&nodes,&rounds,&slots,&PyList_Type,&edges,&PyList_Type,&rates,&startup,&sw,&PyList_Type,&paths,&PyList_Type,&moves,&trials,&baseline))return nullptr;
    try {
        if(n<2||n>256||nodes<n||nodes>1024||rounds<1||rounds>64||slots<1||slots>256||
           trials<1||trials>30000||PyList_Size(moves)<1||PyList_Size(moves)>270000||
           PyList_Size(edges)<1||PyList_Size(rates)<PyList_Size(edges)||PyList_Size(edges)>24000||PyList_Size(rates)>48000||int64_t(nodes)*n*rounds>5000000)
            throw std::invalid_argument("relay exchange bounds");
        Network g{n,nodes,std::vector<std::vector<int>>(nodes),std::vector<std::vector<int>>(nodes),{},{},startup,sw};
        for(int k=0;k<PyList_Size(rates);++k) {
            double rate=PyFloat_AsDouble(PyList_GET_ITEM(rates,k));
            if(!(rate>0)||!std::isfinite(rate))throw std::invalid_argument("invalid bandwidth");
            g.rate.push_back(rate);
        }
        for(int i=0;i<PyList_Size(edges);++i) {
            int u,v;PyObject* refs;
            if(!PyArg_ParseTuple(PyList_GET_ITEM(edges,i),"iiO!",&u,&v,&PyTuple_Type,&refs))return nullptr;
            if(u<0||v<0||u>=nodes||v>=nodes||!PyTuple_Size(refs))throw std::invalid_argument("bad edge");
            Edge e{u,v,{}};
            for(int j=0;j<PyTuple_Size(refs);++j) {
                long k=PyLong_AsLong(PyTuple_GET_ITEM(refs,j));
                if(k<0||size_t(k)>=g.rate.size())throw std::invalid_argument("bad resource");
                e.ids.push_back(k);
            }
            g.edges.push_back(std::move(e));g.incoming[v].push_back(i);g.outgoing[u].push_back(i);
        }
        Exchange solver(g,rounds,slots);
        for(int i=0;i<PyList_Size(paths);++i) {
            auto* row=PyList_GET_ITEM(paths,i);
            if(!PyTuple_Check(row)||!PyTuple_Size(row))throw std::invalid_argument("invalid path");
            std::vector<int> p;
            for(int j=0;j<PyTuple_Size(row);++j){
                long k=PyLong_AsLong(PyTuple_GET_ITEM(row,j));if(k<0||size_t(k)>=g.edges.size())throw std::invalid_argument("invalid path edge");
                p.push_back(k);
            }
            if(solver.paths.intern(p)!=i)throw std::invalid_argument("duplicate path");
        }
        for(int i=0;i<PyList_Size(moves);++i){
            int r,s,p;
            if(!PyArg_ParseTuple(PyList_GET_ITEM(moves,i),"iii",&r,&s,&p))return nullptr;
            if(r<0||r>=rounds||s<0||s>=slots||p<0||size_t(p)>=solver.paths.routes.size())throw std::invalid_argument("bad flow");
            auto& route=solver.paths.routes[p];int u=route.src,v=g.edges[route.path.back()].v;
            if(u>=n||v>=n)throw std::invalid_argument("invalid endpoint");
            solver.moves.push_back({r,s,u,v,p});
            solver.incident[s*n+u].push_back(i);solver.incident[s*n+v].push_back(i);
            auto& options=solver.existing[r][u*n+v];
            if(std::find(options.begin(),options.end(),p)==options.end())options.push_back(p);
            solver.change(r,p,1);
        }
        std::pair<double,std::vector<Transfer>> best;
        {AllowThreads release;best=solver.run(baseline,trials);}
        if(best.first>=.992*baseline)Py_RETURN_NONE;
        std::vector<int> used, mapping(solver.paths.routes.size(),-1);
        for(const auto& transfer:best.second) {
            if(mapping[transfer.path]<0) {
                mapping[transfer.path]=used.size();
                used.push_back(transfer.path);
            }
        }
        Owned routes(PyList_New(used.size())),selected(PyList_New(best.second.size()));
        if(!routes.p||!selected.p)return nullptr;
        for(size_t j=0;j<used.size();++j) {
            const auto& p=solver.paths.routes[used[j]].path;
            Owned row(PyTuple_New(p.size()));if(!row.p)return nullptr;
            for(size_t k=0;k<p.size();++k) {
                auto* v=PyLong_FromLong(p[k]);if(!v)return nullptr;PyTuple_SET_ITEM(row.p,k,v);
            }
            PyList_SET_ITEM(routes.p,j,row.release());
        }
        for(size_t i=0;i<best.second.size();++i) {
            auto* id=PyLong_FromLong(mapping[best.second[i].path]);if(!id)return nullptr;
            PyList_SET_ITEM(selected.p,i,id);
        }
        return Py_BuildValue("dNN",best.first,routes.release(),selected.release());
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
     catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
}
