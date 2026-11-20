// Path generation on actual directed edges followed by integer changes of flow.
// A group's bit-compatible senders all own its payload at round start.
namespace flow_routes {
using Term=std::pair<int,int>;
struct Edge {int u,v;std::vector<int> ids;};
struct Network {
    int ranks, nodes;
    std::vector<std::vector<int>> incoming, outgoing;
    std::vector<Edge> edges;
    std::vector<double> rate;
    double startup, switching;
};
struct Route {
    int src;
    std::vector<int> path;
    std::vector<Term> terms;
    double overhead, serial;
};
struct Group {
    int dst;
    int eligible;
    std::vector<std::pair<int,int>> parts;
};
using Portions=std::vector<std::pair<int,int>>;
struct Optimizer {
    const Network& g;
    int focus = -1;
    std::vector<Route> routes;
    std::map<std::vector<int>,int> indices;
    std::vector<Group> groups;
    std::vector<std::vector<int>> eligible;
    std::vector<double> load,dist,weights;
    std::vector<int> pred;
    std::vector<std::vector<int>> by_dest;
    Optimizer(const Network& net):g(net),load(net.rate.size()),dist(net.nodes),
        weights(net.edges.size()),pred(net.nodes),by_dest(2*net.ranks) {}
    int intern(const std::vector<int>& path) {
        auto found=indices.find(path);
        if(found!=indices.end()) return found->second;
        if(routes.size()>=180000) throw std::invalid_argument("adaptive route budget");
        std::map<int,int> refs;
        double serial=0;
        for(int e:path) {
            for(int k:g.edges[e].ids) ++refs[k];
            serial+=g.rate[e];
        }
        int id=routes.size();
        routes.push_back({g.edges[path.front()].u,path,{refs.begin(),refs.end()},
                          g.startup+(path.size()-1)*g.switching,serial});
        indices.emplace(path,id);return id;
    }
    void add(int route,int count) {
        for(auto [k,n]:routes[route].terms) load[k]+=count*n*g.rate[k];
    }
    static double power(double x) {double v=x*x; return v*v*v;}
    double marginal(int id,int step,double inverse) const {
        double cost=0;
        for(auto [k,n]:routes[id].terms) {
            double delta=n*g.rate[k];
            double x=std::max(0.,load[k]+.5*step*delta)*inverse;
            cost+=delta*(.00002+power(x))*(k==focus ? 2.5 : 1.0);
        }
        return cost;
    }
    double duration() const {
        double overhead=0,serial=0;
        for(auto& f:groups) for(auto [id,n]:f.parts) if(n) {
            const auto& r=routes[id];
            overhead=std::max(overhead,r.overhead);serial=std::max(serial,r.overhead+r.serial);
        }
        return std::max(serial,overhead+*std::max_element(load.begin(),load.end()));
    }
    void shortest(int dest,bool forward=false) {
        std::fill(dist.begin(),dist.end(),std::numeric_limits<double>::infinity());
        using Item=std::pair<double,int>;
        std::priority_queue<Item,std::vector<Item>,std::greater<Item>> heap;
        dist[dest]=0;heap.emplace(0,dest);
        while(!heap.empty()) {
            auto [d,v]=heap.top();heap.pop();
            if(d!=dist[v]) continue;
            const auto& incident=forward ? g.outgoing[v] : g.incoming[v];
            for(int e:incident) {
                int u=forward ? g.edges[e].v : g.edges[e].u; double nd=d+weights[e];
                if(nd<dist[u]) {
                    dist[u]=nd;pred[u]=e;
                    if(u>=g.ranks) heap.emplace(nd,u);
                }
            }
        }
    }
    std::vector<Portions> run(int passes,double& best) {
        best=std::min(best,duration());
        std::vector<Portions> saved;
        auto save=[&](){saved.clear();for(auto& f:groups)saved.push_back(f.parts);};
        save();
        for(int pass=0;pass<passes;++pass) {
            double peak=*std::max_element(load.begin(),load.end());
            if(peak<=0) break;
            double inverse=1/peak;
            for(size_t e=0;e<weights.size();++e) {
                double w=0; for(int k:g.edges[e].ids) w+=g.rate[k]*(.00002+power(load[k]*inverse))*(k==focus ? 2.5 : 1.0);
                // Prices alone can prefer an idle, very slow path that cannot
                // carry even one atom under the incumbent ceiling.
                constexpr double latency_scales[] = {.025, .003, .16, .0001};
                weights[e]=w+latency_scales[pass%4]*g.rate[e];
            }
            int changed=0;
            for(int step=0;step<2*g.ranks;++step) {
                bool forward=step>=g.ranks;
                int dest=(step* (g.ranks%2==0 ? g.ranks-1:1) + 17*pass)%g.ranks;
                auto& jobs=by_dest[dest+(forward?g.ranks:0)];
                if(jobs.empty()) continue;
                shortest(dest,forward);
                std::vector<int> lookup(g.ranks,-1);
                std::vector<int> nearest(eligible.size(),-1);
                for(int index:jobs) {
                    auto& group=groups[index];
                    int& src=nearest[group.eligible];
                    if(src<0) {
                        const auto& senders=eligible[group.eligible];
                        src=*std::min_element(senders.begin(),senders.end(),
                            [&](int a,int b){return dist[a]<dist[b];});
                    }
                    if(src==dest || !std::isfinite(dist[src])) continue;
                    int alt=lookup[src];
                    if(alt<0) {
                        std::vector<int> path;int u=src;
                        while(u!=dest) {
                            int e=pred[u];path.push_back(e);u=forward?g.edges[e].u:g.edges[e].v;
                            if(path.size()>64) break;
                        }
                        if(u!=dest) continue;
                        if(forward)std::reverse(path.begin(),path.end());
                        alt=lookup[src]=intern(path);
                    }
                    if(routes[alt].overhead+routes[alt].serial>=best) continue;
                    int moved=0;
                    for(auto& old:group.parts) {
                        if(old.first==alt || !old.second) continue;
                        // A shared term cancels when changing path. Remove the
                        // incumbent unit before evaluating the two directions.
                        while(old.second>0) {
                            add(old.first,-1);
                            double a=marginal(alt,1,inverse), b=marginal(old.first,1,inverse);
                            if(a>=b*(1.-1e-7)) {add(old.first,1);break;}
                            --old.second;++moved;add(alt,1);
                        }
                    }
                    if(moved) {
                        auto pos=std::find_if(group.parts.begin(),group.parts.end(),[&](auto v){return v.first==alt;});
                        if(pos==group.parts.end()) group.parts.emplace_back(alt,moved);
                        else pos->second+=moved;
                        changed+=moved;
                    }
                }
            }
            double cost=duration();
            if(cost+1e-8<best) {best=cost;save();}
            if(!changed && pass>=3) break;
        }
        return saved;
    }
};

PyObject* run(PyObject*,PyObject* args) {
    int ranks,nodes,passes,focus=-1;double startup,sw,base;
    PyObject *edges,*rates,*paths,*groups;
    if(!PyArg_ParseTuple(args,"iiO!O!ddO!O!id|i",&ranks,&nodes,&PyList_Type,&edges,&PyList_Type,&rates,
        &startup,&sw,&PyList_Type,&paths,&PyList_Type,&groups,&passes,&base,&focus))return nullptr;
    try {
        if(ranks<2 || ranks>512 || nodes<ranks || nodes>2048 || PyList_Size(edges)>24000 ||
           PyList_Size(rates)<PyList_Size(edges) || PyList_Size(edges)<1 || PyList_Size(rates)>48000 || PyList_Size(groups)>300000 || passes<1 || passes>32)
            throw std::invalid_argument("adaptive routing dimensions");
        Network g{ranks,nodes,std::vector<std::vector<int>>(nodes),std::vector<std::vector<int>>(nodes),{},{},startup,sw};
        for(int i=0;i<PyList_Size(rates);++i) {
            double x=PyFloat_AsDouble(PyList_GET_ITEM(rates,i));
            if(!(x>0) || !std::isfinite(x)) throw std::invalid_argument("invalid bandwidth");
            g.rate.push_back(x);
        }
        for(int i=0;i<PyList_Size(edges);++i) {
            int u,v;PyObject* refs;
            if(!PyArg_ParseTuple(PyList_GET_ITEM(edges,i),"iiO!",&u,&v,&PyTuple_Type,&refs))return nullptr;
            if(u<0||u>=nodes||v<0||v>=nodes||!PyTuple_Size(refs))throw std::invalid_argument("bad edge");
            Edge edge{u,v,{}};
            for(int j=0;j<PyTuple_Size(refs);++j) {
                long k=PyLong_AsLong(PyTuple_GET_ITEM(refs,j));
                if(k<0||size_t(k)>=g.rate.size()) throw std::invalid_argument("bad pool");
                edge.ids.push_back(k);
            }
            g.incoming[v].push_back(i);g.outgoing[u].push_back(i);g.edges.push_back(std::move(edge));
        }
        if(focus < -1 || focus >= int(g.rate.size())) throw std::invalid_argument("invalid focus resource");
        Optimizer solver(g);
        solver.focus=focus;
        for(int i=0;i<PyList_Size(paths);++i) {
            auto* row=PyList_GET_ITEM(paths,i);if(!PyTuple_Check(row)||!PyTuple_Size(row))throw std::invalid_argument("bad path");
            std::vector<int> path;
            for(int j=0;j<PyTuple_Size(row);++j) {
                long e=PyLong_AsLong(PyTuple_GET_ITEM(row,j)); if(e<0||size_t(e)>=g.edges.size())throw std::invalid_argument("bad edge index");
                path.push_back(e);
            }
            if(solver.intern(path)!=i)throw std::invalid_argument("repeated path");
        }
        std::unordered_map<PyObject*,int> masks;
        for(int i=0;i<PyList_Size(groups);++i) {
            int old,count,forward=0;
            PyObject* senders;
            if(!PyArg_ParseTuple(PyList_GET_ITEM(groups,i),"iiO!p",&old,&count,&PyTuple_Type,
                                 &senders,&forward)) return nullptr;
            if(old<0 || old>=int(solver.routes.size()) || count<=0 || !PyTuple_Size(senders))
                throw std::invalid_argument("bad flow");
            const auto& route=solver.routes[old];
            int dest=forward ? route.src : g.edges[route.path.back()].v;
            if(dest<0 || dest>=ranks) throw std::invalid_argument("bad destination");
            auto [it,fresh]=masks.emplace(senders,solver.eligible.size());
            Group f{dest,it->second,{{old,count}}};
            if(fresh) {
                solver.eligible.emplace_back();
                for(int j=0;j<PyTuple_Size(senders);++j) {
                    long u=PyLong_AsLong(PyTuple_GET_ITEM(senders,j));
                    if(u<0 || u>=ranks) throw std::invalid_argument("bad sender");
                    solver.eligible.back().push_back(u);
                }
            }
            solver.groups.push_back(std::move(f));
            solver.by_dest[dest+(forward?ranks:0)].push_back(i);
            solver.add(old,count);
        }
        std::vector<Portions> selected;
        double result=base;
        {AllowThreads release;selected=solver.run(passes,result);}
        if(result>=base*.993) Py_RETURN_NONE;
        Owned out(PyList_New(selected.size()));
        if(!out.p)return nullptr;
        for(size_t i=0;i<selected.size();++i) {
            Owned parts(PyList_New(0));
            if(!parts.p)return nullptr;
            for(auto [id,count]:selected[i]) if(count) {
                const auto& path=solver.routes[id].path; Owned ids(PyTuple_New(path.size()));
                if(!ids.p)return nullptr;
                for(size_t j=0;j<path.size();++j) {
                    auto* k=PyLong_FromLong(path[j]);if(!k)return nullptr;PyTuple_SET_ITEM(ids.p,j,k);
                }
                Owned item(Py_BuildValue("(iO)",count,ids.p));
                if(!item.p||PyList_Append(parts.p,item.p)<0)return nullptr;
            }
            PyList_SET_ITEM(out.p,i,parts.release());
        }
        return Py_BuildValue("(dN)",result,out.release());
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
     catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
}
