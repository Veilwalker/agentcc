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
    int single;
    std::vector<std::pair<int,int>> parts;
};
using Portions=std::vector<std::pair<int,int>>;
struct Optimizer {
    const Network& g;
    int focus = -1;
    std::vector<Route> routes;
    // At most 24,000 edges are admitted. Four 15-bit edge IDs and a
    // leading 1 fit in 61 bits and encode both the path and its length.
    // Both indices are ordered maps, including the fallback for long routes.
    std::map<uint64_t,int> short_indices;
    std::map<std::vector<int>,int> indices;
    std::vector<Group> groups;
    std::vector<std::vector<int>> eligible;
    std::vector<double> load,dist,weights;
    std::vector<int> pred;
    std::vector<std::vector<int>> by_dest;
    Optimizer(const Network& net):g(net),load(net.rate.size()),dist(net.nodes),
        weights(net.edges.size()),pred(net.nodes),by_dest(2*net.ranks) {}
    int intern(const std::vector<int>& path) {
        uint64_t key=0;
        if(path.size()<=4) {
            key=1;
            for(int e:path)key=(key<<15)|static_cast<uint64_t>(e);
            auto found=short_indices.find(key);
            if(found!=short_indices.end()) return found->second;
        } else {
            auto found=indices.find(path);
            if(found!=indices.end()) return found->second;
        }
        if(routes.size()>=180000) throw std::invalid_argument("adaptive route budget");
        std::array<int,32> refs;
        size_t count=0;
        double serial=0;
        for(int e:path) {
            const auto& ids=g.edges[e].ids;
            if(count<=refs.size()){
                if(ids.size()<=refs.size()-count)std::copy(ids.begin(),ids.end(),refs.begin()+count);
                count+=std::min(ids.size(),refs.size()+1-count);
            }
            serial+=g.rate[e];
        }
        std::vector<Term> incidence;
        if(count<=refs.size()){
            std::sort(refs.begin(),refs.begin()+count);
            incidence.reserve(count);
            for(size_t i=0;i<count;++i){
                const int k=refs[i];
                if(!incidence.empty() && incidence.back().first==k) ++incidence.back().second;
                else incidence.emplace_back(k,1);
            }
        } else {
            // Keep memory proportional to distinct resources on long or highly shared paths.
            std::map<int,int> multiple;
            for(int e:path)for(int k:g.edges[e].ids)++multiple[k];
            incidence.assign(multiple.begin(),multiple.end());
        }
        int id=routes.size();
        routes.push_back({g.edges[path.front()].u,path,std::move(incidence),
                          g.startup+(path.size()-1)*g.switching,serial});
        if(key)short_indices.emplace(key,id);
        else indices.emplace(path,id);
        return id;
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
        auto visit = [&](int id) {
            const auto& r=routes[id];
            overhead=std::max(overhead,r.overhead);serial=std::max(serial,r.overhead+r.serial);
        };
        for (const auto& group : groups) {
            if (group.single >= 0) visit(group.single);
            else for (auto [id,count] : group.parts) if (count) visit(id);
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
        std::vector<Portions> saved(groups.size());
        auto save = [&]() {
            for (size_t i=0; i<groups.size(); ++i) {
                if (groups[i].single >= 0) saved[i].assign(1, {groups[i].single,1});
                else saved[i]=groups[i].parts;
            }
        };
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
                    // A single atom always has exactly one active route. Its
                    // pressure test and load updates match the general case.
                    if (group.single >= 0) {
                        const int previous = group.single;
                        if (previous == alt) continue;
                        add(previous,-1);
                        const double a=marginal(alt,1,inverse), b=marginal(previous,1,inverse);
                        if (!(a >= b*(1.-1e-7))) { group.single=alt; ++changed; }
                        add(group.single,1);
                        continue;
                    }
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
    PyObject *edges,*rates,*paths,*groups,*holders=Py_None,*emission=Py_None;
    if(!PyArg_ParseTuple(args,"iiO!O!ddO!O!id|iOO",&ranks,&nodes,&PyList_Type,&edges,&PyList_Type,&rates,
        &startup,&sw,&PyList_Type,&paths,&PyList_Type,&groups,&passes,&base,&focus,&holders,&emission))return nullptr;
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
        auto add_group = [&](int old, int count, PyObject* senders, bool forward) {
            if (old<0 || old>=int(solver.routes.size()) || count<1 ||
                !PyTuple_Check(senders) || !PyTuple_GET_SIZE(senders))
                throw std::invalid_argument("bad flow");
            const auto& route=solver.routes[old];
            int dest=forward ? route.src : g.edges[route.path.back()].v;
            if(dest<0 || dest>=ranks) throw std::invalid_argument("bad destination");
            auto [it,fresh]=masks.emplace(senders,solver.eligible.size());
            if(fresh) {
                solver.eligible.emplace_back();
                for(int j=0;j<PyTuple_GET_SIZE(senders);++j) {
                    long u=PyLong_AsLong(PyTuple_GET_ITEM(senders,j));
                    if(u<0 || u>=ranks) throw std::invalid_argument("bad sender");
                    solver.eligible.back().push_back(u);
                }
            }
            Group group{dest,it->second,count==1 ? old : -1,{}};
            if (count>1) group.parts.emplace_back(old,count);
            solver.by_dest[dest+(forward?ranks:0)].push_back(solver.groups.size());
            solver.groups.push_back(std::move(group));
            solver.add(old,count);
        };
        // The gather producer can pass original tokens grouped by transfer.
        // Preserve first-seen flow order, exactly as a Python ordered dict.
        std::vector<std::vector<PyObject*>> payloads;
        if (holders != Py_None) {
            if (!PyDict_Check(holders)) throw std::invalid_argument("gather owners must be a mapping");
            std::vector<std::unordered_map<PyObject*,int>> by_path(solver.routes.size());
            std::vector<std::pair<int,PyObject*>> seeds;
            size_t atoms=0;
            for(int i=0;i<PyList_GET_SIZE(groups);++i) {
                int path;
                PyObject* tokens;
                if(!PyArg_ParseTuple(PyList_GET_ITEM(groups,i),"iO!",&path,&PyList_Type,&tokens)) return nullptr;
                if (path<0 || size_t(path)>=by_path.size()) throw std::invalid_argument("bad gather path");
                atoms+=PyList_GET_SIZE(tokens);
                if (atoms>3000000) throw std::invalid_argument("gather atom budget");
                for (int j=0;j<PyList_GET_SIZE(tokens);++j) {
                    PyObject* token=PyList_GET_ITEM(tokens,j);
                    PyObject* senders=PyDict_GetItemWithError(holders,token);
                    if (!senders) {
                        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,"missing gather owners");
                        return nullptr;
                    }
                    auto [it,fresh]=by_path[path].emplace(senders,payloads.size());
                    if (fresh) {
                        if (payloads.size()>=300000) throw std::invalid_argument("adaptive flow budget");
                        payloads.emplace_back();
                        seeds.emplace_back(path,senders);
                    }
                    payloads[it->second].push_back(token);
                }
            }
            for (size_t i=0; i<seeds.size(); ++i)
                add_group(seeds[i].first,payloads[i].size(),seeds[i].second,false);
        } else {
            for(int i=0;i<PyList_GET_SIZE(groups);++i) {
                int old,count,forward=0;
                PyObject* senders;
                if(!PyArg_ParseTuple(PyList_GET_ITEM(groups,i),"iiO!p",&old,&count,&PyTuple_Type,
                                     &senders,&forward)) return nullptr;
                add_group(old,count,senders,forward);
            }
        }
        std::vector<Portions> selected;
        double result=base;
        {AllowThreads release;selected=solver.run(passes,result);}
        if(result>=base*.993) Py_RETURN_NONE;
        if (holders != Py_None) {
            std::vector<int> used;
            std::vector<std::vector<PyObject*>> batches(solver.routes.size());
            for (size_t i=0;i<selected.size();++i) {
                size_t offset=0;
                for (auto [path,atoms] : selected[i]) if (atoms>0) {
                    if (offset+atoms>payloads[i].size()) throw std::invalid_argument("gather flow over-allocation");
                    auto& batch=batches[path];
                    if (batch.empty()) used.push_back(path);
                    auto first=payloads[i].begin()+offset;
                    batch.insert(batch.end(),first,first+atoms);
                    offset+=atoms;
                }
                if (offset!=payloads[i].size()) throw std::invalid_argument("gather flow under-allocation");
            }
            if (emission!=Py_None) {
                PyObject *labels,*inverse;
                double bytes;
                if (!PyArg_ParseTuple(emission,"O!dO!",&PyList_Type,&labels,&bytes,&PyList_Type,&inverse)) return nullptr;
                if (PyList_GET_SIZE(labels)!=nodes || PyList_GET_SIZE(inverse)!=PyList_GET_SIZE(edges) ||
                    !std::isfinite(bytes) || bytes<=0)
                    throw std::invalid_argument("invalid gather packet materialization inputs");
                Owned out(PyList_New(0)), action(PyUnicode_InternFromString("copy"));
                Owned source_key(PyUnicode_InternFromString("src")), destination_key(PyUnicode_InternFromString("dst"));
                Owned action_key(PyUnicode_InternFromString("action")), path_key(PyUnicode_InternFromString("path"));
                Owned range_key(PyUnicode_InternFromString("chunk_ranges"));
                if(!out.p||!action.p||!source_key.p||!destination_key.p||!action_key.p||!path_key.p||!range_key.p) return nullptr;
                for(int id:used) {
                    const auto& route=solver.routes[id];
                    const auto& tokens=batches[id];
                    double inverse_sum=0.;
                    for(int e:route.path){
                        const double coefficient=PyFloat_AsDouble(PyList_GET_ITEM(inverse,e));
                        if(PyErr_Occurred())return nullptr;
                        if(!(coefficient>0.) || !std::isfinite(coefficient))
                            throw std::invalid_argument("invalid gather serialization rate");
                        inverse_sum+=coefficient;
                    }
                    const size_t width=static_cast<size_t>(std::clamp(
                        std::floor((result-route.overhead)/(bytes*inverse_sum)+1e-10),1.,double(tokens.size())));
                    Owned physical(PyList_New(route.path.size()+1));
                    if(!physical.p)return nullptr;
                    auto* src=PyList_GET_ITEM(labels,route.src);
                    Py_INCREF(src);PyList_SET_ITEM(physical.p,0,src);
                    for(size_t j=0;j<route.path.size();++j) {
                        auto* node=PyList_GET_ITEM(labels,g.edges[route.path[j]].v);
                        Py_INCREF(node);PyList_SET_ITEM(physical.p,j+1,node);
                    }
                    auto* dst=PyList_GET_ITEM(physical.p,route.path.size());
                    for(size_t at=0;at<tokens.size();at+=width) {
                        const size_t take=std::min(width,tokens.size()-at);
                        Owned payload(PyList_New(take)), move(PyDict_New());
                        if(!move.p || !payload.p) return nullptr;
                        for(size_t j=0;j<take;++j){auto* token=tokens[at+j];Py_INCREF(token);PyList_SET_ITEM(payload.p,j,token);}
                        // Own the path list per transfer, as the existing Python materializer does.
                        Owned distinct(at ? PyList_GetSlice(physical.p,0,PyList_GET_SIZE(physical.p)) : Py_NewRef(physical.p));
                        if(!distinct.p)return nullptr;
                        if(PyDict_SetItem(move.p,source_key.p,src)<0 || PyDict_SetItem(move.p,destination_key.p,dst)<0 ||
                           PyDict_SetItem(move.p,path_key.p,distinct.p)<0 || PyDict_SetItem(move.p,action_key.p,action.p)<0 ||
                           PyDict_SetItem(move.p,range_key.p,payload.p)<0 || PyList_Append(out.p,move.p)<0) return nullptr;
                    }
                }
                return Py_BuildValue("(dN)",result,out.release());
            }
            Owned out(PyList_New(used.size()));
            if (!out.p) return nullptr;
            for (size_t i=0; i<used.size(); ++i) {
                int id=used[i];
                const auto& path=solver.routes[id].path;
                const auto& tokens=batches[id];
                Owned ids(PyTuple_New(path.size())), payload(PyList_New(tokens.size()));
                if (!ids.p || !payload.p) return nullptr;
                for (size_t j=0;j<path.size();++j) {
                    auto* e=PyLong_FromLong(path[j]);
                    if (!e) return nullptr;
                    PyTuple_SET_ITEM(ids.p,j,e);
                }
                for (size_t j=0;j<tokens.size();++j) {
                    Py_INCREF(tokens[j]);
                    PyList_SET_ITEM(payload.p,j,tokens[j]);
                }
                auto* entry=PyTuple_Pack(2,ids.p,payload.p);
                if (!entry) return nullptr;
                PyList_SET_ITEM(out.p,i,entry);
            }
            return Py_BuildValue("(dN)",result,out.release());
        }
        Owned out(PyList_New(selected.size())), path_ids(PyList_New(solver.routes.size()));
        if (!out.p || !path_ids.p) return nullptr;
        // Allocations often share a physical path. Construct its edge-index
        // tuple once; every result still exposes exactly the same sequence.

        for(size_t i=0;i<selected.size();++i) {
            Owned parts(PyList_New(0));
            if(!parts.p)return nullptr;
            for(auto [id,count]:selected[i]) if(count) {
                if (!PyList_GET_ITEM(path_ids.p, id)) {
                    const auto& path = solver.routes[id].path;
                    Owned ids(PyTuple_New(path.size()));
                    if (!ids.p) return nullptr;
                    for (size_t j = 0; j < path.size(); ++j) {
                        auto* edge = PyLong_FromLong(path[j]);
                        if (!edge) return nullptr;
                        PyTuple_SET_ITEM(ids.p, j, edge);
                    }
                    PyList_SET_ITEM(path_ids.p, id, ids.release());
                }
                Owned item(Py_BuildValue("(iO)",count,PyList_GET_ITEM(path_ids.p, id)));
                if(!item.p||PyList_Append(parts.p,item.p)<0)return nullptr;
            }
            PyList_SET_ITEM(out.p,i,parts.release());
        }
        return Py_BuildValue("(dN)",result,out.release());
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
     catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
}
