// Rebuild local gather deliveries near a fixed inter-domain cut.
namespace local_tail {
struct Route : RouteOption { int src, dst; };
struct Move { int time, path, atom; };
struct Price {
    std::vector<double> loads;
    double peak=0., latency=0., serial=0.;
    explicit Price(int resources): loads(resources) {}
    double duration() const { return std::max(peak+latency,serial); }
    void add(const Route& route) {
        for (auto [k,w]:route.terms) peak=std::max(peak,loads[k]+=w);
        latency=std::max(latency,route.overhead);
        serial=std::max(serial,route.overhead+route.serial);
    }
    std::pair<double,double> price(const Route& route) const {
        double work=0., energy=0.;
        for (auto [k,w]:route.terms) {
            work=std::max(work,loads[k]+w);
            energy+=(2*loads[k]+w)*w;
        }
        return {std::max(route.overhead+route.serial, std::max(latency,route.overhead)+work),energy};
    }
};

PyObject* run(PyObject*, PyObject* args) {
    int resources, first, stages, local_count, tapered=0;
    double bytes;
    PyObject *domains, *missing, *specs, *fixed;
    if (!PyArg_ParseTuple(args,"idiiO!O!O!iO!|p", &resources,&bytes,&first,&stages,
                          &PyList_Type,&domains,&PyBytes_Type,&missing,&PyList_Type,&specs,
                          &local_count,&PyList_Type,&fixed,&tapered)) return nullptr;
    const int n=PyList_GET_SIZE(domains), path_count=PyList_GET_SIZE(specs);
    const int never=127, depth=stages-2;
    auto invalid=[](const char* message)->PyObject* { PyErr_SetString(PyExc_ValueError,message); return nullptr; };
    if (n<2 || n>256 || PyBytes_GET_SIZE(missing)!=n*n || stages<3 || stages>(tapered?16:6) || first<0 || first>64 ||
        resources<1 || resources>50000 || !std::isfinite(bytes) || bytes<=0 || local_count<1 ||
        path_count<local_count || path_count>30000 || PyList_GET_SIZE(fixed)>n*n)
        return invalid("unsupported local gather dimensions");
    try {
        std::vector<std::vector<int>> members(n), incoming(n);
        for(int r=0;r<n;++r) {
            long d=PyLong_AsLong(PyList_GET_ITEM(domains,r));
            if(PyErr_Occurred() || d<0 || d>=n) return invalid("invalid local domain");
            members[d].push_back(r);
            if(members[d].size()>64) return invalid("oversized local domain");
        }
        std::vector<Route> routes(path_count);
        for(int i=0;i<path_count;++i) {
            Route& r=routes[i]; PyObject *cost,*terms,*path;
            if(!PyArg_ParseTuple(PyList_GET_ITEM(specs,i),"iiO!",&r.src,&r.dst,&PyTuple_Type,&cost) ||
               !PyArg_ParseTuple(cost,"O!ddO",&PyTuple_Type,&terms,&r.overhead,&r.serial,&path)) return nullptr;
            if(r.src<0 || r.src>=n || r.dst<0 || r.dst>=n || r.src==r.dst || !std::isfinite(r.overhead) ||
               r.overhead<0 || !std::isfinite(r.serial) || r.serial<=0 || PyTuple_GET_SIZE(terms)>128)
                return invalid("invalid local gather route");
            r.serial*=bytes;
            for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(terms);++j) {
                Py_ssize_t k; double w;
                if(!read_term(terms,j,resources,k,w)) return nullptr;
                r.terms.emplace_back(k,w*bytes);
            }
            if(i<local_count) incoming[r.dst].push_back(i);
        }
        std::vector<int> initial(n*n);
        const char* mask=PyBytes_AS_STRING(missing);
        for(int i=0;i<n*n;++i)initial[i]=mask[i]?never:0;
        std::vector<Price> anchor(stages,Price(resources));
        std::vector<Move> baseline;
        for(Py_ssize_t i=0;i<PyList_GET_SIZE(fixed);++i) {
            Move m;
            if(!PyArg_ParseTuple(PyList_GET_ITEM(fixed,i),"iii",&m.time,&m.path,&m.atom)) return nullptr;
            if(m.time<0 || m.time>=(tapered?64:depth) || m.path<local_count || m.path>=path_count || m.atom<0 || m.atom>=n)
                return invalid("invalid inter-domain gather anchor");
            baseline.push_back(m);
        }
        // Rounding a fractional allocation of each fixed route preserves its
        // physical load budget while moving long-path atoms into larger waves.
        if(tapered) {
            std::vector<double> total(resources), weights(depth,1.);
            for(auto m:baseline) for(auto [k,w]:routes[m.path].terms)total[k]+=w;
            double bandwidth_floor=.45 * *std::max_element(total.begin(),total.end());
            if(!(bandwidth_floor>0.))Py_RETURN_NONE;
            for(int t=1;t<depth;++t) weights[t]=weights[t-1]*.70;
            double norm=0.;for(double w:weights)norm+=w;for(double& w:weights)w/=norm;
            std::vector<std::vector<double>> work(depth,std::vector<double>(resources));
            std::vector<int> assigned(depth);
            std::vector<double> pressure(path_count);
            for (int i=local_count;i<path_count;++i) {
                for(auto [k,w]:routes[i].terms)pressure[i]+=total[k]*w;
            }
            std::stable_sort(baseline.begin(),baseline.end(),[&](Move a, Move b){
                const auto& ra=routes[a.path];const auto& rb=routes[b.path];
                return std::pair<double,double>{ra.serial,pressure[a.path]}>
                       std::pair<double,double>{rb.serial,pressure[b.path]};
            });
            for(auto& m:baseline) {
                const Route& arc=routes[m.path];
                auto lowest=std::make_tuple(std::numeric_limits<double>::infinity(),0.,0.);
                for(int t=0;t<depth;++t) {
                    double fraction=0.,energy=0.;
                    for(auto [k,w]:arc.terms) {
                        double relative=(work[t][k]+.5*w)/(weights[t]*std::max(total[k],bandwidth_floor));
                        fraction=std::max(fraction,relative);energy+=w*relative;
                    }
                    double excess=std::max(0.,(arc.overhead+arc.serial)*.45/(weights[t]*bandwidth_floor)-1.);
                    auto price=std::make_tuple(fraction+4.*excess,energy,(assigned[t]+.5)/weights[t]);
                    if(price<lowest){lowest=price;m.time=t;}
                }
                ++assigned[m.time];
                for(auto [k,w]:arc.terms)work[m.time][k]+=w;
            }
        }
        for(auto m:baseline) {
            const Route& r=routes[m.path];
            if(initial[m.atom*n+r.dst]!=never || initial[m.atom*n+r.src]!=0) Py_RETURN_NONE;
            initial[m.atom*n+r.dst]=m.time+1;
            anchor[m.time].add(r);
        }
        int domain_count=n; while(domain_count>0 && members[domain_count-1].empty()) --domain_count;
        double previous_peak=0.; for(auto& p:anchor)previous_peak=std::max(previous_peak,p.duration());
        double best=std::numeric_limits<double>::infinity();
        std::vector<double> durations;
        std::vector<Move> selected;
        {
            AllowThreads release;
            for(double share:{.10,.15}) {
                if(tapered && share>.10) break;
                auto rows=anchor; auto ready=initial; auto moves=baseline;
                auto send=[&](int t,int owner,int pid) {
                    const Route& r=routes[pid];
                    rows[t].add(r);ready[owner*n+r.dst]=t+1;moves.push_back({t,pid,owner});
                };
                for(int t=0;t<stages;++t) {
                    auto& row=rows[t];
                    double ceiling=t<depth ? row.duration() : t==depth ? previous_peak*share : best;
                    if(tapered && t==depth) {
                        double minimum=std::numeric_limits<double>::infinity();
                        for(int i=0;i<local_count;++i) minimum=std::min(minimum,routes[i].overhead+routes[i].serial);
                        ceiling=std::min(ceiling,std::max(minimum*1.35,anchor[depth-1].duration()*.18));
                    }
                    if(t==depth) {
                        // Before the last round, reserve relays for any target unreachable from all current holders.
                        for(int owner=0;owner<n;++owner)for(int dst=0;dst<n;++dst) {
                            const int base=owner*n;
                            if(ready[base+dst]<=t+1)continue;
                            bool covered=false;
                            for(int p:incoming[dst])if(ready[base+routes[p].src]<=t+1){covered=true;break;}
                            if(covered)continue;
                            double lowest=std::numeric_limits<double>::infinity();int bridge=-1;
                            for(int p:incoming[dst])for(int q:incoming[routes[p].src]) {
                                const Route& arc=routes[q];
                                if(ready[base+arc.src]>t || ready[base+arc.dst]!=never)continue;
                                double score=row.price(arc).first;
                                if(score<lowest){lowest=score;bridge=q;}
                            }
                            if(bridge>=0)send(t,owner,bridge);
                        }
                    }
                    for(int j=0;j<64;++j)for(int k=0;k<n;++k) {
                        const int phase=first+t;
                        int owner=(k*(n%137?137:1)+phase*29)%n;
                        for(int d=0;d<domain_count;++d) {
                            const auto& group=members[d];
                            if(j>=int(group.size()))continue;
                            int dst=group[(owner*17+d*5+phase*7+j)%group.size()];
                            if(ready[owner*n+dst]!=never)continue;
                            std::pair<double,double> lowest{std::numeric_limits<double>::infinity(),0.};
                            int chosen=-1;
                            for(int p:incoming[dst]) {
                                const Route& arc=routes[p];
                                if(ready[owner*n+arc.src]>t)continue;
                                auto score=row.price(arc);
                                if(score<lowest){lowest=score;chosen=p;}
                            }
                            if(chosen>=0 && lowest.first<=ceiling+1e-7)send(t,owner,chosen);
                        }
                    }
                }
                if(std::find(ready.begin(),ready.end(),never)!=ready.end())continue;
                std::vector<double> times;double total=0.;
                for(auto& p:rows){times.push_back(p.duration());total+=p.duration();}
                if(total<best){best=total;selected=std::move(moves);durations=std::move(times);}
            }
        }
        if(durations.empty())Py_RETURN_NONE;
        Owned moves(PyList_New(selected.size())), times(PyList_New(durations.size()));
        if(!moves.p || !times.p)return nullptr;
        for(size_t i=0;i<selected.size();++i){auto m=selected[i];auto* item=Py_BuildValue("(iii)",m.time,m.path,m.atom);if(!item)return nullptr;PyList_SET_ITEM(moves.p,i,item);}
        for(size_t i=0;i<durations.size();++i){auto* item=PyFloat_FromDouble(durations[i]);if(!item)return nullptr;PyList_SET_ITEM(times.p,i,item);}
        return PyTuple_Pack(2,moves.p,times.p);
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
}
} // namespace local_tail
