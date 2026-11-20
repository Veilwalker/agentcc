// Move complete atoms into measured capacity headroom, using the original
// round-start ownership as a conservative earliest possible send time.
namespace round_slack {
struct Route {
    std::vector<std::pair<int, double>> terms;
    double delay, serial;
};
struct Packet {
    int round, route, src, dst;
    bool copy;
    std::vector<int> slots;
    size_t offset;
};
struct Row {
    std::vector<double> load;
    double delay = 0, serial = 0, peak = 0, ceiling = 0;
    int ready = 0, moved = 0, received = 0, alternate = 0, hot = 0;
};
static PyObject* run(PyObject*, PyObject* args) {
    int ranks, slots, resources, focus, prefer_relay=0;
    double size;
    PyObject *routes_arg, *packets_arg, *owners_arg, *times_arg;
    if (!PyArg_ParseTuple(args, "iiidiO!O!O!O!|p", &ranks, &slots, &resources, &size, &focus,
        &PyList_Type, &routes_arg, &PyList_Type, &packets_arg, &PyList_Type, &owners_arg,
        &PyList_Type, &times_arg, &prefer_relay)) return nullptr;
    const int nr = PyList_GET_SIZE(times_arg);
    if (ranks < 2 || slots < 1 || int64_t(ranks)*slots > 530000 || nr < 1 || nr > 128 ||
        resources < 1 || int64_t(resources)*nr > 3000000 || !(size>0) ||
        PyList_GET_SIZE(owners_arg)!=slots || focus < -1 || focus >= resources) {
        PyErr_SetString(PyExc_ValueError, "slack search dimensions");
        return nullptr;
    }
    try {
        std::vector<Route> routes;
        for (Py_ssize_t i=0; i<PyList_GET_SIZE(routes_arg); ++i) {
            PyObject *terms, *path;
            Route r;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(routes_arg,i),"O!ddO",&PyTuple_Type,&terms,
                                  &r.delay,&r.serial,&path)) return nullptr;
            r.serial = r.delay + size*r.serial;
            for (int j=0; j<PyTuple_GET_SIZE(terms); ++j) {
                Py_ssize_t index; double rate;
                if (!read_term(terms,j,resources,index,rate)) return nullptr;
                r.terms.emplace_back(index, rate*size);
            }
            routes.push_back(std::move(r));
        }
        std::vector<Row> rows(nr);
        for (int t=0; t<nr; ++t) {
            rows[t].load.assign(resources,0);
            rows[t].ceiling=PyFloat_AsDouble(PyList_GET_ITEM(times_arg,t));
            if (PyErr_Occurred() || !std::isfinite(rows[t].ceiling) || rows[t].ceiling<=0)
                throw std::invalid_argument("slack round duration");
        }
        std::vector<int> available(ranks*slots,nr+1), last_reduce(slots,-1);
        for (int s=0; s<slots; ++s) {
            long owner=PyLong_AsLong(PyList_GET_ITEM(owners_arg,s));
            if (owner < -1 || owner >= ranks || PyErr_Occurred()) throw std::invalid_argument("atom owner");
            if (owner>=0) available[owner*slots+s]=0;
        }
        std::vector<Packet> packets;
        std::vector<int> path_source(routes.size(), -1);
        std::vector<std::vector<int>> incoming(ranks);
        std::vector<int> selected_path;
        std::vector<std::vector<int>> by_round(nr);
        std::vector<unsigned char> assignment, earliest;
        for (Py_ssize_t i=0; i<PyList_GET_SIZE(packets_arg); ++i) {
            int t,p,u,v,copy;
            PyObject* atoms;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(packets_arg,i),"iiiipO!",&t,&p,&u,&v,&copy,
                                  &PyTuple_Type,&atoms)) return nullptr;
            if (t<0 || t>=nr || p<0 || size_t(p)>=routes.size() || u<0 || v<0 || u>=ranks || v>=ranks)
                throw std::invalid_argument("slack packet coordinates");
            Packet pkt{t,p,u,v,bool(copy),{},assignment.size()};
            size_t num=PyTuple_GET_SIZE(atoms);
            if (assignment.size()+num>600000) throw std::invalid_argument("slack delivery budget");
            const auto& route=routes[p];
            auto& row=rows[t];
            row.delay=std::max(row.delay,route.delay);
            row.serial=std::max(row.serial,route.serial);
            for (auto [id,rate]:route.terms) row.load[id]+=num*rate;
            for (size_t j=0; j<num; ++j) {
                long s=PyLong_AsLong(PyTuple_GET_ITEM(atoms,j));
                if (s<0 || s>=slots || PyErr_Occurred()) throw std::invalid_argument("slack slot");
                pkt.slots.push_back(s);
                if (!copy) last_reduce[s]=std::max(last_reduce[s],t);
            }
            if (copy && path_source[p]<0) {path_source[p]=u;incoming[v].push_back(p);}
            selected_path.insert(selected_path.end(),num,p);
            by_round[t].push_back(packets.size());
            packets.push_back(std::move(pkt));
            assignment.insert(assignment.end(),num,t);
            earliest.insert(earliest.end(),num,nr);
        }
        std::vector<double> durations(nr), before_peak(nr), mean_slack(nr);
        {
            AllowThreads release;
            // In a certified AR plan a copy source with no preceding copy
            // already owns the complete sum. Never advance a copy across the
            // last reduction of its slot, even if another root finishes early.
            for (int t=0;t<nr;++t) {
                auto& row=rows[t];
                row.hot=int(std::max_element(row.load.begin(),row.load.end())-row.load.begin());
                row.peak=row.load[row.hot];
                before_peak[t]=row.peak;
                int used=0;
                double occupied=0.;
                for(double v:row.load) if(v>0) {++used;occupied+=v;}
                double budget=std::max(0.,row.ceiling-row.delay);
                mean_slack[t]=used && budget>0 ? std::max(0.,1-occupied/(used*budget)) : 0;
                for (int id:by_round[t]) {
                    auto& packet=packets[id];
                    if (!packet.copy) continue;
                    for(size_t j=0;j<packet.slots.size();++j) {
                        int s=packet.slots[j];
                        int lo=available[packet.src*slots+s];
                        if(last_reduce[s]>=0) {
                            if(lo==nr+1) lo=last_reduce[s]+1;
                            lo=std::max(lo,last_reduce[s]+1);
                        }
                        earliest[packet.offset+j]=std::min(t,lo);
                        row.ready += lo<t;
                    }
                }
                for (int id:by_round[t]) {
                    auto& p=packets[id];
                    if(p.copy) for(int s:p.slots) {
                        int& ready=available[p.dst*slots+s];
                        ready=std::min(ready,t+1);
                    }
                }
            }
            for (auto& paths:incoming)
                std::stable_sort(paths.begin(),paths.end(),[&](int a,int b){return routes[a].serial<routes[b].serial;});
            // A copy sender in an independently verified reduction knows the
            // full slot when it first copies. The slot-wide reduction barrier
            // provides a safe lower bound for roots that did not receive a copy.
            for(const auto& p:packets) if(p.copy) for(int s:p.slots)
                if(last_reduce[s]>=0 && available[p.src*slots+s]>p.round)
                    available[p.src*slots+s]=last_reduce[s]+1;
            for (int t=nr-1;t>0;--t) {
                auto& source=rows[t];

                auto priority=[&](int id) {
                    const auto& r=routes[packets[id].route];
                    double best=0;
                    for(auto [k,rate]:r.terms)
                        best=std::max(best,source.load[k]*(k==focus?1.08:1.));
                    return best;
                };
                auto ids=by_round[t];
                std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return priority(a)>priority(b);});
                for(int id:ids) {
                    const auto& p=packets[id];
                    if(!p.copy) continue;
                    const auto& path=routes[p.route];
                    // Skip cold paths when the round is bandwidth limited.
                    if(priority(id)<.68*before_peak[t] && path.serial<.95*source.ceiling) continue;
                    const auto& alternatives=incoming[p.dst];
                    std::array<int,11> options{};
                    int count=1;
                    options[0]=p.route;
                    size_t offset=size_t(id)*31+t;
                    for(size_t j=0;j<alternatives.size() && count<int(options.size());++j) {
                        size_t index = j<4 ? j : (offset+j*13)%alternatives.size();
                        int candidate=alternatives[index];
                        if(std::find(options.begin(),options.begin()+count,candidate)==options.begin()+count)
                            options[count++]=candidate;
                        if(j>=20) break;
                    }
                    if(prefer_relay && count>1) std::rotate(options.begin(),options.begin()+1,options.begin()+count);
                    size_t remaining=p.slots.size();
                    for(int dst=t-1;dst>=0 && remaining;--dst) {
                        auto& dest=rows[dst];
                        for(int i=0;i<count && remaining;++i) {
                            int q=options[i];
                            const auto& next=routes[q];
                            const double delay=std::max(dest.delay,next.delay);
                            const double budget=dest.ceiling-delay;
                            if(next.serial>dest.ceiling+1e-9 || dest.peak>budget+1e-9) continue;
                            int room=int(remaining);
                            for(auto [k,unit]:next.terms) {
                                double free=std::max(0.,(budget-dest.load[k])/unit+1e-9);
                                if(free<room) room=static_cast<int>(free);
                                if(!room) break;
                            }
                            if(!room) continue;
                            int moved=0;
                            for(size_t j=0;j<p.slots.size() && moved<room;++j) {
                                size_t at=p.offset+j;
                                if(assignment[at]!=t) continue;
                                int s=p.slots[j];
                                int lo=q==p.route ? earliest[at] : std::max(last_reduce[s]+1,available[path_source[q]*slots+s]);
                                if(lo<=dst) {
                                    selected_path[at]=q;
                                    assignment[at]=dst;
                                    ++moved;
                                }
                            }
                            if(!moved) continue;
                            for(auto [k,unit]:path.terms)
                                source.load[k]=std::max(0.,source.load[k]-moved*unit);
                            for(auto [k,unit]:next.terms) {
                                dest.load[k]+=moved*unit;
                                dest.peak=std::max(dest.peak,dest.load[k]);
                            }
                            dest.delay=delay;
                            dest.serial=std::max(dest.serial,next.serial);
                            source.moved+=moved;
                            source.alternate+=q!=p.route ? moved : 0;
                            dest.received+=moved;
                            remaining-=moved;
                        }
                    }
                }
            }
            // Recompute overhead only for paths still present and exact atom
            // serialization floors. The emitter repacks under these ceilings.
            for(auto& row:rows) {row.serial=0;row.delay=0;}
            for(const auto& p:packets) {
                int last=-1, last_path=-1;
                for(size_t j=0;j<p.slots.size();++j) {
                    int t=assignment[p.offset+j];
                    int q=selected_path[p.offset+j];
                    if(t==last && q==last_path) continue;
                    rows[t].delay=std::max(rows[t].delay,routes[q].delay);
                    rows[t].serial=std::max(rows[t].serial,routes[q].serial);
                    last=t; last_path=q;
                }
            }
            for(int t=0;t<nr;++t) {
                const auto& row=rows[t];
                double peak=*std::max_element(row.load.begin(),row.load.end());
                durations[t]=std::max(row.serial,row.delay+peak);
                if(row.serial==0) durations[t]=0;
            }
        }
        Owned summary(PyList_New(nr));
        if(!summary.p) return nullptr;
        for(int t=0;t<nr;++t) {
            const auto& row=rows[t];
            auto item=Py_BuildValue("(dddiiiii)",durations[t],before_peak[t],mean_slack[t],row.ready,row.moved,row.received,row.alternate,row.hot);
            if(!item) return nullptr;
            PyList_SET_ITEM(summary.p,t,item);
        }
        return Py_BuildValue("y#y#N",assignment.data(),Py_ssize_t(assignment.size()),
                             selected_path.data(),Py_ssize_t(selected_path.size()*sizeof(int)),summary.release());
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
      catch(const std::exception& error) {PyErr_SetString(PyExc_ValueError,error.what());return nullptr;}
}
}
