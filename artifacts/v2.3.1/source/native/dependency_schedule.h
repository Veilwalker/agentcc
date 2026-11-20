// A bounded coordinate search on the data-dependency graph of a validated
// schedule. A vertex moves one cohort of a physical transfer. Rank state is
// read at the start of a round, so a read may coincide with a later write.
struct TransferVertex {
    int time, route, count = 0;
    std::vector<std::pair<int,int>> before, after;
};
struct SlotHistory {
    std::vector<int> writers, readers;
};
struct TimingRow {
    std::vector<double> loads;
    MaxLoads peak;
    explicit TimingRow(int resources) : loads(resources), peak(resources) {}
    void update(const RouteOption& path, double bytes) {
        for (auto [index, rate] : path.terms) {
            loads[index] = std::max(0.0, loads[index] + bytes * rate);
            peak.set(index, loads[index]);
        }
    }
    std::pair<double,double> insertion(const RouteOption& path, double bytes) const {
        double busiest = 0, energy = 0;
        for (auto [index, rate] : path.terms) {
            const double work = rate * bytes, old = loads[index];
            busiest = std::max(busiest, old + work);
            energy += work * (2 * old + work);
        }
        return {std::max(0.0, busiest - peak.maximum()), energy};
    }
};
inline void unique_ids(std::vector<int>& ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}
PyObject* retime_transfers(PyObject*, PyObject* args) {
    int n, slots, resources, rounds, limit, bands, reduction;
    double bytes;
    PyObject *paths, *packets;
    if (!PyArg_ParseTuple(args, "iiiiiiidO!O!", &n,&slots,&resources,&rounds,&limit,
            &bands,&reduction,&bytes,&PyList_Type,&paths,&PyList_Type,&packets)) return nullptr;
    const int packet_count = PyList_GET_SIZE(packets);
    if (n < 2 || slots < 1 || static_cast<long long>(n)*slots > 300000
        || rounds < 1 || limit < rounds || limit > 128 || bands < 1 || bands > 16
        || resources < 1 || static_cast<long long>(resources)*limit > 4000000
        || packet_count > 60000 || !std::isfinite(bytes) || bytes <= 0) {
        PyErr_SetString(PyExc_ValueError,"unsupported timing dimensions"); return nullptr;
    }
    try {
        std::vector<RouteOption> routes;
        for (Py_ssize_t i=0; i<PyList_GET_SIZE(paths); ++i) {
            PyObject *terms, *path;
            RouteOption r;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(paths,i),"O!ddO",&PyTuple_Type,&terms,
                                  &r.overhead,&r.serial,&path)) return nullptr;
            for (Py_ssize_t j=0; j<PyTuple_GET_SIZE(terms); ++j) {
                Py_ssize_t index; double rate;
                if (!read_term(terms,j,resources,index,rate)) return nullptr;
                r.terms.emplace_back(index,rate);
            }
            routes.push_back(std::move(r));
        }
        struct PacketInput { int time, path, src, dst; std::vector<int> atoms; };
        std::vector<PacketInput> inputs;
        int previous = -1;
        size_t deliveries = 0;
        for (int i=0; i<packet_count; ++i) {
            PyObject* atoms; PacketInput p;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(packets,i),"iiiiO!",&p.time,&p.path,
                                  &p.src,&p.dst,&PyTuple_Type,&atoms)) return nullptr;
            if (p.src<0 || p.src>=n || p.dst<0 || p.dst>=n || p.time<previous ||
                p.time>=rounds || p.time<0 || p.path<0 || size_t(p.path)>=routes.size()) {
                PyErr_SetString(PyExc_ValueError,"invalid timing packet"); return nullptr;
            }
            previous = p.time;
            deliveries += PyTuple_GET_SIZE(atoms);
            if (deliveries > 1000000) {
                PyErr_SetString(PyExc_ValueError,"too many timing deliveries"); return nullptr;
            }
            for (Py_ssize_t j=0; j<PyTuple_GET_SIZE(atoms); ++j) {
                long a=PyLong_AsLong(PyTuple_GET_ITEM(atoms,j));
                if (a<0 || a>=slots || PyErr_Occurred()) {
                    PyErr_SetString(PyExc_ValueError,"invalid timing slot"); return nullptr;
                }
                p.atoms.push_back(a);
            }
            inputs.push_back(std::move(p));
        }
        std::vector<unsigned char> assignment(static_cast<size_t>(packet_count)*bands);
        std::vector<double> durations(limit);
        {
            AllowThreads unlock;
            std::vector<TransferVertex> nodes;
            std::vector<int> mapping(assignment.size(),-1);
            std::vector<SlotHistory> histories(n*slots);
            // A strict edge requires the producer to finish in an earlier round.
            // A zero-lag edge lets a read and a subsequent update share a round.
            auto add_edges = [&](int node, const std::vector<int>& deps, int lag) {
                for (int p : deps) {
                    nodes[node].before.emplace_back(p,lag);
                    nodes[p].after.emplace_back(node,lag);
                }
            };
            for (int start=0; start<packet_count;) {
                int stop=start;
                while (stop<packet_count && inputs[stop].time==inputs[start].time) ++stop;
                std::unordered_map<int,std::vector<int>> writes;
                for (int i=start; i<stop; ++i) {
                    const auto& p=inputs[i];
                    for (int a : p.atoms) {
                        int band = (a ^ (a>>3)) % bands;
                        int& node = mapping[i*bands+band];
                        if (node<0) {
                            node=static_cast<int>(nodes.size());
                            nodes.push_back({p.time,p.path});
                        }
                        ++nodes[node].count;
                        auto& h=histories[p.src*slots+a];
                        add_edges(node,h.writers,1);
                        if (reduction) h.readers.push_back(node);
                        writes[p.dst*slots+a].push_back(node);
                    }
                }
                for (auto& [key,ids] : writes) {
                    unique_ids(ids);
                    auto& h=histories[key];
                    if (reduction) {
                        unique_ids(h.readers);
                        for (int v : ids) {
                            add_edges(v,h.readers,0);
                            add_edges(v,h.writers,0);
                        }
                        h.readers.clear();
                    }
                    h.writers.swap(ids);
                }
                start=stop;
            }
            std::vector<TimingRow> rows;
            for (int i=0;i<limit;++i) rows.emplace_back(resources);
            for (auto& v : nodes) {
                auto compact = [](auto& list) {
                    std::sort(list.begin(),list.end());
                    list.erase(std::unique(list.begin(),list.end()),list.end());
                };
                compact(v.before); compact(v.after);
                rows[v.time].update(routes[v.route],v.count*bytes);
            }
            for (int pass=0;pass<4;++pass) {
                int moved=0;
                for (size_t j=0;j<nodes.size();++j) {
                    auto& v=nodes[pass%2 ? j : nodes.size()-j-1];
                    int lo=0, hi=limit-1;
                    for (auto [p,lag] : v.before) lo=std::max(lo,nodes[p].time+lag);
                    for (auto [p,lag] : v.after) hi=std::min(hi,nodes[p].time-lag);
                    if (hi<=lo) continue;
                    auto& path=routes[v.route];
                    double work=v.count*bytes;
                    rows[v.time].update(path,-work);
                    int best=v.time;
                    auto score = [&](int t) {
                        auto [delta,energy]=rows[t].insertion(path,work);
                        // Keep empty tail rounds expensive enough to pay startup.
                        if (rows[t].peak.maximum()<1e-9) delta=std::max(delta,bytes*path.serial)+path.overhead;
                        return std::make_pair(std::round(delta*1e7),std::round(energy*1e5));
                    };
                    auto price=score(best);
                    for (int t=lo;t<=hi;++t) {
                        if (t==v.time) continue;
                        auto value=score(t);
                        if (value<price) { best=t; price=value; }
                    }
                    moved += best!=v.time;
                    v.time=best;
                    rows[best].update(path,work);
                }
                if (!moved) break;
            }
            std::vector<double> latency(limit);
            for (const auto& v : nodes) {
                const auto& path=routes[v.route];
                latency[v.time]=std::max(latency[v.time],path.overhead);
                durations[v.time]=std::max(durations[v.time],path.overhead+bytes*path.serial);
            }
            for (int t=0;t<limit;++t)
                durations[t]=std::max(durations[t],rows[t].peak.maximum()+latency[t]);
            for (size_t i=0;i<mapping.size();++i)
                if (mapping[i]>=0) assignment[i]=static_cast<unsigned char>(nodes[mapping[i]].time);
        }
        Owned times(PyList_New(limit));
        if (!times.p) return nullptr;
        for (int t=0;t<limit;++t) {
            auto* f=PyFloat_FromDouble(durations[t]);
            if (!f) return nullptr;
            PyList_SET_ITEM(times.p,t,f);
        }
        return Py_BuildValue("y#O",assignment.data(),Py_ssize_t(assignment.size()),times.p);
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}
