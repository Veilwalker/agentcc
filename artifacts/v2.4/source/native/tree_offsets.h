// Round continuous (fractional) pipeline shares to whole multicast trees.
// Moving a tree as a unit preserves its complete causal sequence.
struct TreeLoad { int round, resource; double work; };
struct MulticastTree {
    std::vector<TreeLoad> terms;
    std::vector<double> latency, serial;
};
#include "slot_packing.h"
#include "pace_trees.h"

PyObject* assign_trees(PyObject* args, bool pack, bool pace=false) {
    int rounds, resources, flow_count, lanes;
    int copies=1;
    double bytes;
    PyObject *paths, *packets;
    if (!PyArg_ParseTuple(args, pace ? "iiiidO!O!i" : "iiiidO!O!", &rounds,&resources,&flow_count,&lanes,&bytes,
                          &PyList_Type,&paths,&PyList_Type,&packets,&copies)) return nullptr;

    if ((pace && (copies<1 || copies>8 || rounds>16 || 1LL*flow_count*copies>2048)) || rounds<1 || rounds>64 || (pack ? lanes!=1 || flow_count>32 : lanes<2 || lanes>12) || flow_count<2 || flow_count>1024 ||
        resources<1 || 1LL*resources*(rounds+lanes-1)>2500000 || !(bytes>0) || !std::isfinite(bytes)) {
        PyErr_SetString(PyExc_ValueError,"unsupported tree timing dimensions"); return nullptr;
    }
    const int count=rounds+lanes-1;
    try {
        std::vector<RouteOption> routes;
        for (Py_ssize_t i=0;i<PyList_GET_SIZE(paths);++i) {
            PyObject *terms,*path; RouteOption route;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(paths,i),"O!ddO",&PyTuple_Type,&terms,
                                  &route.overhead,&route.serial,&path)) return nullptr;
            if (!std::isfinite(route.overhead) || route.overhead<0 || !std::isfinite(route.serial) || route.serial<=0) {
                PyErr_SetString(PyExc_ValueError,"invalid multicast route cost"); return nullptr;
            }
            for (Py_ssize_t j=0;j<PyTuple_GET_SIZE(terms);++j) {
                Py_ssize_t id; double rate;
                if (!read_term(terms,j,resources,id,rate)) return nullptr;
                route.terms.emplace_back(id,rate);
            }
            routes.push_back(std::move(route));
        }
        std::vector<MulticastTree> trees(flow_count);
        for (auto& tree:trees) { tree.latency.resize(rounds); tree.serial.resize(rounds); }
        size_t work=0;
        for (Py_ssize_t i=0;i<PyList_GET_SIZE(packets);++i) {
            int r,p; PyObject* atoms;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(packets,i),"iiO!",&r,&p,&PyTuple_Type,&atoms)) return nullptr;
            if (r<0 || r>=rounds || p<0 || static_cast<size_t>(p)>=routes.size()) {
                PyErr_SetString(PyExc_ValueError,"invalid multicast packet"); return nullptr;
            }
            const auto& route=routes[p];
            for (Py_ssize_t j=0;j<PyTuple_GET_SIZE(atoms);++j) {
                long a=PyLong_AsLong(PyTuple_GET_ITEM(atoms,j));
                if (PyErr_Occurred() || a<0 || a>=flow_count || (work+=route.terms.size())>1800000) {
                    PyErr_SetString(PyExc_ValueError,"invalid multicast tree or incidence budget"); return nullptr;
                }
                auto& tree=trees[a];
                tree.latency[r]=std::max(tree.latency[r],route.overhead);
                tree.serial[r]=std::max(tree.serial[r],route.overhead+bytes*route.serial);
                for (auto [k,rate]:route.terms) tree.terms.push_back({r,static_cast<int>(k),bytes*rate});
            }
        }
        std::vector<int> best_delays(flow_count), delays(flow_count);
        std::vector<double> best_times(count);
        double best=std::numeric_limits<double>::infinity();
        {
            AllowThreads release;
            for (auto& tree:trees) {
                auto& ts=tree.terms;
                std::sort(ts.begin(),ts.end(),[](auto& a,auto& b){return std::tie(a.round,a.resource)<std::tie(b.round,b.resource);});
                size_t kept=0;
                for (auto term:ts) {
                    if (kept && ts[kept-1].round==term.round && ts[kept-1].resource==term.resource)
                        ts[kept-1].work+=term.work;
                    else ts[kept++]=term;
                }
                ts.resize(kept);
            }
        }
        if (pace) {
            AllowThreads release;
            pace_trees(trees,rounds,resources,lanes,copies,best_delays,best_times);
        } else if (pack) {
            AllowThreads release;
            pack_slots(trees, rounds, resources, best_delays, best_times);
        } else {
            AllowThreads release;
            // A dense array plus one tournament tree per round supports exact
            // removal/reinsertion without rescanning unmodified resources.
            std::vector<TimingRow> rows;
            for (int t=0;t<count;++t) rows.emplace_back(resources);
            auto update=[&](const MulticastTree& tree,int d,int sign) {
                for (auto term:tree.terms) {
                    auto& row=rows[term.round+d];
                    auto& load=row.loads[term.resource];
                    load=std::max(0.,load+sign*term.work);
                    row.peak.set(term.resource,load);
                }
            };
            auto save=[&] {
                std::vector<double> lat(count),serial(count),times(count);
                for(int a=0;a<flow_count;++a) for(int r=0;r<rounds;++r) {
                    int t=delays[a]+r;
                    lat[t]=std::max(lat[t],trees[a].latency[r]);
                    serial[t]=std::max(serial[t],trees[a].serial[r]);
                }
                double total=0;
                for(int t=0;t<count;++t) {times[t]=std::max(serial[t],lat[t]+rows[t].peak.maximum());total+=times[t];}
                if(total<best) {best=total;best_times=std::move(times);best_delays=delays;}
            };
            for(int a=0;a<flow_count;++a) {delays[a]=(a^(a>>3))%lanes;update(trees[a],delays[a],1);}
            save();
            for(int pass=0;pass<3;++pass) {
                int moves=0;
                for(int j=0;j<flow_count;++j) {
                    int a=pass%2?flow_count-1-j:j;
                    auto& tree=trees[a]; int old=delays[a];
                    update(tree,old,-1);
                    auto score=[&](int d) {
                        double delta=0,energy=0;
                        std::array<double,64> growth{};
                        for(auto term:tree.terms) {
                            auto& row=rows[term.round+d];
                            double before=row.loads[term.resource];
                            growth[term.round]=std::max(growth[term.round],
                                before+term.work-row.peak.maximum());
                            energy+=before*term.work;
                        }
                        for(int r=0;r<rounds;++r)delta+=growth[r];
                        return std::make_pair(std::round(delta*1e7),energy);
                    };
                    auto price=score(old);
                    for(int d=0;d<lanes;++d) if(d!=old) {
                        auto s=score(d);
                        if(s<price) {price=s;delays[a]=d;}
                    }
                    moves+=delays[a]!=old;
                    update(tree,delays[a],1);
                }
                save(); if(!moves) break;
            }
        }
        Owned offsets(PyList_New(best_delays.size())), times(PyList_New(count));
        if(!offsets.p || !times.p) return nullptr;
        for(size_t i=0;i<best_delays.size();++i) {
            auto* v=PyLong_FromLong(best_delays[i]);if(!v)return nullptr;PyList_SET_ITEM(offsets.p,i,v);
        }
        for(int i=0;i<count;++i) {
            auto* v=PyFloat_FromDouble(best_times[i]);if(!v)return nullptr;PyList_SET_ITEM(times.p,i,v);
        }
        return Py_BuildValue("OO",offsets.p,times.p);
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
}

PyObject* choose_tree_offsets(PyObject*, PyObject* args) { return assign_trees(args, false); }
PyObject* choose_slot_trees(PyObject*, PyObject* args) { return assign_trees(args, true); }

PyObject* paced_launch(PyObject*, PyObject* args){return assign_trees(args,false,true);}
