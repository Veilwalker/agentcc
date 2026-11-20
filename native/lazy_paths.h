// Resolve the original SHA-256 path tie only when a requested destination
// actually has equal-cost routes. Ordinary minimum distances need no hashes.
namespace lazy_paths {
using Edge = std::pair<size_t,double>;
constexpr const char* graph_name = "agentcc.routing.LazyGraph.v1";
constexpr const char* tree_name = "agentcc.routing.LazyTree.v1";

struct Graph {
    std::vector<std::vector<Edge>> outgoing, incoming;
};
struct Tree {
    PyObject* owner;
    Graph* graph;
    size_t ranks, source;
    std::vector<double> distance;
    std::vector<int> parent;
    std::vector<uint64_t> tie;
    std::vector<unsigned> mark;
    unsigned generation = 0;
    Tree(PyObject* handle, Graph* g, size_t n, size_t from)
        : owner(handle), graph(g), ranks(n), source(from),
          distance(g->outgoing.size(), std::numeric_limits<double>::infinity()),
          parent(distance.size(), -1), tie(distance.size(), UINT64_MAX), mark(distance.size()) {
        Py_INCREF(owner);
        parent[source] = static_cast<int>(source);
        tie[source] = 0;
        distance[source] = 0.;
    }
    ~Tree() {Py_DECREF(owner);}

    bool shortest() {
        using Entry = std::pair<double,size_t>;
        std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> queue;
        queue.emplace(0.,source);
        while (!queue.empty()) {
            auto [cost,u] = queue.top(); queue.pop();
            if (cost!=distance[u]) continue;
            for(auto [v,w]:graph->outgoing[u]) {
                double next = cost + w;
                // Equal distances along an edge violate the strictly increasing
                // order used during backtracking. Leave this case to the full
                // cost-and-priority Dijkstra (and its numerical semantics).
                if(next<=cost || !std::isfinite(next)) return false;
                if(next<distance[v]) {
                    distance[v]=next;
                    if(v>=ranks) queue.emplace(next,v);
                }
            }
        }
        return true;
    }

    bool shortest_predecessor(size_t u,size_t v,double w) const {
        return (u==source || u>=ranks) && distance[u]+w==distance[v];
    }

    bool path_tie(size_t v, PyObject* priority) {
        std::vector<size_t> missing;
        while(tie[v]==UINT64_MAX) {
            missing.push_back(v);
            v=static_cast<size_t>(parent[v]);
        }
        uint64_t sum=tie[v];
        for(auto it=missing.rbegin(); it!=missing.rend(); ++it) {
            const size_t node=*it;
            if(node>=ranks) {
                Owned value(PyObject_CallFunction(priority,"nn",static_cast<Py_ssize_t>(source),
                                                   static_cast<Py_ssize_t>(node)));
                if(!value.p) return false;
                unsigned long add=PyLong_AsUnsignedLong(value.p);
                if(PyErr_Occurred()) return false;
                if(add>UINT32_MAX) {PyErr_SetString(PyExc_ValueError,"invalid path tie"); return false;}
                sum+=add;
            }
            tie[node]=sum;
        }
        return true;
    }

    bool resolve(size_t target, PyObject* priority) {
        if(parent[target]>=0) return true;
        std::vector<size_t> needed{target};
        if(++generation==0) {std::fill(mark.begin(),mark.end(),0);++generation;}
        mark[target]=generation;
        for(size_t i=0;i<needed.size();++i) {
            size_t v=needed[i];
            for(auto [u,w]:graph->incoming[v])
                if(parent[u]<0 && mark[u]!=generation && shortest_predecessor(u,v,w)) {
                    needed.push_back(u);mark[u]=generation;
                }
        }
        // A shortest-path predecessor always has strictly smaller distance.
        std::sort(needed.begin(),needed.end(),[&](size_t u,size_t v){return distance[u]<distance[v];});
        for(size_t v:needed) {
            int best=-1;
            for(auto [u,w]:graph->incoming[v]) {
                if(!shortest_predecessor(u,v,w)) continue;
                if(best<0) {best=static_cast<int>(u);continue;}
                if(!path_tie(u,priority) || !path_tie(best,priority)) return false;
                // For the same child, the added priority is identical.
                // For identical total priorities Dijkstra fixes the parent
                // popped first: lowest predecessor distance, then node index.
                if(std::tie(tie[u],distance[u],u) <
                   std::make_tuple(tie[best],distance[best],static_cast<size_t>(best)))
                    best=static_cast<int>(u);
            }
            if(best<0) {PyErr_SetString(PyExc_ValueError,"unresolved shortest path");return false;}
            parent[v]=best;
        }
        return true;
    }
};

void free_graph(PyObject* c){delete static_cast<Graph*>(PyCapsule_GetPointer(c,graph_name));}
void free_tree(PyObject* c){delete static_cast<Tree*>(PyCapsule_GetPointer(c,tree_name));}

PyObject* compile(PyObject* self, PyObject* rows) {
    Owned original(compile_adjacency(self,rows));
    if(!original.p) return nullptr;
    auto* parsed=static_cast<RoutingGraph*>(PyCapsule_GetPointer(original.p,routing_capsule));
    if(!parsed) return nullptr;
    try {
        auto g=std::make_unique<Graph>();
        g->outgoing=std::move(parsed->edges);
        g->incoming.resize(g->outgoing.size());
        for(size_t u=0;u<g->outgoing.size();++u)
            for(auto [v,w]:g->outgoing[u]) g->incoming[v].emplace_back(u,w);
        PyObject* cap=PyCapsule_New(g.get(),graph_name,free_graph);
        if(cap) g.release();
        return cap;
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
}
PyObject* build(PyObject*, PyObject* args) {
    PyObject* handle;
    Py_ssize_t source,ranks;
    if(!PyArg_ParseTuple(args,"Onn",&handle,&source,&ranks)) return nullptr;
    auto* graph=static_cast<Graph*>(PyCapsule_GetPointer(handle,graph_name));
    if(!graph) return nullptr;
    if(source<0 || ranks<=source || size_t(ranks)>graph->outgoing.size()) {
        PyErr_SetString(PyExc_ValueError,"invalid lazy routing coordinates");return nullptr;
    }
    try {
        auto t=std::make_unique<Tree>(handle,graph,ranks,source);
        if(!t->shortest()) Py_RETURN_NONE;
        PyObject* cap=PyCapsule_New(t.get(),tree_name,free_tree);
        if(cap) t.release();
        return cap;
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
}
PyObject* path(PyObject*, PyObject* args) {
    PyObject *handle,*priority;
    Py_ssize_t target;
    if(!PyArg_ParseTuple(args,"OnO",&handle,&target,&priority)) return nullptr;
    auto* tree=static_cast<Tree*>(PyCapsule_GetPointer(handle,tree_name));
    if(!tree) return nullptr;
    if(target<0 || size_t(target)>=tree->ranks || !PyCallable_Check(priority)) {
        PyErr_SetString(PyExc_ValueError,"invalid lazy path request");return nullptr;
    }
    if(!std::isfinite(tree->distance[target])) Py_RETURN_NONE;
    try {
        if(!tree->resolve(target,priority)) return nullptr;
        std::vector<size_t> result;
        for(size_t v=target;v!=tree->source;v=static_cast<size_t>(tree->parent[v])) result.push_back(v);
        result.push_back(tree->source);
        Owned out(PyList_New(result.size()));
        if(!out.p) return nullptr;
        for(size_t i=0;i<result.size();++i) {
            PyObject* id=PyLong_FromSize_t(result[result.size()-1-i]);
            if(!id) return nullptr;
            PyList_SET_ITEM(out.p,i,id);
        }
        return out.release();
    } catch(const std::bad_alloc&) {return PyErr_NoMemory();}
}
}
