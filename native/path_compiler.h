// Request-local numeric graph for compiling many different legal paths.
// Preserve edge order, first-seen pool order and Python arithmetic exactly.
struct CapacityEdge {
    double inverse;
    PyObject* term;
    std::vector<size_t> pools;
};
struct CapacityGraph {
    PyObject* nodes = PyDict_New();
    std::unordered_map<uint64_t, CapacityEdge> edges;
    std::vector<std::pair<Py_ssize_t, double>> pools;
    std::vector<std::array<PyObject*,4>> pool_terms;
    double startup = 0, switching = 0;
    ~CapacityGraph() {
        for (auto& edge : edges) Py_DECREF(edge.second.term);
        for (auto& row : pool_terms) for (auto* term : row) Py_XDECREF(term);
        Py_XDECREF(nodes);
    }
};
constexpr const char* capacity_graph_name = "agentcc.capacity.Graph.v1";
void release_capacity_graph(PyObject* capsule) {
    delete static_cast<CapacityGraph*>(PyCapsule_GetPointer(capsule, capacity_graph_name));
}

bool capacity_node(CapacityGraph& graph, PyObject* name, uint64_t& id, bool insert) {
    PyObject* value = PyDict_GetItemWithError(graph.nodes, name);
    if (!value) {
        if (PyErr_Occurred()) return false;
        if (!insert) { PyErr_SetObject(PyExc_KeyError, name); return false; }
        id = static_cast<uint64_t>(PyDict_Size(graph.nodes));
        if (id >= 100000) {
            PyErr_SetString(PyExc_ValueError, "capacity graph node budget exceeded"); return false;
        }
        Owned number(PyLong_FromUnsignedLongLong(id));
        return number.p && PyDict_SetItem(graph.nodes, name, number.p) == 0;
    }
    id = PyLong_AsUnsignedLongLong(value);
    return !PyErr_Occurred();
}

bool capacity_index(PyObject* indices, const char* tag, PyObject* key, Py_ssize_t& index) {
    Owned composite(Py_BuildValue("sO", tag, key));
    if (!composite.p) return false;
    PyObject* value = PyDict_GetItemWithError(indices, composite.p);
    if (!value) {
        if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, composite.p);
        return false;
    }
    index = PyLong_AsSsize_t(value);
    if (PyErr_Occurred()) return false;
    if (index < 0) {
        PyErr_SetString(PyExc_ValueError, "negative capacity index"); return false;
    }
    return true;
}

PyObject* make_capacity_graph(PyObject*, PyObject* args) {
    PyObject *edges, *members, *resources, *indices;
    double startup, switching;
    if (!PyArg_ParseTuple(args, "OOOOdd", &edges, &members, &resources, &indices,
                          &startup, &switching)) return nullptr;
    if (!PyDict_Check(edges) || !PyDict_Check(members) || !PyDict_Check(resources)
        || !PyDict_Check(indices) || !std::isfinite(startup) || startup < 0
        || !std::isfinite(switching) || switching < 0 || PyDict_Size(edges) > 2000000) {
        PyErr_SetString(PyExc_ValueError, "invalid capacity graph"); return nullptr;
    }
    try {
        auto graph = std::make_unique<CapacityGraph>();
        Owned pool_ids(PyDict_New());
        if (!graph->nodes || !pool_ids.p) return nullptr;
        graph->startup = startup; graph->switching = switching;
        Py_ssize_t position = 0;
        PyObject *key, *value;
        while (PyDict_Next(resources, &position, &key, &value)) {
            double capacity = PyFloat_AsDouble(value);
            if (PyErr_Occurred()) return nullptr;
            if (!std::isfinite(capacity) || capacity <= 0) {
                PyErr_SetString(PyExc_ValueError, "invalid pool capacity"); return nullptr;
            }
            Py_ssize_t index;
            if (!capacity_index(indices, "pool", key, index)) return nullptr;
            Owned id(PyLong_FromSize_t(graph->pools.size()));
            if (!id.p || PyDict_SetItem(pool_ids.p, key, id.p) < 0) return nullptr;
            graph->pools.emplace_back(index, capacity);
        }
        position = 0;
        graph->pool_terms.resize(graph->pools.size(), {});
        while (PyDict_Next(edges, &position, &key, &value)) {
            if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) != 2) {
                PyErr_SetString(PyExc_ValueError, "edge must have two endpoints"); return nullptr;
            }
            uint64_t source, destination;
            if (!capacity_node(*graph, PyTuple_GET_ITEM(key, 0), source, true)
                || !capacity_node(*graph, PyTuple_GET_ITEM(key, 1), destination, true)) return nullptr;
            double capacity = PyFloat_AsDouble(value);
            if (PyErr_Occurred()) return nullptr;
            if (!std::isfinite(capacity) || capacity <= 0) {
                PyErr_SetString(PyExc_ValueError, "invalid edge capacity"); return nullptr;
            }
            Py_ssize_t index;
            if (!capacity_index(indices, "edge", key, index)) return nullptr;
            PyObject* raw = PyDict_GetItemWithError(members, key);
            if (!raw) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, key); return nullptr; }
            Owned sequence(PySequence_Fast(raw, "pool members must be a sequence"));
            if (!sequence.p) return nullptr;
            std::vector<size_t> pools;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(sequence.p); ++i) {
                PyObject* pool = PySequence_Fast_GET_ITEM(sequence.p, i);
                PyObject* id = PyDict_GetItemWithError(pool_ids.p, pool);
                if (!id) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, pool); return nullptr; }
                pools.push_back(PyLong_AsSize_t(id));
                if (PyErr_Occurred()) return nullptr;
            }
            double inverse = 1e6 / capacity;
            Owned term(Py_BuildValue("nd", index, inverse));
            if (!term.p) return nullptr;
            graph->edges.emplace((source << 32) | destination,
                                 CapacityEdge{inverse, term.p, std::move(pools)});
            term.release();
        }
        PyObject* result = PyCapsule_New(graph.get(), capacity_graph_name, release_capacity_graph);
        if (result) graph.release();
        return result;
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}

PyObject* compile_one(CapacityGraph* graph, PyObject* path) {
    if (!PyTuple_Check(path) || PyTuple_GET_SIZE(path) < 2 || PyTuple_GET_SIZE(path) > 100000) {
        PyErr_SetString(PyExc_ValueError, "invalid capacity path"); return nullptr;
    }
    try {
        std::vector<const CapacityEdge*> edges;
        std::vector<std::pair<size_t, size_t>> pools;
        uint64_t source;
        if (!capacity_node(*graph, PyTuple_GET_ITEM(path, 0), source, false)) return nullptr;
        double serial = 0.0;
        const auto n = PyTuple_GET_SIZE(path);
        edges.reserve(static_cast<size_t>(n - 1));
        for (Py_ssize_t i = 1; i < n; ++i) {
            uint64_t destination;
            if (!capacity_node(*graph, PyTuple_GET_ITEM(path, i), destination, false)) return nullptr;
            auto found = graph->edges.find((source << 32) | destination);
            if (found == graph->edges.end()) {
                PyErr_SetString(PyExc_ValueError, "path contains undeclared directed edge"); return nullptr;
            }
            const auto& edge = found->second;
            serial += edge.inverse;
            edges.push_back(&edge);
            for (size_t pool : edge.pools) {
                auto it = std::find_if(pools.begin(), pools.end(),
                                      [pool](const auto& entry) { return entry.first == pool; });
                if (it == pools.end()) pools.emplace_back(pool, 1);
                else ++it->second;
            }
            source = destination;
        }
        Owned terms(PyTuple_New(static_cast<Py_ssize_t>(edges.size() + pools.size())));
        if (!terms.p) return nullptr;
        Py_ssize_t position = 0;
        for (const auto* edge : edges) {
            Py_INCREF(edge->term);
            PyTuple_SET_ITEM(terms.p, position++, edge->term);
        }
        for (const auto& pool : pools) {
            const auto& spec = graph->pools[pool.first];
            // The scaled rate of a pool is shared by paths with equal incidence.
            PyObject* term;
            if (pool.second<=4) {
                auto& cached = graph->pool_terms[pool.first][pool.second-1];
                if (!cached) cached=Py_BuildValue("nd",spec.first,pool.second*1e6/spec.second);
                term=Py_XNewRef(cached);
            } else term=Py_BuildValue("nd",spec.first,pool.second*1e6/spec.second);
            if (!term) return nullptr;
            PyTuple_SET_ITEM(terms.p, position++, term);
        }
        return Py_BuildValue("OddO", terms.p, graph->startup + (n - 2) * graph->switching, serial, path);
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}

PyObject* compile_cached_path(PyObject*, PyObject* args) {
    PyObject *capsule, *path;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &path)) return nullptr;
    auto* graph = static_cast<CapacityGraph*>(PyCapsule_GetPointer(capsule, capacity_graph_name));
    if (!graph) return nullptr;
    return compile_one(graph,path);
}

// Traverse stage packets once. The Python route catalog defines all option and
// packet ordering, and both caches are shared with subsequent materialization.
PyObject* stage_route_jobs(PyObject*, PyObject* args) {
    PyObject *cap, *entries, *pairs, *paths, *catalog;
    if (!PyArg_ParseTuple(args,"OO!O!O!O",&cap,&PyList_Type,&entries,&PyDict_Type,&pairs,
                          &PyDict_Type,&paths,&catalog)) return nullptr;
    auto* graph = static_cast<CapacityGraph*>(PyCapsule_GetPointer(cap, capacity_graph_name));
    if (!graph || !PyCallable_Check(catalog)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError,"expected a route catalog");
        return nullptr;
    }
    Owned jobs(PyList_New(PyList_GET_SIZE(entries)));
    if (!jobs.p) return nullptr;
    for (Py_ssize_t i=0; i<PyList_GET_SIZE(entries); ++i) {
        Owned row(PySequence_Fast(PyList_GET_ITEM(entries,i),"invalid stage entry"));
        if (!row.p) return nullptr;
        if (PySequence_Fast_GET_SIZE(row.p)!=2) {
            PyErr_SetString(PyExc_ValueError,"invalid stage entry"); return nullptr;
        }
        PyObject* count=PySequence_Fast_GET_ITEM(row.p,1);
        Owned key(PySequence_Fast(PySequence_Fast_GET_ITEM(row.p,0),"invalid stage movement"));
        if (!key.p) return nullptr;
        if (PySequence_Fast_GET_SIZE(key.p)!=3) {
            PyErr_SetString(PyExc_ValueError,"invalid stage movement"); return nullptr;
        }
        PyObject* source=PySequence_Fast_GET_ITEM(key.p,1);
        PyObject* destination=PySequence_Fast_GET_ITEM(key.p,2);
        Owned pair(PyTuple_Pack(2,source,destination));
        if (!pair.p) return nullptr;
        PyObject* options = PyDict_GetItemWithError(pairs,pair.p);
        if (!options) {
            if (PyErr_Occurred()) return nullptr;
            Owned raw(PyObject_CallObject(catalog,pair.p));
            if (!raw.p) return nullptr;
            Owned seq(PySequence_Fast(raw.p,"invalid route options"));
            if (!seq.p) return nullptr;
            Owned compiled(PyTuple_New(PySequence_Fast_GET_SIZE(seq.p)));
            if (!compiled.p) return nullptr;
            for (Py_ssize_t j=0; j<PySequence_Fast_GET_SIZE(seq.p); ++j) {
                PyObject* element = PySequence_Fast_GET_ITEM(seq.p,j);
                Owned path(PyTuple_CheckExact(element) ? Py_NewRef(element) : PySequence_Tuple(element));
                if (!path.p) return nullptr;
                PyObject* hit = PyDict_GetItemWithError(paths,path.p);
                if (hit) {
                    PyTuple_SET_ITEM(compiled.p,j,Py_NewRef(hit));
                } else {
                    if (PyErr_Occurred()) return nullptr;
                    Owned fresh(compile_one(graph,path.p));
                    if (!fresh.p || PyDict_SetItem(paths,path.p,fresh.p)<0) return nullptr;
                    PyTuple_SET_ITEM(compiled.p,j,fresh.release());
                }
            }
            if (PyDict_SetItem(pairs,pair.p,compiled.p)<0) return nullptr;
            options=compiled.p;
        }
        PyObject* job=PyTuple_Pack(2,options,count);
        if (!job) return nullptr;
        PyList_SET_ITEM(jobs.p,i,job);
    }
    return jobs.release();
}
