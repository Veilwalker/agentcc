// Switch-only Dijkstra. Arithmetic and tie ordering match Topology.path.
// A compiled adjacency belongs to one request and one routing weight.

struct RoutingGraph {
    std::vector<std::vector<std::pair<size_t, double>>> edges;
};
constexpr const char* routing_capsule = "agentcc.routing.Graph.v1";
void release_routing(PyObject* capsule) {
    delete static_cast<RoutingGraph*>(PyCapsule_GetPointer(capsule, routing_capsule));
}

PyObject* compile_adjacency(PyObject*, PyObject* input) {
    Owned rows(PySequence_Fast(input, "adjacency must be a sequence"));
    if (!rows.p) return nullptr;
    const auto n = PySequence_Fast_GET_SIZE(rows.p);
    if (n < 1 || n > 100000) {
        PyErr_SetString(PyExc_ValueError, "unsupported routing graph size"); return nullptr;
    }
    try {
        auto graph = std::make_unique<RoutingGraph>();
        graph->edges.resize(n);
        size_t count = 0;
        for (Py_ssize_t i = 0; i < n; ++i) {
            Owned row(PySequence_Fast(PySequence_Fast_GET_ITEM(rows.p, i), "invalid adjacency row"));
            if (!row.p) return nullptr;
            for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(row.p); ++j) {
                PyObject* pair = PySequence_Fast_GET_ITEM(row.p, j);
                if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
                    PyErr_SetString(PyExc_ValueError, "invalid adjacency edge"); return nullptr;
                }
                const auto v = PyLong_AsSsize_t(PyTuple_GET_ITEM(pair, 0));
                const double cost = PyFloat_AsDouble(PyTuple_GET_ITEM(pair, 1));
                if (PyErr_Occurred()) return nullptr;
                if (v < 0 || v >= n || cost < 0 || !std::isfinite(cost) || ++count > 2000000) {
                    PyErr_SetString(PyExc_ValueError, "invalid routing edge or work bound"); return nullptr;
                }
                graph->edges[i].emplace_back(v, cost);
            }
        }
        PyObject* result = PyCapsule_New(graph.get(), routing_capsule, release_routing);
        if (result) graph.release();
        return result;
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}

PyObject* shortest_tree(PyObject*, PyObject* args) {
    PyObject *capsule, *priorities;
    Py_ssize_t source, ranks;
    if (!PyArg_ParseTuple(args, "OnnO", &capsule, &source, &ranks, &priorities)) return nullptr;
    auto* graph = static_cast<RoutingGraph*>(PyCapsule_GetPointer(capsule, routing_capsule));
    if (!graph) return nullptr;
    Owned values(PySequence_Fast(priorities, "invalid routing priorities"));
    if (!values.p) return nullptr;
    const size_t n = graph->edges.size();
    if (source < 0 || source >= ranks || ranks < 1 || static_cast<size_t>(ranks) > n
        || static_cast<size_t>(PySequence_Fast_GET_SIZE(values.p)) != n) {
        PyErr_SetString(PyExc_ValueError, "invalid routing source or dimensions"); return nullptr;
    }
    try {
        std::vector<uint64_t> priority(n), ties(n, 0);
        std::vector<double> distances(n, std::numeric_limits<double>::infinity());
        std::vector<Py_ssize_t> previous(n, -1);
        for (size_t i = 0; i < n; ++i) {
            priority[i] = PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(values.p, i));
            if (PyErr_Occurred()) return nullptr;
            if (priority[i] > UINT32_MAX) {
                PyErr_SetString(PyExc_ValueError, "routing priority exceeds 32 bits"); return nullptr;
            }
        }
        using Entry = std::tuple<double, uint64_t, size_t>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
        distances[source] = 0;
        for (const auto& [v, cost] : graph->edges[source]) {
            distances[v] = cost; ties[v] = priority[v]; previous[v] = source;
            if (v >= static_cast<size_t>(ranks)) queue.emplace(cost, ties[v], v);
        }
        // No Python objects are accessed in the numerical search. Keep the
        // GIL here: graph sizes are bounded and exceptions remain ordinary C++.
        while (!queue.empty()) {
            const auto [cost, tie, u] = queue.top(); queue.pop();
            if (cost != distances[u] || tie != ties[u]) continue;
            for (const auto& [v, weight] : graph->edges[u]) {
                const double next = cost + weight;
                const uint64_t next_tie = tie + priority[v];
                if (next < distances[v] || (next == distances[v] && next_tie < ties[v])) {
                    distances[v] = next; ties[v] = next_tie; previous[v] = u;
                    if (v >= static_cast<size_t>(ranks)) queue.emplace(next, next_tie, v);
                }
            }
        }
        Owned p(PyList_New(n)), d(PyList_New(n)), t(PyList_New(n));
        if (!p.p || !d.p || !t.p) return nullptr;
        for (size_t i = 0; i < n; ++i) {
            PyObject *pi = PyLong_FromSsize_t(previous[i]), *di = PyFloat_FromDouble(distances[i]),
                     *ti = PyLong_FromUnsignedLongLong(ties[i]);
            if (!pi || !di || !ti) { Py_XDECREF(pi); Py_XDECREF(di); Py_XDECREF(ti); return nullptr; }
            PyList_SET_ITEM(p.p, i, pi); PyList_SET_ITEM(d.p, i, di); PyList_SET_ITEM(t.p, i, ti);
        }
        return PyTuple_Pack(3, p.p, d.p, t.p);
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}
