// Verify the emitted transfers against the raw topology, independently of
// the constructor's route cache and capacity estimates.
namespace gather_replay {
struct Edge {
    size_t index;
    std::vector<size_t> pools;
};
using EdgeMap = std::unordered_map<uint64_t, Edge>;

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

long index_of(PyObject* dict, PyObject* name) {
    require(name && PyUnicode_Check(name), "expected a named node or chunk");
    auto item = PyDict_GetItemWithError(dict, name);
    require(item && PyLong_CheckExact(item), "unknown node or chunk");
    long id = PyLong_AsLong(item);
    require(!PyErr_Occurred() && id >= 0, "invalid node or chunk index");
    return id;
}

PyObject* run(PyObject*, PyObject* args) {
    PyObject *nodes, *atoms, *edges, *members, *pools, *rounds;
    int ranks, split;
    double chunk, startup, switching;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iiddd", &PyDict_Type, &nodes,
            &PyDict_Type, &atoms, &PyDict_Type, &edges, &PyDict_Type, &members,
            &PyDict_Type, &pools, &PyList_Type, &rounds, &ranks, &split,
            &chunk, &startup, &switching)) return nullptr;
    try {
        require(ranks > 0 && split > 0 && ranks <= 512 && split <= 512
                && int64_t(ranks) * ranks * split <= 8000000, "replay dimensions too large");
        const size_t width = size_t(ranks) * split;
        const auto node_count = PyDict_Size(nodes);
        require(node_count >= ranks && node_count <= 100000 && PyDict_Size(atoms) == (long)width,
                "inconsistent replay indices");
        Owned pool_ids(PyDict_New()), times(PyList_New(0));
        if (!times.p || !pool_ids.p) return nullptr;
        std::vector<double> rate;
        EdgeMap links;
        PyObject *key, *val;
        Py_ssize_t it = 0;
        while (PyDict_Next(pools, &it, &key, &val)) {
            Owned idx(PyLong_FromSize_t(rate.size()));
            if (!idx.p || PyDict_SetItem(pool_ids.p, key, idx.p) < 0) return nullptr;
            rate.push_back(1e6 / PyFloat_AsDouble(val));
        }
        it = 0;
        while (PyDict_Next(edges, &it, &key, &val)) {
            require(PyTuple_Check(key) && PyTuple_GET_SIZE(key) == 2, "invalid directed edge");
            long u = index_of(nodes, PyTuple_GET_ITEM(key, 0));
            long v = index_of(nodes, PyTuple_GET_ITEM(key, 1));
            Edge info{rate.size(), {}};
            rate.push_back(1e6 / PyFloat_AsDouble(val));
            auto refs = PyDict_GetItemWithError(members, key);
            require(refs && PyTuple_Check(refs), "missing edge membership");
            for (Py_ssize_t p = 0; p < PyTuple_GET_SIZE(refs); ++p)
                info.pools.push_back(index_of(pool_ids.p, PyTuple_GET_ITEM(refs, p)));
            links.emplace((uint64_t(u) << 32) | uint64_t(v), std::move(info));
        }
        if (PyErr_Occurred()) return nullptr;
        std::vector<Py_ssize_t> acquired(ranks * width, -1);
        for (int rank = 0; rank < ranks; ++rank)
            for (int s = 0; s < split; ++s) acquired[rank * width + rank * split + s] = 0;
        size_t missing = (ranks - 1) * width;
        std::vector<double> load(rate.size());
        std::vector<long> path;
        for (Py_ssize_t r = 0; r < PyList_GET_SIZE(rounds); ++r) {
            auto row = PyList_GET_ITEM(rounds, r);
            require(PyDict_Check(row), "invalid round");
            auto number = PyDict_GetItemString(row, "round");
            require(number && PyLong_CheckExact(number) && PyLong_AsSsize_t(number) == r + 1,
                    "round numbers must be sequential");
            auto moves = PyDict_GetItemString(row, "transfers");
            require(moves && PyList_Check(moves), "missing transfer list");
            if (!PyList_GET_SIZE(moves)) continue;
            std::fill(load.begin(), load.end(), 0.);
            double overhead = 0, serial = 0, busy = 0;
            for (Py_ssize_t m = 0; m < PyList_GET_SIZE(moves); ++m) {
                auto move = PyList_GET_ITEM(moves, m);
                require(PyDict_Check(move), "invalid transfer");
                auto action = PyDict_GetItemString(move, "action");
                require(action && PyUnicode_Check(action) && !PyUnicode_CompareWithASCIIString(action,"copy"),
                        "AllGather requires copy");
                long u = index_of(nodes, PyDict_GetItemString(move,"src"));
                long v = index_of(nodes, PyDict_GetItemString(move,"dst"));
                require(u < ranks && v < ranks && u != v, "invalid transfer endpoints");
                auto payload = PyDict_GetItemString(move,"chunk_ranges");
                require(payload && PyList_Check(payload) && PyList_GET_SIZE(payload), "empty or invalid payload");
                double bytes = chunk * PyList_GET_SIZE(payload);
                for (Py_ssize_t k = 0; k < PyList_GET_SIZE(payload); ++k) {
                    size_t a = index_of(atoms, PyList_GET_ITEM(payload,k));
                    require(a < width, "chunk out of range");
                    const auto from = acquired[u * width + a];
                    auto& to = acquired[v * width + a];
                    require(from >= 0 && from <= r, "sender lacks chunk at round start");
                    require(to < 0, "duplicate delivery");
                    to = r + 1;
                    --missing;
                }
                auto route = PyDict_GetItemString(move,"path");
                if (!route) route = PyDict_GetItemString(move,"route");
                require(route && PyList_Check(route) && PyList_GET_SIZE(route) >= 2
                        && PyList_GET_SIZE(route) <= node_count, "invalid path");
                path.clear();
                double inverse = 0;
                for (Py_ssize_t j = 0; j < PyList_GET_SIZE(route); ++j) {
                    long next = index_of(nodes, PyList_GET_ITEM(route,j));
                    require(j == 0 ? next == u : j + 1 == PyList_GET_SIZE(route) ? next == v : next >= ranks,
                            "path endpoint or switch mismatch");
                    require(std::find(path.begin(),path.end(),next) == path.end(), "repeated path node");
                    if (!path.empty()) {
                        auto edge = links.find((uint64_t(path.back()) << 32) | uint64_t(next));
                        require(edge != links.end(), "undeclared directed edge");
                        const auto& e = edge->second;
                        inverse += rate[e.index];
                        busy = std::max(busy, (load[e.index] += bytes) * rate[e.index]);
                        for (size_t p : e.pools) busy = std::max(busy, (load[p] += bytes) * rate[p]);
                    }
                    path.push_back(next);
                }
                double latency = startup + (path.size() - 2) * switching;
                serial = std::max(serial, latency + bytes * inverse);
                overhead = std::max(overhead, latency);
            }
            Owned duration(PyFloat_FromDouble(std::max(serial, overhead + busy)));
            if (!duration.p || PyList_Append(times.p,duration.p) < 0) return nullptr;
        }
        require(!missing, "incomplete final possession");
        return times.release();
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError,e.what());
        return nullptr;
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}
} // namespace gather_replay
