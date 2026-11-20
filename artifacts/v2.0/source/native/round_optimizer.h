// Whole-round route balancing. Inputs are checked before the GIL is released.
// Each move preserves bytes and endpoints; the caller separately verifies Replay.

struct AllowThreads {
    PyThreadState* state = PyEval_SaveThread();
    ~AllowThreads() { PyEval_RestoreThread(state); }
};

struct RouteOption {
    std::vector<std::pair<size_t, double>> terms;
    double overhead, serial;
};
struct Packet {
    std::shared_ptr<std::vector<RouteOption>> options;
    const std::vector<RouteOption>& routes() const { return *options; }
    double bytes, packet_bytes;
};

class MaxLoads {
    size_t base = 1;
    std::vector<double> tree;
public:
    explicit MaxLoads(size_t n) {
        while (base < n) base *= 2;
        tree.assign(base * 2, 0.0);
    }
    void set(size_t index, double value) {
        size_t at = base + index;
        tree[at] = value;
        while ((at /= 2)) {
            const double next = std::max(tree[at * 2], tree[at * 2 + 1]);
            if (tree[at] == next) break;
            tree[at] = next;
        }
    }
    double maximum() const { return tree[1]; }
};

struct RoundOptimizer {
    std::vector<double> loads;
    MaxLoads maximum;
    std::multiset<double> overheads{0.0}, serials{0.0};
    explicit RoundOptimizer(size_t n) : loads(n, 0.0), maximum(n) {}

    double time() const {
        return std::max(*serials.rbegin(), *overheads.rbegin() + maximum.maximum());
    }
    std::array<double, 4> score(const Packet& packet, const RouteOption& route) const {
        double busy = 0.0, energy = 0.0;
        for (auto [index, factor] : route.terms) {
            const double before = loads[index], after = before + packet.bytes * factor;
            busy = std::max(busy, after);
            energy += after * after - before * before;
        }
        double serial = route.overhead + packet.packet_bytes * route.serial;
        return {std::max({*serials.rbegin(), serial,
                   std::max(*overheads.rbegin(), route.overhead) +
                   std::max(maximum.maximum(), busy)}), busy, energy, serial};
    }
    void change(const Packet& packet, const RouteOption& route, bool add) {
        for (auto [index, factor] : route.terms) {
            loads[index] = std::max(0.0, loads[index] + (add ? 1 : -1) * packet.bytes * factor);
            maximum.set(index, loads[index]);
        }
        const double serial = route.overhead + packet.packet_bytes * route.serial;
        if (add) {
            overheads.insert(route.overhead);
            serials.insert(serial);
        } else {
            overheads.erase(overheads.find(route.overhead));
            serials.erase(serials.find(serial));
        }
    }
};

PyObject* optimize_round(PyObject*, PyObject* args) {
    Py_ssize_t resources;
    PyObject* input;
    int passes = 2;
    if (!PyArg_ParseTuple(args, "nO|i", &resources, &input, &passes)) return nullptr;
    if (resources < 0 || resources > 2000000 || passes < 0 || passes > 8) {
        PyErr_SetString(PyExc_ValueError, "round dimensions or passes outside bounds");
        return nullptr;
    }
    Owned rows(PySequence_Fast(input, "packets must be a sequence"));
    if (!rows.p) return nullptr;
    const auto n = PySequence_Fast_GET_SIZE(rows.p);
    if (n > 200000) {
        PyErr_SetString(PyExc_ValueError, "too many route packets"); return nullptr;
    }
    try {
        std::vector<Packet> packets;
        packets.reserve(n);
        size_t total_terms = 0;
        std::unordered_map<PyObject*, std::shared_ptr<std::vector<RouteOption>>> compiled;
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* row = PySequence_Fast_GET_ITEM(rows.p, i);
            PyObject* options;
            Packet packet;
            if (!PyArg_ParseTuple(row, "Odd", &options, &packet.bytes, &packet.packet_bytes)) return nullptr;
            if (!PyTuple_Check(options) || PyTuple_GET_SIZE(options) == 0 ||
                PyTuple_GET_SIZE(options) > 128 || !std::isfinite(packet.bytes) ||
                !std::isfinite(packet.packet_bytes) || packet.bytes <= 0 || packet.packet_bytes <= 0) {
                PyErr_SetString(PyExc_ValueError, "invalid route packet"); return nullptr;
            }
            auto found = compiled.find(options);
            if (found != compiled.end()) {
                packet.options = found->second;
            } else {
                packet.options = std::make_shared<std::vector<RouteOption>>();
                for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(options); ++j) {
                    PyObject* option = PyTuple_GET_ITEM(options, j);
                    PyObject *terms, *path;
                    RouteOption route;
                    if (!PyArg_ParseTuple(option, "OddO", &terms, &route.overhead, &route.serial, &path)) return nullptr;
                    if (!PyTuple_Check(terms) || !std::isfinite(route.overhead) || route.overhead < 0 ||
                        !std::isfinite(route.serial) || route.serial < 0) {
                        PyErr_SetString(PyExc_ValueError, "invalid route coefficients"); return nullptr;
                    }
                    std::set<size_t> seen;
                    for (Py_ssize_t k = 0; k < PyTuple_GET_SIZE(terms); ++k) {
                        Py_ssize_t index;
                        double factor;
                        if (!read_term(terms, k, resources, index, factor)) return nullptr;
                        if (!seen.insert(index).second) {
                            PyErr_SetString(PyExc_ValueError, "duplicate compiled resource index"); return nullptr;
                        }
                        route.terms.emplace_back(index, factor);
                    }
                    packet.options->push_back(std::move(route));
                }
                compiled.emplace(options, packet.options);
            }
            for (const auto& route : packet.routes()) {
                if (!std::isfinite(route.overhead + route.serial * packet.packet_bytes)) {
                    PyErr_SetString(PyExc_ValueError, "packet duration exceeds finite bounds"); return nullptr;
                }
                for (auto [index, factor] : route.terms) {
                    if (++total_terms > 16000000 || !std::isfinite(factor * packet.bytes)) {
                        PyErr_SetString(PyExc_ValueError, "route incidence or byte load exceeds bounds"); return nullptr;
                    }
                }
            }
            packets.push_back(std::move(packet));
        }
        auto released = std::make_unique<AllowThreads>();
        std::vector<size_t> selected(n, 0);
        RoundOptimizer state(resources);
        for (size_t i = 0; i < packets.size(); ++i) {
            auto& packet = packets[i];
            auto best = state.score(packet, packet.routes()[0]);
            for (size_t j = 1; j < packet.routes().size(); ++j) {
                auto score = state.score(packet, packet.routes()[j]);
                if (score < best) { best = score; selected[i] = j; }
            }
            state.change(packet, packet.routes()[selected[i]], true);
        }
        const auto original = selected;
        const double before = state.time();
        size_t moved = 0;
        for (int pass = 0; pass < passes; ++pass) {
            size_t changed = 0;
            for (size_t offset = 0; offset < packets.size(); ++offset) {
                size_t i = pass % 2 ? offset : packets.size() - 1 - offset;
                auto& packet = packets[i];
                if (packet.routes().size() == 1) continue;
                size_t best_index = selected[i];
                state.change(packet, packet.routes()[best_index], false);
                auto s = state.score(packet, packet.routes()[best_index]);
                // The exact duration is primary. Quadratic load breaks plateaus.
                std::array<double, 3> best{s[0], s[2], s[3]};
                for (size_t j = 0; j < packet.routes().size(); ++j) {
                    s = state.score(packet, packet.routes()[j]);
                    std::array<double, 3> score{s[0], s[2], s[3]};
                    if (score < best) { best = score; best_index = j; }
                }
                changed += best_index != selected[i];
                selected[i] = best_index;
                state.change(packet, packet.routes()[best_index], true);
            }
            moved += changed;
            if (!changed) break;
        }
        // Rebuild once to remove accumulated subtraction roundoff.
        RoundOptimizer final_state(resources);
        for (size_t i = 0; i < packets.size(); ++i)
            final_state.change(packets[i], packets[i].routes()[selected[i]], true);
        double after = final_state.time();
        if (after > before + std::max(1e-9, before * 1e-12)) {
            selected = original; after = before; moved = 0;
        }
        released.reset();
        Owned result(PyList_New(n));
        if (!result.p) return nullptr;
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* value = PyLong_FromSize_t(selected[i]);
            if (!value) return nullptr;
            PyList_SET_ITEM(result.p, i, value);
        }
        return Py_BuildValue("Oddn", result.p, before, after, static_cast<Py_ssize_t>(moved));
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
      catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what()); return nullptr;
    }
}
