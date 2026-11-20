// Price a complete rotation of each multicast lane. Every tour visits each
// access domain once. The same arcs operate concurrently in successive rounds.
namespace domain_cycles {

struct Path {
    std::vector<std::pair<int, double>> work;
    double serial;
};

struct Leg {
    int source, destination, path;
};

using Tour = std::vector<Leg>;

class Search {
    int domains, lanes;
    std::vector<double> load;
    std::vector<std::vector<double>> prices;
    std::vector<Tour> current;

    double peak() const {
        return *std::max_element(load.begin(), load.end());
    }

    double marginal(const Path& path, double scale) const {
        double value = .000003 * path.serial;
        for (auto [resource, unit] : path.work) {
            const double x = (load[resource] + .5 * unit) * scale;
            const double square = x * x;
            value += unit * (.0005 + square * square * square);
        }
        return value;
    }

    void add(const Path& path, int sign) {
        for (auto [resource, unit] : path.work) {
            load[resource] = std::max(0., load[resource] + sign * unit);
        }
    }

    int best_path(int source, int destination, double scale, double* price = nullptr) const {
        const auto& choices = paths[source * domains + destination];
        int best = 0;
        double minimum = marginal(choices[0], scale);
        for (size_t i = 1; i < choices.size(); ++i) {
            const double value = marginal(choices[i], scale);
            if (value < minimum) {
                best = static_cast<int>(i);
                minimum = value;
            }
        }
        if (price) *price = minimum;
        return best;
    }

    std::vector<int> order_from(int start) const {
        std::vector<int> order = {start};
        std::vector<bool> used(domains);
        used[start] = true;
        while (static_cast<int>(order.size()) < domains) {
            double minimum = std::numeric_limits<double>::infinity();
            int next = -1;
            for (int peer = 0; peer < domains; ++peer) {
                if (!used[peer] && (next < 0 || prices[order.back()][peer] < minimum)) {
                    minimum = prices[order.back()][peer];
                    next = peer;
                }
            }
            used[next] = true;
            order.push_back(next);
        }

        // A one-domain relocation changes only three directed arcs.
        for (int pass = 0; pass < 3; ++pass) {
            bool changed = false;
            for (int i = 0; i < domains; ++i) {
                int u = order[i], a = order[(i + domains - 1) % domains];
                int b = order[(i + 1) % domains], insert = -1;
                double best_gain = 1e-12;
                for (int j = 0; j < domains; ++j) {
                    int c = order[j], d = order[(j + 1) % domains];
                    if (c == u || d == u) continue;
                    double gain = prices[a][u] + prices[u][b] + prices[c][d]
                                    - prices[a][b] - prices[c][u] - prices[u][d];
                    if (gain > best_gain) {
                        best_gain = gain;
                        insert = j;
                    }
                }
                if (insert >= 0) {
                    order.erase(order.begin() + i);
                    order.insert(order.begin() + insert + 1 - (insert > i), u);
                    changed = true;
                }
            }
            if (!changed) break;
        }
        return order;
    }

    void replace_lane(int lane, int pass) {
        for (const auto& leg : current[lane]) {
            add(paths[leg.source * domains + leg.destination][leg.path], -1);
        }
        const double scale = 1 / std::max(1., peak());
        for (int u = 0; u < domains; ++u) {
            for (int v = 0; v < domains; ++v) {
                if (u != v) best_path(u, v, scale, &prices[u][v]);
            }
        }

        double minimum = std::numeric_limits<double>::infinity();
        std::vector<int> chosen;
        for (int i = 0; i < std::min(5, domains); ++i) {
            auto order = order_from((pass + lane + 7 * i) % domains);
            double value = 0.;
            for (int j = 0; j < domains; ++j) {
                value += prices[order[j]][order[(j + 1) % domains]];
            }
            if (chosen.empty() || value < minimum) {
                minimum = value;
                chosen = std::move(order);
            }
        }

        Tour replacement;
        for (int i = 0; i < domains; ++i) {
            int u = chosen[i], v = chosen[(i + 1) % domains];
            int route = best_path(u, v, scale);
            add(paths[u * domains + v][route], 1);
            replacement.push_back({u, v, route});
        }
        current[lane] = std::move(replacement);
    }

public:
    std::vector<std::vector<Path>> paths;
    std::vector<Tour> best;
    double best_peak = std::numeric_limits<double>::infinity();

    Search(int domains, int lanes, int resources) :
        domains(domains), lanes(lanes), load(resources),
        prices(domains, std::vector<double>(domains)), current(lanes), paths(domains * domains) {}

    void solve() {
        for (int pass = 0; pass < 12; ++pass) {
            for (int i = 0; i < lanes; ++i) replace_lane((i + pass * 7) % lanes, pass);
            const double scale = 1 / std::max(1., peak());
            for (auto& tour : current) {
                for (auto& leg : tour) {
                    auto& choices = paths[leg.source * domains + leg.destination];
                    add(choices[leg.path], -1);
                    leg.path = best_path(leg.source, leg.destination, scale);
                    add(choices[leg.path], 1);
                }
            }
            const double value = peak();
            if (best.empty() || value < best_peak) {
                best_peak = value;
                best = current;
            }
        }
    }
};

static PyObject* run(PyObject*, PyObject* args) {
    int domains, lanes, resources;
    double bytes;
    PyObject* entries;
    if (!PyArg_ParseTuple(args, "iiidO!", &domains, &lanes, &resources, &bytes,
                          &PyList_Type, &entries)) return nullptr;
    if (domains < 3 || domains > 32 || lanes < 1 || lanes > 32 || resources < 1 ||
        resources > 60000 || !std::isfinite(bytes) || !(bytes > 0) ||
        PyList_GET_SIZE(entries) != domains * domains) {
        PyErr_SetString(PyExc_ValueError, "unsupported multicast lane dimensions");
        return nullptr;
    }
    try {
        Search search(domains, lanes, resources);
        for (int i = 0; i < domains * domains; ++i) {
            if (i / domains == i % domains) continue;
            auto* choices = PyList_GET_ITEM(entries, i);
            if (!PyList_Check(choices) || PyList_GET_SIZE(choices) < 1 || PyList_GET_SIZE(choices) > 16)
                throw std::invalid_argument("missing or oversized domain transition");
            for (int j = 0; j < PyList_GET_SIZE(choices); ++j) {
                PyObject *terms, *nodes;
                double delay, serial;
                if (!PyArg_ParseTuple(PyList_GET_ITEM(choices, j), "O!ddO",
                                      &PyTuple_Type, &terms, &delay, &serial, &nodes)) return nullptr;
                Path path{{}, delay + bytes * serial};
                if (!std::isfinite(path.serial) || !(path.serial > 0) || !PyTuple_GET_SIZE(terms))
                    throw std::invalid_argument("invalid domain path cost");
                for (int k = 0; k < PyTuple_GET_SIZE(terms); ++k) {
                    Py_ssize_t resource;
                    double unit;
                    if (!read_term(terms, k, resources, resource, unit)) return nullptr;
                    if (!std::isfinite(unit * bytes)) throw std::invalid_argument("domain work overflow");
                    path.work.emplace_back(resource, bytes * unit);
                }
                search.paths[i].push_back(std::move(path));
            }
        }
        {
            AllowThreads release;
            search.solve();
        }
        Owned out(PyList_New(lanes));
        if (!out.p) return nullptr;
        for (int lane = 0; lane < lanes; ++lane) {
            auto* row = PyList_New(domains);
            if (!row) return nullptr;
            PyList_SET_ITEM(out.p, lane, row);
            for (int j = 0; j < domains; ++j) {
                const auto& leg = search.best[lane][j];
                auto* item = Py_BuildValue("(iii)", leg.source, leg.destination, leg.path);
                if (!item) return nullptr;
                PyList_SET_ITEM(row, j, item);
            }
        }
        return Py_BuildValue("dN", search.best_peak, out.release());
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}
}
