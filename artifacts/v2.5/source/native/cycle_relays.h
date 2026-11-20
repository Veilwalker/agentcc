// Price one new multicast delivery per lane and receiving domain per wave.
// Both its parent and its token may change. Reads use the previous boundary;
// all receipts become available together at the next boundary.
namespace cycle_relays {
struct Path {
    std::vector<std::pair<int, double>> work;
    double latency = 0., serial = 0.;
    void append(const Path& other) {
        serial += other.serial;
        for (auto term : other.work) {
            auto old =
                std::find_if(work.begin(), work.end(), [&](auto x) { return x.first == term.first; });
            if (old == work.end()) {
                work.push_back(term);
            } else {
                old->second += term.second;
            }
        }
    }
};
struct Leg {
    int src, dst, path;
};
struct Choice {
    int src, index;
    Path path;
};
struct Delivery {
    int parent, route_index, owner;
};

static Path parse(PyObject* value, int resources, double bytes) {
    PyObject *terms, *physical;
    Path path;
    if (!PyArg_ParseTuple(value, "O!ddO", &PyTuple_Type, &terms, &path.latency, &path.serial,
                          &physical)) {
        throw std::invalid_argument("invalid multicast path");
    }
    path.serial *= bytes;
    if (!(path.serial > 0) || !std::isfinite(path.serial) || !std::isfinite(path.latency) ||
        path.latency < 0 || PyTuple_GET_SIZE(terms) > 128) {
        throw std::invalid_argument("invalid multicast cost");
    }
    for (Py_ssize_t k = 0; k < PyTuple_GET_SIZE(terms); ++k) {
        Py_ssize_t index;
        double factor;
        if (!read_term(terms, k, resources, index, factor) || !std::isfinite(factor * bytes)) {
            throw std::invalid_argument("invalid multicast resource use");
        }
        path.work.emplace_back(index, factor * bytes);
    }
    return path;
}

static PyObject* run(PyObject*, PyObject* args) {
    int resources;
    double bytes, switch_delay, local_delay, local_serial;
    PyObject *paths, *endpoints, *tours, *loads, *wave_specs;
    if (!PyArg_ParseTuple(args, "iddddO!O!O!O!O!", &resources, &bytes, &switch_delay, &local_delay,
                          &local_serial, &PyList_Type, &paths, &PyList_Type, &endpoints, &PyList_Type,
                          &tours, &PyList_Type, &loads, &PyList_Type, &wave_specs)) {
        return nullptr;
    }
    int n = PyList_GET_SIZE(wave_specs), lanes = PyList_GET_SIZE(tours);
    if (n < 3 || n > 24 || lanes < 1 || lanes > 32 || resources < 1 || resources > 60000 ||
        PyList_GET_SIZE(paths) != n * n || PyList_GET_SIZE(endpoints) != 2 * n * lanes ||
        PyList_GET_SIZE(loads) > resources || !std::isfinite(bytes) || bytes <= 0 ||
        !std::isfinite(switch_delay) || switch_delay < 0 || !std::isfinite(local_delay) ||
        local_delay < 0 || !std::isfinite(local_serial) || local_serial <= 0) {
        Py_RETURN_NONE;
    }
    try {
        std::vector<std::vector<Path>> cores(n * n);
        std::vector<std::pair<int, double>> local;
        std::vector<std::pair<int, int>> window;
        int option_count = 0;
        for (int pair = 0; pair < n * n; ++pair) {
            PyObject* options = PyList_GET_ITEM(paths, pair);
            if (!PyList_Check(options) || PyList_GET_SIZE(options) > 16 ||
                (pair / n != pair % n && !PyList_GET_SIZE(options))) {
                throw std::invalid_argument("invalid multicast alternatives");
            }
            for (int k = 0; k < PyList_GET_SIZE(options); ++k) {
                cores[pair].push_back(parse(PyList_GET_ITEM(options, k), resources, bytes));
            }
            option_count += cores[pair].size();
        }
        // Bound the expanded per-lane route table and the per-wave repricing work.
        if (1LL * option_count * lanes > 90000 || 1LL * option_count * lanes * n * n > 24000000) {
            Py_RETURN_NONE;
        }
        std::vector<Path> endpoint;
        for (int i = 0; i < 2 * n * lanes; ++i) {
            endpoint.push_back(parse(PyList_GET_ITEM(endpoints, i), resources, bytes));
        }
        for (int j = 0; j < PyList_GET_SIZE(loads); ++j) {
            int k;
            double work;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(loads, j), "id", &k, &work) || k < 0 ||
                k >= resources || work < 0 || !std::isfinite(work)) {
                throw std::invalid_argument("invalid local multicast resource");
            }
            local.emplace_back(k, work);
        }
        for (int t = 0; t < n; ++t) {
            int count, packet;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(wave_specs, t), "ii", &count, &packet) || count < 0 ||
                count > n * lanes || packet < 0 || packet > n) {
                throw std::invalid_argument("invalid local multicast packing");
            }
            window.emplace_back(count, packet);
        }
        std::vector<std::vector<Leg>> order(lanes, std::vector<Leg>(n));
        for (int l = 0; l < lanes; ++l) {
            PyObject* tour = PyList_GET_ITEM(tours, l);
            if (!PyList_Check(tour) || PyList_GET_SIZE(tour) != n) {
                throw std::invalid_argument("invalid multicast tour");
            }
            std::vector<bool> visited(n);
            for (int j = 0; j < n; ++j) {
                Leg& m = order[l][j];
                if (!PyArg_ParseTuple(PyList_GET_ITEM(tour, j), "iii", &m.src, &m.dst, &m.path) ||
                    m.src < 0 || m.src >= n || m.dst < 0 || m.dst >= n || m.path < 0 ||
                    m.path >= int(cores[m.src * n + m.dst].size()) || visited[m.src]) {
                    throw std::invalid_argument("invalid multicast tour arc");
                }
                visited[m.src] = true;
            }
            for (int j = 0; j < n; ++j) {
                if (order[l][j].dst != order[l][(j + 1) % n].src) {
                    throw std::invalid_argument("noncyclic multicast tour");
                }
            }
        }
        std::vector<std::vector<Choice>> options(lanes);
        std::vector<std::vector<int>> first(lanes, std::vector<int>(n * n));
        std::vector<std::vector<Delivery>> result(n - 1);
        double total = 0;
        {
            AllowThreads release;
            // Prefix/suffix rank edges and core edges may use the same declared
            // pool. Merge duplicate terms before evaluating its marginal load.
            for (int l = 0; l < lanes; ++l) {
                for (int pair = 0; pair < n * n; ++pair) {
                    first[l][pair] = options[l].size();
                    for (int k = 0; k < int(cores[pair].size()); ++k) {
                        Path path = cores[pair][k];
                        path.latency += 2 * switch_delay;
                        path.append(endpoint[2 * ((pair / n) * lanes + l)]);
                        path.append(endpoint[2 * ((pair % n) * lanes + l) + 1]);
                        options[l].push_back({pair / n, k, std::move(path)});
                    }
                }
            }
            const int count = n * lanes;
            // Each access domain receives one fresh owner on this lane per wave.
            std::vector<std::vector<unsigned>> held(lanes, std::vector<unsigned>(n));
            for (auto& lane : held) {
                for (int d = 0; d < n; ++d) {
                    lane[d] = 1u << d;
                }
            }
            for (int t = 0; t < n - 1; ++t) {
                std::vector<double> load(resources);
                for (auto [k, v] : local) {
                    load[k] = v * window[t].first;
                }
                std::vector<int> routes(count);
                auto add = [&](int l, int q, int sign) {
                    for (auto [k, v] : options[l][q].path.work) {
                        load[k] += v * sign;
                    }
                };
                for (int i = 0; i < count; ++i) {
                    auto m = order[i / n][i % n];
                    routes[i] = first[i / n][m.src * n + m.dst] + m.path;
                    if (!(held[i / n][m.src] & ~held[i / n][m.dst])) {
                        for (int parent = 0; parent < n; ++parent) {
                            if (held[i / n][parent] & ~held[i / n][m.dst]) {
                                routes[i] = first[i / n][parent * n + m.dst];
                                break;
                            }
                        }
                    }
                    add(i / n, routes[i], 1);
                }
                auto duration = [&]() {
                    double delay = window[t].first ? local_delay : 0.;
                    double serial =
                        window[t].first ? local_delay + window[t].second * local_serial : 0.;
                    for (int i = 0; i < count; ++i) {
                        auto& p = options[i / n][routes[i]].path;
                        delay = std::max(delay, p.latency);
                        serial = std::max(serial, p.latency + p.serial);
                    }
                    return std::max(serial, *std::max_element(load.begin(), load.end()) + delay);
                };
                // Start with a legal parent for every receiver. Keep the best
                // complete wave while changing one route at a time.
                double best = duration();
                auto chosen = routes;
                for (int pass = 0; pass < 4; ++pass) {
                    double scale = 1 / std::max(1., best);
                    for (int i = 0; i < count; ++i) {
                        // 137 is prime and exceeds either dimension, so the
                        // stride visits every lane/domain pair exactly once.
                        const int position = (i * 137 + pass * 29) % count;
                        const int lane = position / n;
                        const int destination = order[lane][position % n].dst;
                        const int old = routes[position];
                        int winner = old;
                        add(lane, old, -1);
                        double minimum = std::numeric_limits<double>::infinity();
                        for (int parent = 0; parent < n; ++parent) {
                            if (!(held[lane][parent] & ~held[lane][destination])) {
                                continue;
                            }
                            int pair = parent * n + destination;
                            for (int k = 0; k < int(cores[pair].size()); ++k) {
                                int q = first[lane][pair] + k;
                                auto& p = options[lane][q].path;
                                double peak = (p.serial + p.latency) * scale, energy = 0.;
                                for (auto [resource, work] : p.work) {
                                    double increment = work * scale,
                                           after = load[resource] * scale + increment,
                                           mid = after - .5 * increment, sq = mid * mid;
                                    // Higher powers increasingly emphasize the loaded
                                    // links and pools in subsequent passes.
                                    double fourth = sq * sq, eighth = fourth * fourth,
                                           price = mid * sq * fourth;
                                    if (pass >= 1) {
                                        price *= eighth;
                                    }
                                    if (pass >= 3) {
                                        price *= eighth * eighth;
                                    }
                                    peak = std::max(peak, after);
                                    energy += increment * (.0000003 + price);
                                }
                                double score = energy + .2 * std::max(0., peak - .997);
                                if (score < minimum - 1e-12 ||
                                    (std::abs(score - minimum) < 1e-12 && q == old)) {
                                    minimum = score;
                                    winner = q;
                                }
                            }
                        }
                        routes[position] = winner;
                        add(lane, winner, 1);
                    }
                    double now = duration();
                    if (now < best - 1e-7) {
                        best = now;
                        chosen = routes;
                    }
                }
                // Tokens are interchangeable for route pricing. Choose a
                // relatively scarce eligible token after fixing the route,
                // without reading any receipt generated in this wave.
                auto next = held;
                std::vector<std::vector<int>> copies(lanes, std::vector<int>(n));
                for (int l = 0; l < lanes; ++l) {
                    for (unsigned mask : held[l]) {
                        for (int owner = 0; owner < n; ++owner) {
                            copies[l][owner] += (mask >> owner) & 1u;
                        }
                    }
                }
                result[t].resize(count);
                for (int offset = 0; offset < count; ++offset) {
                    int i = (offset * 137 + t * 29) % count, l = i / n, j = i % n,
                        dst = order[l][j].dst;
                    const auto& q = options[l][chosen[i]];
                    unsigned available = held[l][q.src] & ~held[l][dst];
                    int owner = -1;
                    for (int k = 0; k < n; ++k) {
                        int candidate = (k + t * 5 + l * 3 + j * 7) % n;
                        if ((available >> candidate & 1u) &&
                            (owner < 0 || copies[l][candidate] < copies[l][owner])) {
                            owner = candidate;
                        }
                    }
                    if (owner < 0) {
                        throw std::invalid_argument("unavailable multicast parent");
                    }
                    ++copies[l][owner];
                    next[l][dst] |= 1u << owner;
                    result[t][i] = {q.src, q.index, owner};
                }
                held.swap(next);
                total += best;
            }
            double tail = local_delay + window.back().second * local_serial;
            for (auto [k, v] : local) {
                tail = std::max(tail, local_delay + window.back().first * v);
            }
            total += tail;
        }
        Owned out(PyList_New(n - 1));
        if (!out.p) {
            return nullptr;
        }
        for (int t = 0; t < n - 1; ++t) {
            Owned row(PyList_New(result[t].size()));
            if (!row.p) {
                return nullptr;
            }
            for (int i = 0; i < int(result[t].size()); ++i) {
                auto delivery = result[t][i];
                auto* item =
                    Py_BuildValue("(iii)", delivery.parent, delivery.route_index, delivery.owner);
                if (!item) {
                    return nullptr;
                }
                PyList_SET_ITEM(row.p, i, item);
            }
            PyList_SET_ITEM(out.p, t, row.release());
        }
        return Py_BuildValue("dN", total, out.release());
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}
} // namespace cycle_relays
