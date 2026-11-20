// C++ capacity kernels with the same arithmetic and tie order as Python.
// Python owns graph/path objects; native state owns only numerical round loads.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <set>
#include <map>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <limits>
#include <queue>
#include <tuple>
#include <string_view>

namespace {
struct Owned {
    PyObject* p;
    explicit Owned(PyObject* value) : p(value) {}
    ~Owned() { Py_XDECREF(p); }
    Owned(const Owned&) = delete;
    PyObject* release() { PyObject* value = p; p = nullptr; return value; }
};

// Equivalent to CapacityCosts.compile, including resource multiplicity and order.
PyObject* compile_path(PyObject*, PyObject* args) {
    PyObject *path, *edges, *edge_resources, *resources, *indices;
    double startup, switching;
    if (!PyArg_ParseTuple(args, "OOOOOdd", &path, &edges, &edge_resources,
                          &resources, &indices, &startup, &switching)) return nullptr;
    if (!PyTuple_Check(path) || PyTuple_GET_SIZE(path) < 2 || !PyDict_Check(edges)
        || !PyDict_Check(edge_resources) || !PyDict_Check(resources)
        || !PyDict_Check(indices) || !std::isfinite(startup) || startup < 0
        || !std::isfinite(switching) || switching < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid capacity graph or path");
        return nullptr;
    }
    Owned terms(PyList_New(0)), pools(PyDict_New()), edge_tag(PyUnicode_FromString("edge")),
          pool_tag(PyUnicode_FromString("pool"));
    if (!terms.p || !pools.p || !edge_tag.p || !pool_tag.p) return nullptr;
    double serial = 0.0;
    const auto count = PyTuple_GET_SIZE(path);
    for (Py_ssize_t i = 0; i + 1 < count; ++i) {
        Owned edge(PyTuple_Pack(2, PyTuple_GET_ITEM(path, i), PyTuple_GET_ITEM(path, i + 1)));
        if (!edge.p) return nullptr;
        PyObject* capacity = PyDict_GetItemWithError(edges, edge.p);
        if (!capacity) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, edge.p); return nullptr; }
        const double bandwidth = PyFloat_AsDouble(capacity);
        if (PyErr_Occurred()) return nullptr;
        if (!std::isfinite(bandwidth) || bandwidth <= 0) {
            PyErr_SetString(PyExc_ValueError, "capacity must be positive and finite"); return nullptr;
        }
        double inverse = 1e6 / bandwidth;
        serial += inverse;
        Owned key(PyTuple_Pack(2, edge_tag.p, edge.p));
        if (!key.p) return nullptr;
        PyObject* index = PyDict_GetItemWithError(indices, key.p);
        if (!index) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, key.p); return nullptr; }
        Owned term(Py_BuildValue("Od", index, inverse));
        if (!term.p || PyList_Append(terms.p, term.p) < 0) return nullptr;
        PyObject* bound = PyDict_GetItemWithError(edge_resources, edge.p);
        if (!bound) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, edge.p); return nullptr; }
        Owned members(PySequence_Fast(bound, "resource members must be a sequence"));
        if (!members.p) return nullptr;
        for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(members.p); ++j) {
            PyObject* resource = PySequence_Fast_GET_ITEM(members.p, j);
            PyObject* old = PyDict_GetItemWithError(pools.p, resource);
            if (!old && PyErr_Occurred()) return nullptr;
            long multiple = old ? PyLong_AsLong(old) : 0;
            if (PyErr_Occurred()) return nullptr;
            Owned value(PyLong_FromLong(multiple + 1));
            if (!value.p || PyDict_SetItem(pools.p, resource, value.p) < 0) return nullptr;
        }
    }
    Py_ssize_t position = 0;
    PyObject *name, *multiple;
    while (PyDict_Next(pools.p, &position, &name, &multiple)) {
        PyObject* capacity = PyDict_GetItemWithError(resources, name);
        if (!capacity) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, name); return nullptr; }
        const double bandwidth = PyFloat_AsDouble(capacity);
        if (PyErr_Occurred()) return nullptr;
        if (!std::isfinite(bandwidth) || bandwidth <= 0) {
            PyErr_SetString(PyExc_ValueError, "capacity must be positive and finite"); return nullptr;
        }
        double factor = PyLong_AsLong(multiple) * 1e6 / bandwidth;
        if (PyErr_Occurred()) return nullptr;
        Owned key(PyTuple_Pack(2, pool_tag.p, name));
        if (!key.p) return nullptr;
        PyObject* index = PyDict_GetItemWithError(indices, key.p);
        if (!index) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, key.p); return nullptr; }
        Owned term(Py_BuildValue("Od", index, factor));
        if (!term.p || PyList_Append(terms.p, term.p) < 0) return nullptr;
    }
    Owned tuple(PyList_AsTuple(terms.p));
    if (!tuple.p) return nullptr;
    return Py_BuildValue("OddO", tuple.p, startup + (count - 2) * switching, serial, path);
}

struct Assignment {
    explicit Assignment(size_t n) : loads(n, 0.0) {}
    std::vector<double> loads;
    double busy = 0.0, overhead = 0.0, serial = 0.0;
};
constexpr const char* capsule_name = "agentcc.capacity.Assignment.v1";
void release_assignment(PyObject* capsule) {
    auto* state = static_cast<Assignment*>(PyCapsule_GetPointer(capsule, capsule_name));
    delete state;
}
PyObject* make_assignment(PyObject*, PyObject* arg) {
    Py_ssize_t n = PyLong_AsSsize_t(arg);
    if (n < 0 || PyErr_Occurred()) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "negative resource count");
        return nullptr;
    }
    try {
        auto state = std::make_unique<Assignment>(static_cast<size_t>(n));
        PyObject* result = PyCapsule_New(state.get(), capsule_name, release_assignment);
        if (result) state.release();
        return result;
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
      catch (const std::length_error&) {
        PyErr_SetString(PyExc_OverflowError, "resource count exceeds vector capacity");
        return nullptr;
    }
}

bool read_term(PyObject* terms, Py_ssize_t i, size_t n, Py_ssize_t& index, double& factor) {
    PyObject* term = PyTuple_GET_ITEM(terms, i);
    if (!PyTuple_Check(term) || PyTuple_GET_SIZE(term) != 2) {
        PyErr_SetString(PyExc_ValueError, "invalid compiled term"); return false;
    }
    index = PyLong_AsSsize_t(PyTuple_GET_ITEM(term, 0));
    factor = PyFloat_AsDouble(PyTuple_GET_ITEM(term, 1));
    if (PyErr_Occurred()) return false;
    if (index < 0 || static_cast<size_t>(index) >= n || !std::isfinite(factor) || factor <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid resource index or factor"); return false;
    }
    return true;
}

// Exact same option iteration and lexicographic score as the Python version.
PyObject* choose(PyObject*, PyObject* args) {
    PyObject *capsule, *options;
    double count, packet_bytes;
    if (!PyArg_ParseTuple(args, "OOdd", &capsule, &options, &count, &packet_bytes)) return nullptr;
    auto* state = static_cast<Assignment*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!state) return nullptr;
    if (!PyTuple_Check(options) || !std::isfinite(count) || !std::isfinite(packet_bytes)
        || count <= 0 || packet_bytes <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid assignment input"); return nullptr;
    }
    std::array<double, 4> best{};
    PyObject* selected = nullptr;
    for (Py_ssize_t o = 0; o < PyTuple_GET_SIZE(options); ++o) {
        PyObject* option = PyTuple_GET_ITEM(options, o);
        if (!PyTuple_Check(option) || PyTuple_GET_SIZE(option) != 4) {
            PyErr_SetString(PyExc_ValueError, "invalid compiled path"); return nullptr;
        }
        PyObject* terms = PyTuple_GET_ITEM(option, 0);
        if (!PyTuple_Check(terms)) { PyErr_SetString(PyExc_ValueError, "terms must be a tuple"); return nullptr; }
        const double overhead = PyFloat_AsDouble(PyTuple_GET_ITEM(option, 1));
        const double serial = PyFloat_AsDouble(PyTuple_GET_ITEM(option, 2));
        if (PyErr_Occurred()) return nullptr;
        if (!std::isfinite(overhead) || overhead < 0 || !std::isfinite(serial) || serial < 0) {
            PyErr_SetString(PyExc_ValueError, "invalid path time coefficients"); return nullptr;
        }
        double busy = 0.0, energy = 0.0;
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(terms); ++i) {
            Py_ssize_t index; double factor;
            if (!read_term(terms, i, state->loads.size(), index, factor)) return nullptr;
            const double before = state->loads[index];
            const double after = before + count * factor;
            if (after > busy) busy = after;
            energy += after * after - before * before;
        }
        const double path_time = overhead + packet_bytes * serial;
        const std::array<double, 4> score{
            std::max({state->serial, path_time,
                std::max(state->overhead, overhead) + std::max(state->busy, busy)}),
            busy, energy, path_time};
        if (!selected || score < best) { selected = option; best = score; }
    }
    if (!selected) { PyErr_SetString(PyExc_ValueError, "no legal route for this transfer"); return nullptr; }
    PyObject* terms = PyTuple_GET_ITEM(selected, 0);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(terms); ++i) {
        Py_ssize_t index; double factor;
        if (!read_term(terms, i, state->loads.size(), index, factor)) return nullptr;
        state->loads[index] += count * factor;
        if (state->loads[index] > state->busy) state->busy = state->loads[index];
    }
    const double overhead = PyFloat_AsDouble(PyTuple_GET_ITEM(selected, 1));
    const double serial = PyFloat_AsDouble(PyTuple_GET_ITEM(selected, 2));
    state->overhead = std::max(state->overhead, overhead);
    state->serial = std::max(state->serial, overhead + packet_bytes * serial);
    PyObject* path = PyTuple_GET_ITEM(selected, 3);
    Py_INCREF(path);
    return path;
}

PyObject* assignment_time(PyObject*, PyObject* capsule) {
    auto* state = static_cast<Assignment*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!state) return nullptr;
    return PyFloat_FromDouble(std::max(state->serial, state->overhead + state->busy));
}

#include "round_optimizer.h"
#include "wave_costs.h"
#include "dependency_schedule.h"
#include "round_slack.h"
#include "tree_offsets.h"
#include "shortest_paths.h"
#include "lazy_paths.h"
#include "trace_profile.h"
#include "path_compiler.h"
#include "aggregate_packets.h"
#include "gather_emit.h"
#include "locality.h"
#include "gather_replay.h"
#include "congestion_routes.h"
#include "domain_cycles.h"
#include "local_tail.h"
#include "relay_exchange.h"

PyMethodDef methods[] = {
    {"exchange_relays", flow_routes::exchange, METH_VARARGS, "Search physical placements of reduction subtrees."},
    {"adaptive_routes", flow_routes::run, METH_VARARGS, "Generate paths from current congestion prices."},
    {"verify_canonical_reduce", gather_replay::canonical_reduce, METH_VARARGS, "Verify reduction with indexed original atoms."},
    {"verify_reduce", gather_replay::run, METH_VARARGS, "Verify actual reduction transfers and round-start ownership."},
    {"verify_gather", gather_replay::run, METH_VARARGS, "Independent verification of actual AllGather transfers."},
    {"locality_permutation", locality_permutation, METH_VARARGS, "Build a balanced rank hierarchy."},
    {"make_capacity_graph", make_capacity_graph, METH_VARARGS, "Compile request-local edge and pool coefficients."},
    {"emit_gather_shards",gather_emit::materialize,METH_VARARGS,"Expand whole shard launches from a declared tree."},
    {"aggregate_gather_packets",aggregate_packets::run,METH_VARARGS,"Reassign whole atoms within a certified gather round."},
    {"compile_cached_path", compile_cached_path, METH_VARARGS, "Compile a path using request-local numeric coefficients."},
    {"compile_adjacency", compile_adjacency, METH_O, "Compile a request-local routing graph."},
    {"trace_profile",trace_profile::price,METH_VARARGS,"Price all rows of a legal baseline."},
    {"compile_lazy_graph",lazy_paths::compile,METH_O,"Compile incoming and outgoing edges for exact lazy routing."},
    {"lazy_tree",lazy_paths::build,METH_VARARGS,"Compute primary distances for one rank."},
    {"lazy_path",lazy_paths::path,METH_VARARGS,"Resolve the requested path and any required priority ties."},
    {"shortest_tree", shortest_tree, METH_VARARGS, "Compute a switch-only shortest-path tree."},
    {"make_route_cache", make_route_cache, METH_O, "Request-local immutable route incidence."},
    {"optimize_groups", optimize_groups, METH_VARARGS, "Price striped movements as one numeric batch."},
    {"optimize_round", optimize_round, METH_VARARGS, "Balance legal routes across a complete round."},
    {"fill_slack", round_slack::run, METH_VARARGS, "Measure round headroom and advance ready copies."},
    {"retime_transfers", retime_transfers, METH_VARARGS, "Reschedule transfers inside causal windows."},
    {"choose_slot_trees", choose_slot_trees, METH_VARARGS, "Pack equal reduction slots onto complete legal trees."},
    {"local_tail", local_tail::run, METH_VARARGS, "Rebuild local gather deliveries after inter-domain injections."},
    {"domain_cycles", domain_cycles::run, METH_VARARGS, "Allocate rotating multicast lanes across physical domains."},
    {"paced_launch", paced_launch, METH_VARARGS, "Choose integer launch times for replicated multicast trees."},
    {"choose_tree_offsets", choose_tree_offsets, METH_VARARGS, "Round fractional wave shares onto complete trees."},
    {"pipeline_costs", pipeline_costs, METH_VARARGS, "Price overlapping stages across split factors."},
    {"compile_path", compile_path, METH_VARARGS, "Compile path resource incidence."},
    {"make_assignment", make_assignment, METH_O, "Allocate numerical round loads."},
    {"choose", choose, METH_VARARGS, "Choose and insert a path using unchanged score."},
    {"assignment_time", assignment_time, METH_O, "Current round duration."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_capacity_native", nullptr, -1, methods};
}
PyMODINIT_FUNC PyInit__capacity_native() {
    PyObject* result = PyModule_Create(&module);
    if (!result) return nullptr;
    if (PyModule_AddIntConstant(result, "API_VERSION", 2) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
