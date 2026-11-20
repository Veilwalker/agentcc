// Each slot grows monotonically. Compare against the saved round-start state
// when sending, and against all pending arrivals when receiving.
class ReductionState {
    using Mask = std::array<uint64_t, 8>;
    struct Expression { long slot; Mask mask; };
    const long split, words;
    PyObject *parser, *atoms;
    std::unordered_map<std::string_view,long> atom_names;
    size_t generation = 0;
    std::vector<size_t> modified;
    Mask full{};
    std::vector<Mask> previous, next;
    std::vector<size_t> last_transfer;
    std::unordered_map<PyObject*, Expression> cache;
    Owned names{PyDict_New()};
    size_t transfer = 0;

    // The built-in Replay supplies rank-major indices for exactly the original
    // atoms. Only those entries can grant a contributor bit to a term.
    bool parse_ascii(PyObject* token, Expression& out) {
        if (!atoms || !PyUnicode_IS_COMPACT_ASCII(token)) return false;
        const auto text = std::string_view(reinterpret_cast<const char*>(PyUnicode_1BYTE_DATA(token)),
                                           PyUnicode_GET_LENGTH(token));
        const auto separator = text.find('+');
        auto add = [&](long index) {
            const long owner = index / split, slot = index % split;
            const uint64_t bit = uint64_t(1) << (owner % 64);
            require(out.slot < 0 || out.slot == slot, "expression mixes reduction slots");
            require(!(out.mask[owner/64] & bit), "expression repeats a contributor");
            out.slot = slot;
            out.mask[owner/64] |= bit;
        };
        if (separator == text.npos) {
            if (text.size() >= 6 && text.front() == 'c' && text.substr(text.size()-4) == "_sum") {
                // Alternate decimal spellings accepted by Python keep the
                // reference parser, including leading zeroes and Unicode digits.
                long slot = 0;
                if (text.size() > 6 && text[1]=='0') return false;
                for (size_t i=1; i<text.size()-4; ++i) {
                    if (text[i]<'0' || text[i]>'9' || slot >= split) return false;
                    slot = slot*10 + (text[i]-'0');
                }
                if (slot >= split) return false;
                out.slot = slot;
                out.mask = full;
                return true;
            }
            auto value = PyDict_GetItemWithError(atoms,token);
            if (!value) return false;
            // Rank-major indices and their bounds were validated at entry.
            add(PyLong_AsLong(value));
            return true;
        }
        if (atom_names.empty()) {
            Py_ssize_t pos = 0;
            PyObject *name, *index;
            atom_names.reserve(PyDict_Size(atoms));
            while (PyDict_Next(atoms,&pos,&name,&index)) {
                if (PyUnicode_IS_COMPACT_ASCII(name)) {
                    auto bytes = reinterpret_cast<const char*>(PyUnicode_1BYTE_DATA(name));
                    atom_names.emplace(std::string_view(bytes,PyUnicode_GET_LENGTH(name)),PyLong_AsLong(index));
                }
            }
        }
        size_t start = 0, end = separator;
        while (true) {
            const auto found = atom_names.find(text.substr(start,end-start));
            if (found == atom_names.end()) return false;
            add(found->second);
            if (end == text.npos) return true;
            start = end + 1;
            end = text.find('+',start);
        }
    }

    const Expression& expression(PyObject* token) {
        auto found = cache.find(token);
        if (found != cache.end()) return found->second;
        // Equal strings in a parsed JSON response need not share identity.
        auto canonical = PyDict_GetItemWithError(names.p, token);
        if (canonical) return cache.at(canonical);
        Expression result{-1,{}};
        if (!parse_ascii(token,result)) {
            result = Expression{};
            Owned parsed(PyObject_CallFunctionObjArgs(parser, token, nullptr));
            require(parsed.p && PyTuple_Check(parsed.p) && PyTuple_GET_SIZE(parsed.p) == 2,
                    "invalid reduction expression");
            result.slot = PyLong_AsLong(PyTuple_GET_ITEM(parsed.p, 0));
            require(result.slot >= 0 && result.slot < split, "reduction slot out of range");
            auto bits = PyTuple_GET_ITEM(parsed.p, 1);
            Owned bytes(PyObject_CallMethod(bits, "to_bytes", "ls", words * 8, "little"));
            require(bytes.p && PyBytes_Check(bytes.p) && PyBytes_GET_SIZE(bytes.p) == words*8,
                    "invalid contributor mask");
            const auto* data = reinterpret_cast<const unsigned char*>(PyBytes_AS_STRING(bytes.p));
            for (long w=0; w<words; ++w)
                for (int b=0; b<8; ++b)
                    result.mask[w] |= uint64_t(data[8*w+b]) << (8*b);
        }
        require(PyDict_SetItem(names.p, token, token) == 0, "cannot store reduction expression");
        return cache.emplace(token, result).first->second;
    }
public:
    ReductionState(long ranks, long slots, PyObject* parse, PyObject* canonical_atoms) :
        split(slots), words((ranks+63)/64), parser(parse), atoms(canonical_atoms), modified(ranks*slots),
        previous(ranks*slots), next(ranks*slots), last_transfer(slots) {
        require(names.p && words <= 8, "unsupported reduction state");
        for (long r=0; r<ranks; ++r) {
            const auto bit = uint64_t(1) << (r%64);
            full[r/64] |= bit;
            for (long s=0; s<slots; ++s) next[r*slots+s][r/64] = bit;
        }
    }
    void begin_round() { ++generation; }
    const Mask& at_start(size_t entry) const {
        return modified[entry] == generation ? previous[entry] : next[entry];
    }
    void receive(long src, long dst, PyObject* payload, bool copy) {
        ++transfer;
        for (Py_ssize_t k=0; k<PyList_GET_SIZE(payload); ++k) {
            auto token = PyList_GET_ITEM(payload,k);
            require(PyUnicode_Check(token), "reduction tokens must be strings");
            const auto& expr = expression(token);
            auto slot = expr.slot;
            require(last_transfer[slot] != transfer, "transfer repeats a reduction slot");
            last_transfer[slot] = transfer;
            require(at_start(src*split+slot) == expr.mask, "sender contribution mismatch");
            const size_t dest = dst*split+slot;
            auto& held = next[dest];
            if (modified[dest] != generation) {
                previous[dest] = held;
                modified[dest] = generation;
            }
            if (copy) {
                require(expr.mask == full && held != full && held == previous[dst*split+slot],
                        "incomplete or conflicting reduction copy");
                held = full;
            } else {
                // A duplicate transmission also overlaps its earlier arrival.
                for (long w=0; w<words; ++w) {
                    require(!(held[w] & expr.mask[w]), "overlapping reduction contributions");
                    held[w] |= expr.mask[w];
                }
            }
        }
    }
    bool complete() const {
        return std::all_of(next.begin(), next.end(), [&](const Mask& v) { return v == full; });
    }
};
