// Every complete reduction tree can carry any single same-sized output slot.
// Reassign one slot at a time, keeping the tree's complete causal order.
void pack_slots(const std::vector<MulticastTree>& trees, int rounds, int resources,
                std::vector<int>& owners, std::vector<double>& durations) {
    std::vector<TimingRow> rows;
    std::vector<double> latency(rounds), serial(rounds);
    for (int r = 0; r < rounds; ++r) rows.emplace_back(resources);
    auto change = [&](int owner, int sign) {
        for (const auto& term : trees[owner].terms) {
            auto& row = rows[term.round];
            double& load = row.loads[term.resource];
            load = std::max(0., load + sign * term.work);
            row.peak.set(term.resource, load);
        }
    };
    for (size_t i = 0; i < trees.size(); ++i) {
        owners[i] = static_cast<int>(i);
        change(owners[i], 1);
        for (int r = 0; r < rounds; ++r) {
            latency[r] = std::max(latency[r], trees[i].latency[r]);
            serial[r] = std::max(serial[r], trees[i].serial[r]);
        }
    }
    auto time_with = [&](int owner) {
        std::array<double, 64> busy{};
        for (int r = 0; r < rounds; ++r) busy[r] = rows[r].peak.maximum();
        for (const auto& term : trees[owner].terms)
            busy[term.round] = std::max(busy[term.round],
                rows[term.round].loads[term.resource] + term.work);
        double total = 0;
        for (int r = 0; r < rounds; ++r) total += std::max(serial[r], latency[r] + busy[r]);
        return total;
    };
    for (int pass = 0; pass < 3; ++pass) {
        bool changed = false;
        for (size_t i = 0; i < owners.size(); ++i) {
            int previous = owners[i];
            change(previous, -1);
            double best = time_with(previous);
            for (size_t j = 0; j < trees.size(); ++j) {
                double time = time_with(static_cast<int>(j));
                if (time < best - 1e-8) { best = time; owners[i] = static_cast<int>(j); }
            }
            change(owners[i], 1);
            changed |= owners[i] != previous;
        }
        if (!changed) break;
    }
    for (int r = 0; r < rounds; ++r) {
        double overhead = 0, path_time = 0;
        for (int owner : owners) {
            overhead = std::max(overhead, trees[owner].latency[r]);
            path_time = std::max(path_time, trees[owner].serial[r]);
        }
        durations[r] = std::max(path_time, overhead + rows[r].peak.maximum());
    }
}
