// Stagger whole source-shard trees. Model the drain of the final wave using
// fractional launch shares, then place complete trees under round barriers.
static void pace_trees(const std::vector<MulticastTree>& trees, int depth, int resources,
                       int lanes, int copies, std::vector<int>& best_offsets,
                       std::vector<double>& best_times) {
    const int rounds = lanes + depth - 1;
    const int owners = static_cast<int>(trees.size());
    std::vector<std::vector<double>> work(depth, std::vector<double>(resources));
    for (const auto& tree : trees) {
        for (const auto& term : tree.terms) {
            work[term.round][term.resource] += term.work * copies;
        }
    }

    std::vector<double> target(rounds), cumulative(lanes);
    double best_fractional = std::numeric_limits<double>::infinity();
    for (double taper : {.55, .68, .78, .89, 1.}) {
        std::vector<double> weight(lanes, 1.), peak(rounds);
        double sum = 1.;
        for (int launch = 1; launch < lanes; ++launch) {
            weight[launch] = weight[launch - 1] * taper;
            sum += weight[launch];
        }
        for (double& share : weight) share /= sum;
        double cost = 0.;
        for (int t = 0; t < rounds; ++t) {
            for (int resource = 0; resource < resources; ++resource) {
                double load = 0.;
                for (int age = std::max(0, t - lanes + 1); age <= std::min(depth - 1, t); ++age) {
                    load += work[age][resource] * weight[t - age];
                }
                peak[t] = std::max(peak[t], load);
            }
            cost += peak[t];
        }
        if (cost < best_fractional) {
            best_fractional = cost;
            target = peak;
            double share = 0.;
            for (int launch = 0; launch < lanes; ++launch) {
                share += weight[launch];
                cumulative[launch] = share;
            }
        }
    }

    std::vector<TimingRow> rows;
    for (int t = 0; t < rounds; ++t) rows.emplace_back(resources);
    std::vector<int> offsets(owners * copies);
    auto update = [&](int owner, int launch, int sign) {
        for (const auto& term : trees[owner].terms) {
            auto& row = rows[launch + term.round];
            double& load = row.loads[term.resource];
            load = std::max(0., load + sign * term.work);
            row.peak.set(term.resource, load);
        }
    };
    for (int owner = 0; owner < owners; ++owner) {
        const double shift = std::fmod(.5 + owner * .61803398875, 1.);
        for (int slot = 0; slot < copies; ++slot) {
            const double phase = (slot + shift) / copies;
            auto at = std::lower_bound(cumulative.begin(), cumulative.end(), phase);
            int launch = std::min(lanes - 1, static_cast<int>(at - cumulative.begin()));
            offsets[owner * copies + slot] = launch;
            update(owner, launch, 1);
        }
    }

    double best = std::numeric_limits<double>::infinity();
    auto save = [&] {
        std::vector<double> times(rounds), latency(rounds);
        for (int atom = 0; atom < static_cast<int>(offsets.size()); ++atom) {
            const auto& tree = trees[atom / copies];
            for (int age = 0; age < depth; ++age) {
                int t = offsets[atom] + age;
                latency[t] = std::max(latency[t], tree.latency[age]);
                times[t] = std::max(times[t], tree.serial[age]);
            }
        }
        double cost = 0.;
        for (int t = 0; t < rounds; ++t) {
            times[t] = std::max(times[t], latency[t] + rows[t].peak.maximum());
            cost += times[t];
        }
        if (cost < best) {
            best = cost;
            best_times = std::move(times);
            best_offsets = offsets;
        }
    };
    save();

    // Price each alternative against the loads with this tree removed. Break
    // ties on congestion across all touched resources, including shared pools.
    auto price = [&](const MulticastTree& tree, int launch) {
        std::array<double, 16> growth{};
        double energy = 0.;
        for (const auto& term : tree.terms) {
            const int t = launch + term.round;
            const auto& row = rows[t];
            const double next = row.loads[term.resource] + term.work;
            growth[term.round] = std::max(growth[term.round], next - row.peak.maximum());
            const double ratio = next / std::max(target[t], 1e-8);
            const double square = ratio * ratio;
            energy += term.work * square * square;
        }
        double delta = 0.;
        for (int age = 0; age < depth; ++age) delta += growth[age];
        return std::make_pair(std::round(delta * 1e6), energy);
    };
    for (int pass = 0; pass < 4; ++pass) {
        int moved = 0;
        for (int j = 0; j < static_cast<int>(offsets.size()); ++j) {
            const int atom = pass % 2 ? static_cast<int>(offsets.size()) - j - 1 : j;
            const int owner = atom / copies;
            const int previous = offsets[atom];
            update(owner, previous, -1);
            auto lowest = price(trees[owner], previous);
            for (int launch = 0; launch < lanes; ++launch) {
                if (launch == previous) continue;
                auto cost = price(trees[owner], launch);
                if (cost < lowest) {
                    lowest = cost;
                    offsets[atom] = launch;
                }
            }
            moved += previous != offsets[atom];
            update(owner, offsets[atom], 1);
        }
        save();
        if (!moved) break;
    }
}
