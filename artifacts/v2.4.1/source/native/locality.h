// Balanced matching over bidirectional affinities. Keep the Python tie order.
PyObject* locality_permutation(PyObject*, PyObject* args) {
    int n;
    PyObject *direct, *ports;
    if (!PyArg_ParseTuple(args, "iO!O!", &n, &PyList_Type, &direct, &PyList_Type, &ports))
        return nullptr;
    if (n < 2 || n > 512 || (n & (n - 1))) {
        PyErr_SetString(PyExc_ValueError, "invalid hierarchy size");
        return nullptr;
    }
    std::vector<double> affinity(n*n, 0);
    auto valid = [n](int i, double c) { return i >= 0 && i < n && c > 0 && std::isfinite(c); };
    for (Py_ssize_t k=0; k<PyList_GET_SIZE(direct); ++k) {
        int i,j;
        double c;
        if (!PyArg_ParseTuple(PyList_GET_ITEM(direct,k),"iid",&i,&j,&c)) return nullptr;
        if (!valid(i,c) || !valid(j,c)) {
            PyErr_SetString(PyExc_ValueError,"invalid direct affinity");
            return nullptr;
        }
        affinity[i*n+j] += c;
    }
    for (Py_ssize_t k=0; k<PyList_GET_SIZE(ports); ++k) {
        auto *items=PyList_GET_ITEM(ports,k);
        if (!PyList_Check(items)) {
            PyErr_SetString(PyExc_TypeError,"port member list required");
            return nullptr;
        }
        std::vector<std::pair<int,double>> members;
        for (Py_ssize_t t=0; t<PyList_GET_SIZE(items); ++t) {
            int i;
            double c;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(items,t),"id",&i,&c)) return nullptr;
            if (!valid(i,c)) { PyErr_SetString(PyExc_ValueError,"invalid port affinity"); return nullptr; }
            members.emplace_back(i,c);
        }
        for (size_t a=0; a<members.size(); ++a) {
            auto [i,x]=members[a];
            for (size_t b=a+1; b<members.size(); ++b) {
                auto [j,y]=members[b];
                double v=x*y/(x+y)/(members.size()-1);
                affinity[i*n+j]+=v;
                affinity[j*n+i]+=v;
            }
        }
    }
    std::vector<std::vector<int>> groups(n);
    for (int i=0; i<n; ++i) groups[i].push_back(i);
    for (int size=n; size>1; size/=2) {
        std::vector<std::tuple<double,int,int>> choices;
        for (int i=0; i<size; ++i)
            for (int j=i+1; j<size; ++j)
                choices.emplace_back(-affinity[i*size+j],i,j);
        std::sort(choices.begin(),choices.end());
        std::vector<bool> used(size);
        std::vector<std::pair<int,int>> pairs;
        for (auto [cost,i,j]: choices) {
            if (used[i] || used[j]) continue;
            used[i]=used[j]=true;
            pairs.emplace_back(i,j);
        }
        // Groups are ordered by minimum original rank, so sorting the left
        // index reproduces sorting by the smallest member of each merged pair.
        std::sort(pairs.begin(),pairs.end());
        int half=size/2;
        std::vector<double> next(half*half);
        std::vector<std::vector<int>> merged;
        for (int i=0; i<half; ++i) {
            auto [a,b]=pairs[i];
            merged.push_back(std::move(groups[a]));
            merged.back().insert(merged.back().end(), groups[b].begin(), groups[b].end());
            for (int j=i+1; j<half; ++j) {
                auto [c,d]=pairs[j];
                next[i*half+j]=next[j*half+i]=affinity[a*size+c]+affinity[a*size+d]
                    +affinity[b*size+c]+affinity[b*size+d];
            }
        }
        groups.swap(merged);
        affinity.swap(next);
    }
    Owned result(PyList_New(n));
    if (!result.p) return nullptr;
    for (int i=0; i<n; ++i) {
        PyObject *v=PyLong_FromLong(groups[0][i]);
        if (!v) return nullptr;
        PyList_SET_ITEM(result.p,i,v);
    }
    return result.release();
}
