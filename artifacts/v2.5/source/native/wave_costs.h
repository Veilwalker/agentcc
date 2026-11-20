// Price a uniform fine-grained pipeline from sparse per-round profiles.
PyObject* pipeline_costs(PyObject*, PyObject* args) {
    int resources;
    double bytes;
    PyObject *rows, *factors;
    if (!PyArg_ParseTuple(args, "idO!O!", &resources, &bytes, &PyList_Type, &rows,
                          &PyList_Type, &factors)) return nullptr;
    const int count = PyList_GET_SIZE(rows);
    if (resources <= 0 || count <= 0 || count > 128 ||
        static_cast<long long>(resources)*count > 2000000 || !(bytes>0) || !std::isfinite(bytes)) {
        PyErr_SetString(PyExc_ValueError,"invalid pipeline dimensions"); return nullptr;
    }
    try {
        std::vector<std::vector<std::pair<int,double>>> demand(count);
        std::vector<double> latency(count), serial(count);
        for (int i=0; i<count; ++i) {
            PyObject* loads;
            if (!PyArg_ParseTuple(PyList_GET_ITEM(rows,i),"O!dd",&PyDict_Type,&loads,
                                  &latency[i],&serial[i])) return nullptr;
            PyObject *key,*value; Py_ssize_t position=0;
            while (PyDict_Next(loads,&position,&key,&value)) {
                long id=PyLong_AsLong(key); double weight=PyFloat_AsDouble(value);
                if (PyErr_Occurred()) return nullptr;
                if (id<0 || id>=resources || !std::isfinite(weight) || weight<0) {
                    PyErr_SetString(PyExc_ValueError,"invalid pipeline resource"); return nullptr;
                }
                demand[i].emplace_back(id,weight);
            }
        }
        Owned result(PyList_New(0));
        if (!result.p) return nullptr;
        for (Py_ssize_t k=0; k<PyList_GET_SIZE(factors); ++k) {
            long factor=PyLong_AsLong(PyList_GET_ITEM(factors,k));
            if (factor<1 || factor>32 || PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError,"invalid pipeline factor"); return nullptr;
            }
            const int length=count+factor-1;
            std::vector<double> loads(resources), times(length);
            const double unit=bytes/factor;
            {
                AllowThreads unlock;
                for (int t=0; t<length; ++t) {
                    if (t<count) for (auto [id,v]:demand[t]) loads[id]+=v;
                    if (t>=factor) for (auto [id,v]:demand[t-factor]) loads[id]-=v;
                    double overhead=0, path_bound=0;
                    for (int i=std::max(0,t-static_cast<int>(factor)+1); i<=std::min(t,count-1); ++i) {
                        overhead=std::max(overhead,latency[i]);
                        path_bound=std::max(path_bound,latency[i]+unit*serial[i]);
                    }
                    times[t]=std::max(path_bound,overhead+unit*(*std::max_element(loads.begin(),loads.end())));
                }
            }
            Owned out(PyList_New(length));
            if (!out.p) return nullptr;
            for (int i=0;i<length;++i) {
                PyObject* v=PyFloat_FromDouble(times[i]);
                if (!v) return nullptr;
                PyList_SET_ITEM(out.p,i,v);
            }
            if (PyList_Append(result.p,out.p)<0) return nullptr;
        }
        return result.release();
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}
