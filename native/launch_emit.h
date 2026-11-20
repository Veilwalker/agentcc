// Place already-selected multicast tree arcs in their assigned launch rounds.
// Preserve first insertion order within each round, including owner/token order.
namespace launch_emit {
struct Path { PyObject *nodes; double latency, inverse; };
struct Wave {
    std::vector<int> positions, paths;
    std::vector<std::vector<PyObject*>> atoms;
    explicit Wave(int count): positions(count,-1) {}
    std::vector<PyObject*>& payload(int id) {
        int& position=positions.at(id);
        if (position<0) {
            position=paths.size();
            paths.push_back(id);
            atoms.emplace_back();
        }
        return atoms[position];
    }
};
int integer(PyObject *obj, int limit) {
    if(!obj)throw std::invalid_argument("missing launch index");
    long value=PyLong_AsLong(obj);
    if(PyErr_Occurred() || value<0 || value>=limit)throw std::invalid_argument("launch index outside table");
    return int(value);
}
Path route(PyObject* obj) {
    if(!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj)!=4)throw std::invalid_argument("invalid compiled launch route");
    auto nodes=PyTuple_GET_ITEM(obj,3);
    double delay=PyFloat_AsDouble(PyTuple_GET_ITEM(obj,1)),inverse=PyFloat_AsDouble(PyTuple_GET_ITEM(obj,2));
    if(!PyTuple_Check(nodes)||PyTuple_GET_SIZE(nodes)<2||!std::isfinite(delay)||!std::isfinite(inverse)||inverse<=0)
        throw std::invalid_argument("invalid launch path");
    return {nodes,delay,inverse};
}
PyObject* emit(const std::vector<Wave>& waves,const std::vector<Path>& paths,PyObject* times,double size) {
    Owned result(PyList_New(0)),src(PyUnicode_InternFromString("src")),dst(PyUnicode_InternFromString("dst")),
          path(PyUnicode_InternFromString("path")),action(PyUnicode_InternFromString("action")),
          copy(PyUnicode_InternFromString("copy")),chunks(PyUnicode_InternFromString("chunk_ranges"));
    if(!result.p||!src.p||!dst.p||!path.p||!action.p||!copy.p||!chunks.p)return nullptr;
    for(size_t t=0;t<waves.size();++t) {
        const Wave& wave=waves[t];
        if(wave.paths.empty())continue;
        double limit=PyFloat_AsDouble(PyList_GET_ITEM(times,t));
        if(!std::isfinite(limit)||PyErr_Occurred())throw std::invalid_argument("invalid launch duration");
        Owned moves(PyList_New(0));
        if(!moves.p)return nullptr;
        for(size_t j=0;j<wave.paths.size();++j) {
            const auto& r=paths[wave.paths[j]];
            const auto& tokens=wave.atoms[j];
            size_t width=std::max(1.,std::min(double(tokens.size()),std::floor((limit-r.latency)/(size*r.inverse)+1e-10)));
            for(size_t k=0;k<tokens.size();k+=width) {
                size_t count=std::min(width,tokens.size()-k);
                Owned packet(PyDict_New()),nodes(PySequence_List(r.nodes)),parts(PyList_New(count));
                if(!packet.p||!nodes.p||!parts.p)return nullptr;
                for(size_t n=0;n<count;++n)PyList_SET_ITEM(parts.p,n,Py_NewRef(tokens[k+n]));
                if(PyDict_SetItem(packet.p,src.p,PyTuple_GET_ITEM(r.nodes,0))<0||
                   PyDict_SetItem(packet.p,dst.p,PyTuple_GET_ITEM(r.nodes,PyTuple_GET_SIZE(r.nodes)-1))<0||
                   PyDict_SetItem(packet.p,action.p,copy.p)<0||PyDict_SetItem(packet.p,path.p,nodes.p)<0||
                   PyDict_SetItem(packet.p,chunks.p,parts.p)<0||PyList_Append(moves.p,packet.p)<0)return nullptr;
            }
        }
        Owned row(Py_BuildValue("{s:n,s:O}","round",PyList_GET_SIZE(result.p)+1,"transfers",moves.p));
        if(!row.p||PyList_Append(result.p,row.p)<0)return nullptr;
    }
    return result.release();
}
PyObject* paced(PyObject*,PyObject* args) {
    PyObject *routes,*packets,*launches,*times;double size;
    if(!PyArg_ParseTuple(args,"O!O!O!O!d",&PyList_Type,&routes,&PyList_Type,&packets,
        &PyList_Type,&launches,&PyList_Type,&times,&size))return nullptr;
    try {
        int nt=PyList_GET_SIZE(times),np=PyList_GET_SIZE(routes),nr=PyList_GET_SIZE(launches);
        if(nt<1||nt>256||np>200000||int64_t(nt)*np>6000000||!(size>0))throw std::invalid_argument("paced layout size");
        std::vector<Path> paths;std::vector<Wave>waves;
        for(int i=0;i<np;++i)paths.push_back(route(PyList_GET_ITEM(routes,i)));
        for(int i=0;i<nt;++i)waves.emplace_back(np);
        std::vector<std::vector<std::pair<int,PyObject*>>> launches_by_rank(nr);
        for(int r=0;r<nr;++r) {
            PyObject* by_time=PyList_GET_ITEM(launches,r);
            if(!PyTuple_Check(by_time))throw std::invalid_argument("paced rank launches");
            for(int j=0;j<PyTuple_GET_SIZE(by_time);++j) {
                auto item=PyTuple_GET_ITEM(by_time,j);
                if(!PyTuple_Check(item)||PyTuple_GET_SIZE(item)!=2||!PyList_Check(PyTuple_GET_ITEM(item,1)))throw std::invalid_argument("paced tokens");
                launches_by_rank[r].emplace_back(integer(PyTuple_GET_ITEM(item,0),nt),PyTuple_GET_ITEM(item,1));
            }
        }
        for(int j=0;j<PyList_GET_SIZE(packets);++j) {
            auto p=PyList_GET_ITEM(packets,j);
            if(!PyTuple_Check(p)||PyTuple_GET_SIZE(p)!=3||!PyTuple_Check(PyTuple_GET_ITEM(p,2)))throw std::invalid_argument("invalid paced arc");
            int age=integer(PyTuple_GET_ITEM(p,0),nt),id=integer(PyTuple_GET_ITEM(p,1),np);auto owners=PyTuple_GET_ITEM(p,2);
            for(int k=0;k<PyTuple_GET_SIZE(owners);++k)for(const auto& [delay,parts]:launches_by_rank[integer(PyTuple_GET_ITEM(owners,k),nr)]) {
                if(age+delay>=nt)throw std::invalid_argument("paced round out of range");
                auto& v=waves[age+delay].payload(id);
                for(int a=0;a<PyList_GET_SIZE(parts);++a)v.push_back(PyList_GET_ITEM(parts,a));
            }
        }
        return emit(waves,paths,times,size);
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
PyObject* shifted(PyObject*,PyObject* args) {
    PyObject *plans,*catalog,*offsets,*times;double size;
    if(!PyArg_ParseTuple(args,"O!O!O!O!d",&PyList_Type,&plans,&PyDict_Type,&catalog,
                         &PyDict_Type,&offsets,&PyList_Type,&times,&size))return nullptr;
    try {
        int nt=PyList_GET_SIZE(times);
        if(nt<1||nt>256||!(size>0))throw std::invalid_argument("invalid shifted launch dimensions");
        Owned ids(PyDict_New());
        if(!ids.p)return nullptr;
        std::vector<Path> paths;
        // Discovery pass bounds the dense per-round route index by used arcs.
        for(int t=0;t<PyList_GET_SIZE(plans);++t) {
            auto row=PyList_GET_ITEM(plans,t);
            if(!PyDict_Check(row))throw std::invalid_argument("invalid shifted tree row");
            Py_ssize_t pos=0;PyObject *key,*atoms;
            while(PyDict_Next(row,&pos,&key,&atoms)) {
                int present=PyDict_Contains(ids.p,key);
                if(present<0)return nullptr;
                if(!present) {
                    auto compiled=PyDict_GetItemWithError(catalog,key);
                    if(!compiled)throw std::invalid_argument("unknown shifted route");
                    Owned id(PyLong_FromSize_t(paths.size()));
                    if(!id.p||PyDict_SetItem(ids.p,key,id.p)<0)return nullptr;
                    paths.push_back(route(compiled));
                }
            }
        }
        if(size_t(nt)*paths.size()>6000000)throw std::invalid_argument("shifted route index too large");
        std::vector<Wave> waves;for(int i=0;i<nt;++i)waves.emplace_back(paths.size());
        for(int t=0;t<PyList_GET_SIZE(plans);++t) {
            auto row=PyList_GET_ITEM(plans,t);Py_ssize_t pos=0;PyObject *key,*atoms;
            while(PyDict_Next(row,&pos,&key,&atoms)) {
                int id=integer(PyDict_GetItemWithError(ids.p,key),paths.size());
                if(!PyList_Check(atoms))throw std::invalid_argument("invalid shifted atom list");
                for(int k=0;k<PyList_GET_SIZE(atoms);++k) {
                    auto atom=PyList_GET_ITEM(atoms,k), delay=PyDict_GetItemWithError(offsets,atom);
                    if(!delay)throw std::invalid_argument("unknown shifted atom");
                    int arrival=t+integer(delay,nt);
                    if(arrival>=nt)throw std::invalid_argument("shifted round out of range");
                    waves[arrival].payload(id).push_back(atom);
                }
            }
        }
        return emit(waves,paths,times,size);
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
}
