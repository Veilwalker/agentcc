// Expand a validated multicast tree into shifted shard launches. Preserve the
// generator's packet, path and token order as well as the serial path bound.
namespace gather_emit {
struct Route {
    PyObject* path;
    double delay, serial;
};
struct Movement {int route;std::vector<PyObject*> shards;};
struct Batch {int route;std::vector<PyObject*> tokens;};
PyObject* materialize(PyObject*,PyObject* args) {
    PyObject *plans,*compiled,*shards,*times;
    int factor; double size;
    if(!PyArg_ParseTuple(args,"O!O!O!O!id",&PyList_Type,&plans,&PyDict_Type,&compiled,&PyDict_Type,&shards,&PyList_Type,&times,&factor,&size))return nullptr;
    if(factor<1 || factor>32 || !(size>0) || PyList_GET_SIZE(times)>256){PyErr_SetString(PyExc_ValueError,"invalid pipeline expansion");return nullptr;}
    Owned ids(PyDict_New()), result(PyList_New(0));
    Owned path_key(PyUnicode_InternFromString("path")),src_key(PyUnicode_InternFromString("src")),
          dst_key(PyUnicode_InternFromString("dst")),chunks_key(PyUnicode_InternFromString("chunk_ranges")),
          action_key(PyUnicode_InternFromString("action")),copy(PyUnicode_InternFromString("copy"));
    if(!ids.p||!result.p||!path_key.p||!src_key.p||!dst_key.p||!chunks_key.p||!action_key.p||!copy.p)return nullptr;
    try {
        std::vector<Route> routes;
        std::vector<std::vector<Movement>> stages(PyList_GET_SIZE(plans));
        for(int stage=0;stage<int(stages.size());++stage) {
            auto row=PyList_GET_ITEM(plans,stage);if(!PyDict_Check(row))throw std::invalid_argument("invalid tree stage");
            Py_ssize_t pos=0;PyObject *path,*atoms;
            while(PyDict_Next(row,&pos,&path,&atoms)) {
                if(!PyTuple_Check(path)||PyTuple_GET_SIZE(path)<2||!PyList_Check(atoms))throw std::invalid_argument("invalid tree movement");
                auto id=PyDict_GetItemWithError(ids.p,path);int pid;
                if(id)pid=PyLong_AsLong(id);
                else {
                    auto c=PyDict_GetItemWithError(compiled,path);
                    if(!c||!PyTuple_Check(c)||PyTuple_GET_SIZE(c)!=4)throw std::invalid_argument("uncosted tree route");
                    pid=routes.size();
                    Owned number(PyLong_FromLong(pid));
                    if(!number.p||PyDict_SetItem(ids.p,path,number.p)<0)return nullptr;
                    routes.push_back(Route{path,PyFloat_AsDouble(PyTuple_GET_ITEM(c,1)),PyFloat_AsDouble(PyTuple_GET_ITEM(c,2))});
                }
                Movement move{pid,{}};
                for(Py_ssize_t j=0;j<PyList_GET_SIZE(atoms);++j) {
                    auto lanes=PyDict_GetItemWithError(shards,PyList_GET_ITEM(atoms,j));
                    if(!lanes||!PyList_Check(lanes)||PyList_GET_SIZE(lanes)!=factor)throw std::invalid_argument("invalid tree shard");
                    move.shards.push_back(lanes);
                }
                stages[stage].push_back(std::move(move));
            }
        }
        for(int t=0;t<PyList_GET_SIZE(times);++t) {
            double ceiling=PyFloat_AsDouble(PyList_GET_ITEM(times,t));
            std::vector<int> index(routes.size(),-1);
            std::vector<Batch> batches;
            for(int i=std::max(0,t-factor+1);i<std::min(t+1,int(stages.size()));++i)
                for(const auto& move:stages[i]) {
                    int& ix=index[move.route];
                    if(ix<0){ix=batches.size();batches.push_back({move.route,{}});}
                    for(auto lanes:move.shards)batches[ix].tokens.push_back(PyList_GET_ITEM(lanes,t-i));
                }
            Owned transfers(PyList_New(0));if(!transfers.p)return nullptr;
            for(const auto& batch:batches) {
                const auto& r=routes[batch.route];const auto n=batch.tokens.size();
                size_t width=std::max(1.,std::min(double(n),std::floor((ceiling-r.delay)/(size*r.serial)+1e-10)));
                for(size_t i=0;i<n;i+=width) {
                    Owned move(PyDict_New()), path(PySequence_List(r.path)), parts(PyList_New(std::min(n-i,width)));
                    if(!move.p||!path.p||!parts.p)return nullptr;
                    for(int j=0;j<PyList_GET_SIZE(parts.p);++j)PyList_SET_ITEM(parts.p,j,Py_NewRef(batch.tokens[i+j]));
                    if(PyDict_SetItem(move.p,src_key.p,PyTuple_GET_ITEM(r.path,0))<0 ||
                       PyDict_SetItem(move.p,dst_key.p,PyTuple_GET_ITEM(r.path,PyTuple_GET_SIZE(r.path)-1))<0 ||
                       PyDict_SetItem(move.p,path_key.p,path.p)<0 || PyDict_SetItem(move.p,action_key.p,copy.p)<0 ||
                       PyDict_SetItem(move.p,chunks_key.p,parts.p)<0 || PyList_Append(transfers.p,move.p)<0)return nullptr;
                }
            }
            if(PyList_GET_SIZE(transfers.p)) {
                Owned row(Py_BuildValue("{s:n,s:O}","round",PyList_GET_SIZE(result.p)+1,"transfers",transfers.p));
                if(!row.p||PyList_Append(result.p,row.p)<0)return nullptr;
            }
        }
        return result.release();
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
}
}
