#include "safetensor/safetensor.h"
#include "utils/safetensor.h"

#include <optional>
#include <string>
#include <vector>

namespace {

// The file of an object, or NULL with a Python exception set when __init__ has not opened one.
safetensor::Safetensor*
file_of
(m00nny::SafetensorObject *self)
{
    if (self->file == nullptr) PyErr_SetString(PyExc_RuntimeError, "Safetensor: no file is open (__init__ was not run)");
    return self->file;
}

// Whether the file has a tensor of that name; KeyError is set when it has not.
bool
has_named
(const safetensor::Safetensor& file, const char *name)
{
    if (file.has_tensor(name)) return true;
    PyErr_SetString(PyExc_KeyError, name);
    return false;
}

/*
An index into a stored tensor, as in tensor[index]: an int, a slice of step 1, or a tuple of them for the leading
dimensions; None (or nothing) is the whole tensor. Gives the block to read, one range per indexed dimension, and
the shape the block has once the dimensions indexed by an int are dropped.
*/
bool
to_block
(PyObject *index, const std::vector<int64_t>& shape,
 std::vector<safetensor::Safetensor::Range>& ranges, std::vector<int64_t>& indexed_shape)
{
    ranges.clear();
    indexed_shape.clear();
    const bool given = m00nny::pyarg::given(index);
    const bool is_tuple = given && PyTuple_Check(index);
    const Py_ssize_t count = !given ? 0 : is_tuple ? PyTuple_GET_SIZE(index) : 1;
    if ((size_t)count > shape.size())
    {
        PyErr_Format(PyExc_IndexError, "too many indices: %zd for a tensor of %zu dimensions", count, shape.size());
        return false;
    }
    for (Py_ssize_t d = 0; d < count; ++d)
    {
        PyObject *item = is_tuple ? PyTuple_GET_ITEM(index, d) : index;
        const Py_ssize_t size = (Py_ssize_t)shape[d];
        if (PySlice_Check(item))
        {
            Py_ssize_t start, stop, step;
            if (PySlice_Unpack(item, &start, &stop, &step) < 0) return false;
            if (step != 1) { PyErr_SetString(PyExc_ValueError, "a slice of a stored tensor has step 1"); return false; }
            const Py_ssize_t length = PySlice_AdjustIndices(size, &start, &stop, step);
            ranges.push_back({(int64_t)start, (int64_t)(start + length)});
            indexed_shape.push_back((int64_t)length);
        }
        else if (PyLong_Check(item) && !PyBool_Check(item))
        {
            long long i = PyLong_AsLongLong(item);
            if (PyErr_Occurred()) return false;
            if (i < -(long long)size || i >= (long long)size)
            {
                PyErr_Format(PyExc_IndexError, "index %lld is out of range for dimension %zd with size %zd", i, d, size);
                return false;
            }
            if (i < 0) i += size;
            ranges.push_back({(int64_t)i, (int64_t)i + 1});
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "index must be an int, a slice, or a tuple of them");
            return false;
        }
    }
    for (size_t d = (size_t)count; d < shape.size(); ++d) indexed_shape.push_back(shape[d]);
    return true;
}

} // namespace

int
Safetensor_init
(m00nny::SafetensorObject *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: Safetensor.__init__\n")
    const char *kwlist[] = {"path", NULL};
    PyObject *py_path = nullptr;   // bytes: a str or an os.PathLike in the file system's encoding
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", (char**)kwlist, PyUnicode_FSConverter, &py_path)) return -1;
    const std::string path(PyBytes_AS_STRING(py_path));
    Py_DECREF(py_path);
    LIBM00NNY_PY_TRY
    {
        safetensor::Safetensor *file = new safetensor::Safetensor(path);
        delete self->file;   // __init__ run again: the object is the new file
        self->file = file;
        return 0;
    }
    LIBM00NNY_PY_CATCH(-1)
}

void
Safetensor_dealloc
(m00nny::SafetensorObject *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: Safetensor.__dealloc__\n")
    delete self->file;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject*
Safetensor_repr
(m00nny::SafetensorObject *self)
{
    if (self->file == nullptr) return PyUnicode_FromString("Safetensor(<no file>)");
    LIBM00NNY_PY_TRY
    {
        return PyUnicode_FromFormat("Safetensor('%s', %zu tensors)",
                                    self->file->filename().c_str(), self->file->get_tensor_names().size());
    }
    LIBM00NNY_PY_CATCH(NULL)
}

PyObject*
Safetensor_get_path
(PyObject *self, void *closure)
{
    safetensor::Safetensor *file = file_of((m00nny::SafetensorObject*)self);
    if (file == nullptr) return NULL;
    return PyUnicode_DecodeFSDefaultAndSize(file->filename().data(), (Py_ssize_t)file->filename().size());
}

PyObject*
Safetensor_keys
(m00nny::SafetensorObject *self, PyObject *ignored)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: Safetensor_keys\n")
    safetensor::Safetensor *file = file_of(self);
    if (file == nullptr) return NULL;
    LIBM00NNY_PY_TRY
    {
        const std::vector<std::string> names = file->get_tensor_names();
        PyObject *list = PyList_New((Py_ssize_t)names.size());
        if (list == NULL) return NULL;
        for (size_t i = 0; i < names.size(); ++i)
        {
            PyObject *name = PyUnicode_FromStringAndSize(names[i].data(), (Py_ssize_t)names[i].size());
            if (name == NULL) { Py_DECREF(list); return NULL; }
            PyList_SET_ITEM(list, (Py_ssize_t)i, name);
        }
        return list;
    }
    LIBM00NNY_PY_CATCH(NULL)
}

PyObject*
Safetensor_info
(m00nny::SafetensorObject *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: Safetensor_info\n")
    const char *kwlist[] = {"name", NULL};
    const char *name = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", (char**)kwlist, &name)) return NULL;
    safetensor::Safetensor *file = file_of(self);
    if (file == nullptr || !has_named(*file, name)) return NULL;
    LIBM00NNY_PY_TRY
    {
        const safetensor::Safetensor::TensorInfo info = file->get_tensor_info(name);
        PyObject *shape = PyTuple_New((Py_ssize_t)info.shape.size());
        if (shape == NULL) return NULL;
        for (size_t d = 0; d < info.shape.size(); ++d)
        {
            PyObject *size = PyLong_FromLongLong((long long)info.shape[d]);
            if (size == NULL) { Py_DECREF(shape); return NULL; }
            PyTuple_SET_ITEM(shape, (Py_ssize_t)d, size);
        }
        // As the header has them: the dtype by the format's name, the offsets from the start of the data section.
        return Py_BuildValue("{s:s,s:N,s:(LL)}",
                             "dtype", info.dtype_info.safetensor_name.c_str(),
                             "shape", shape,
                             "data_offsets", (long long)info.offset_begin, (long long)info.offset_end);
    }
    LIBM00NNY_PY_CATCH(NULL)
}

PyObject*
Safetensor_read
(m00nny::SafetensorObject *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: Safetensor_read\n")
    const char *kwlist[] = {"name", "index", "dtype", "device", NULL};
    const char *py_name = nullptr;
    PyObject *py_index = nullptr, *py_dtype = nullptr, *py_device = nullptr;
    std::optional<torch::ScalarType> dtype;
    torch::Device device(torch::kCPU);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|O$OO", (char**)kwlist, &py_name, &py_index, &py_dtype, &py_device)) return NULL;
    safetensor::Safetensor *file = file_of(self);
    if (file == nullptr || !has_named(*file, py_name)) return NULL;
    if (!m00nny::pyarg::to_dtype(py_dtype, "dtype", dtype)
        || !m00nny::pyarg::to_device(py_device, "device", torch::Device(torch::kCPU), device)) return NULL;
    LIBM00NNY_PY_TRY
    {
        const std::string name(py_name);
        const safetensor::Safetensor::TensorInfo info = file->get_tensor_info(name);
        std::vector<safetensor::Safetensor::Range> ranges;
        std::vector<int64_t> indexed_shape;
        if (!to_block(py_index, info.shape, ranges, indexed_shape)) return NULL;
        torch::Tensor tensor;
        {
            // Reading the file, converting and moving to the device touch no Python object: the other
            // Python threads (other tensors being read) run meanwhile.
            LibM00nnyNoGILScope without_gil;
            tensor = file->read(name, ranges, dtype, device);
            if (indexed_shape.size() != info.shape.size()) tensor = tensor.reshape(indexed_shape);
        }
        return THPVariable_Wrap(tensor);
    }
    LIBM00NNY_PY_CATCH(NULL)
}

PyMODINIT_FUNC
PyInit_safetensor
(void)
{
    PyObject *m;
    m = LibM00nnyCreateModule(&Safetensor_Module);
    if (m == NULL)
        return NULL;
    for (PyTypeObject *type : Safetensor_Types)
        PyModule_AddType(m, type);
    return m;
}
