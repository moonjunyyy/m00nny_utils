#ifndef LIBM00NNY_PYARG_H
#define LIBM00NNY_PYARG_H

#include "common/common.h"
#include "common/torch_common.h"
#include "common/macros.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

/*
Reading the arguments of a Python call, the same way in every binding:
- each PyObject* an argument is parsed into starts as nullptr: PyArg_ParseTupleAndKeywords leaves an omitted
  optional argument untouched, so an uninitialized pointer would be read as garbage;
- an optional argument is read here with its default (omitted or None -> default), so no temporary Python object
  is made for a default and none is leaked;
- a reader returns false with a Python exception set when the value has the wrong type; the binding returns.
*/
namespace m00nny {
namespace pyarg {

inline bool given(PyObject* o) { return o != nullptr && !Py_IsNone(o); }

inline bool to_tensor(PyObject* o, const char* name, torch::Tensor& out)
{
    if (!given(o)) { PyErr_Format(PyExc_TypeError, "%s is required", name); return false; }
    if (!THPVariable_Check(o)) { PyErr_Format(PyExc_TypeError, "%s must be a torch.Tensor", name); return false; }
    out = THPVariable_Unpack(o);
    return true;
}

inline bool to_int(PyObject* o, const char* name, int64_t dflt, int64_t& out)
{
    if (!given(o)) { out = dflt; return true; }
    if (!PyLong_Check(o)) { PyErr_Format(PyExc_TypeError, "%s must be an int", name); return false; }
    out = PyLong_AsLongLong(o);
    return !PyErr_Occurred();
}

inline bool to_double(PyObject* o, const char* name, double dflt, double& out)
{
    if (!given(o)) { out = dflt; return true; }
    if (!PyFloat_Check(o) && !PyLong_Check(o)) { PyErr_Format(PyExc_TypeError, "%s must be a float", name); return false; }
    out = PyFloat_AsDouble(o);
    return !PyErr_Occurred();
}

// A name from a declared table {name, value}; the error lists the names.
template <typename T, std::size_t N>
inline bool to_choice(PyObject* o, const char* name, const std::pair<const char*, T> (&names)[N], T dflt, T& out)
{
    if (!given(o)) { out = dflt; return true; }
    if (!PyUnicode_Check(o)) { PyErr_Format(PyExc_TypeError, "%s must be a str", name); return false; }
    const char* s = PyUnicode_AsUTF8(o);
    if (s == nullptr) return false;
    for (const auto& entry : names)
        if (std::strcmp(entry.first, s) == 0) { out = entry.second; return true; }
    std::string allowed;
    for (const auto& entry : names) allowed += (allowed.empty() ? "'" : ", '") + std::string(entry.first) + "'";
    PyErr_Format(PyExc_ValueError, "%s must be one of %s; got '%s'", name, allowed.c_str(), s);
    return false;
}

// A torch.dtype; omitted or None -> no value (what that means is the function's: e.g. the dtype as stored).
inline bool to_dtype(PyObject* o, const char* name, std::optional<torch::ScalarType>& out)
{
    if (!given(o)) { out = std::nullopt; return true; }
    if (!THPDtype_Check(o)) { PyErr_Format(PyExc_TypeError, "%s must be a torch.dtype", name); return false; }
    out = ((THPDtype*)o)->scalar_type;
    return true;
}

// A device: a torch.device, a device string ("cpu", "cuda:1"), or an int, the index of a CUDA device.
inline bool to_device(PyObject* o, const char* name, const torch::Device& dflt, torch::Device& out)
{
    if (!given(o)) { out = dflt; return true; }
    if (THPDevice_Check(o)) { out = ((THPDevice*)o)->device; return true; }
    if (PyUnicode_Check(o))
    {
        const char* s = PyUnicode_AsUTF8(o);
        if (s == nullptr) return false;
        try { out = torch::Device(std::string(s)); }
        catch (const c10::Error&) { PyErr_Format(PyExc_ValueError, "%s: '%s' does not name a device", name, s); return false; }
        return true;
    }
    if (PyLong_Check(o) && !PyBool_Check(o))
    {
        const long index = PyLong_AsLong(o);
        if (PyErr_Occurred()) return false;
        if (index < 0 || index > INT16_MAX) { PyErr_Format(PyExc_ValueError, "%s as an int is the index of a CUDA device: got %ld", name, index); return false; }
        out = torch::Device(torch::kCUDA, (c10::DeviceIndex)index);
        return true;
    }
    PyErr_Format(PyExc_TypeError, "%s must be a torch.device, a str or an int", name);
    return false;
}

template <typename T, std::size_t N>
inline const char* name_of(const std::pair<const char*, T> (&names)[N], T value)
{
    for (const auto& entry : names)
        if (entry.second == value) return entry.first;
    return "unknown";
}

} // namespace pyarg
} // namespace m00nny

/*
A C++ exception must not cross the C frames of a Python call: it would terminate the process. A binding runs
its C++ work as  LIBM00NNY_PY_TRY { ... } LIBM00NNY_PY_CATCH(error_value) , which sets the matching Python
exception and returns error_value (NULL for a function, -1 for init/setters):
std::invalid_argument and c10::ValueError -> ValueError, c10::TypeError -> TypeError, c10::Error and any other
std::exception -> RuntimeError.
*/
#define LIBM00NNY_PY_TRY try
#define LIBM00NNY_PY_CATCH(error_value)                                                                          \
    catch (const c10::ValueError& e)       { PyErr_SetString(PyExc_ValueError,   e.what_without_backtrace()); return error_value; } \
    catch (const c10::TypeError& e)        { PyErr_SetString(PyExc_TypeError,    e.what_without_backtrace()); return error_value; } \
    catch (const c10::Error& e)            { PyErr_SetString(PyExc_RuntimeError, e.what_without_backtrace()); return error_value; } \
    catch (const std::invalid_argument& e) { PyErr_SetString(PyExc_ValueError,   e.what());                   return error_value; } \
    catch (const std::exception& e)        { PyErr_SetString(PyExc_RuntimeError, e.what());                   return error_value; }

#endif // LIBM00NNY_PYARG_H
