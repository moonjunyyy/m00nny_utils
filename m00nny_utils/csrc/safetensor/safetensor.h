#ifndef LIBM00NNY_SAFETENSOR_H
#define LIBM00NNY_SAFETENSOR_H

#include "common/common.h"
#include "common/torch_common.h"
#include "common/macros.h"
#include "common/pyarg.h"

// The reader itself is utils/safetensor.h. It is included by safetensor.cpp alone: it and the headers it uses
// (utils/json.h, utils/utils.h) define functions that are not inline, so a second file including them would
// define them twice. Here the class is only named.
namespace safetensor { class Safetensor; }

namespace m00nny {
/*
m00nny_utils.safetensor.Safetensor: one .safetensors file. The header is read when the object is made; a tensor,
or a block of one, is read from the file when asked for. The object is not changed after __init__, so one object
can be read from several threads at once.
*/
struct SafetensorObject
{
    PyObject_HEAD
    safetensor::Safetensor* file;
};
} // namespace m00nny

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int       Safetensor_init     (m00nny::SafetensorObject *self, PyObject *args, PyObject *kwds);
void      Safetensor_dealloc  (m00nny::SafetensorObject *self);
PyObject* Safetensor_repr     (m00nny::SafetensorObject *self);
PyObject* Safetensor_keys     (m00nny::SafetensorObject *self, PyObject *ignored);
PyObject* Safetensor_info     (m00nny::SafetensorObject *self, PyObject *args, PyObject *kwds);
PyObject* Safetensor_read     (m00nny::SafetensorObject *self, PyObject *args, PyObject *kwds);
PyObject* Safetensor_get_path (PyObject *self, void *closure);

#ifdef __cplusplus
}
#endif // __cplusplus

static PyMethodDef Safetensor_methods[] =
{
    {"keys", (PyCFunction)Safetensor_keys, METH_NOARGS,                  "The names of the tensors in the file."},
    {"info", (PyCFunction)Safetensor_info, METH_VARARGS | METH_KEYWORDS, "What the header says of a tensor: dtype, shape, data_offsets."},
    {"read", (PyCFunction)Safetensor_read, METH_VARARGS | METH_KEYWORDS, "Reads a tensor, or the block of it an index names, from the file."},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static PyGetSetDef Safetensor_getsetters[] =
{
    {"path", (getter)Safetensor_get_path, NULL, "The file's path", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject SafetensorType =
{
    .ob_base      = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "m00nny_utils.safetensor.Safetensor",
    .tp_basicsize = sizeof(m00nny::SafetensorObject),
    .tp_dealloc   = (destructor)Safetensor_dealloc,
    .tp_repr      = (reprfunc)Safetensor_repr,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "A .safetensors file: its header, and its tensors read on request",
    .tp_methods   = Safetensor_methods,
    .tp_getset    = Safetensor_getsetters,
    .tp_init      = (initproc)Safetensor_init,
    .tp_new       = PyType_GenericNew,
};

static PyModuleDef Safetensor_Module =
{
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "safetensor",
    .m_doc  = "Reading .safetensors files into torch tensors",
    .m_size = -1
};

// Types of m00nny_utils.safetensor, added in this order by PyInit_safetensor (safetensor.cpp).
static PyTypeObject* const Safetensor_Types[] =
{
    &SafetensorType,
};

PyMODINIT_FUNC PyInit_safetensor(void);

#endif // LIBM00NNY_SAFETENSOR_H
