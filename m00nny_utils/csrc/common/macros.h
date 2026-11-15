#ifndef LIBM00NNY_MACROS_H
#define LIBM00NNY_MACROS_H

#ifdef LIBM00NNY_DEBUG
#define LIBM00NNY_DEBUG_MESSAGE(...) printf(__VA_ARGS__);
#else
#define LIBM00NNY_DEBUG_MESSAGE(...)
#endif // KMEANS_DEBUG

#ifdef LIBM00NNY_DEBUG
#define LIBM00NNY_ERROR_MESSAGE(...) fprintf(stderr, __VA_ARGS__);
#else
#define LIBM00NNY_ERROR_MESSAGE(...) fprintf(stderr, __VA_ARGS__);
#endif // KMEANS_DEBUG

/*
    If the condition is true, execute the behavior and increment the stack.
    If the condition is false, do nothing and shift the stack.
*/
#define IfSaveState(stack, cond, behavior) if (cond) { behavior(); stack++; } stack <<= 1;

/*
    If the condition in the stack is true, execute the behavior.
    If the condition in the stack is false, do nothing.
    Note that the restored state is not saved, and it works with LIFO manner.
*/
#define IfRestoreState(stack, behavior) stack >>= 1; if (stack & 1) { behavior(); }

#ifdef TORCH_API
#endif // TORCH_API

// Python Macros
#ifdef Py_PYTHON_H // Python.h is included

/*
The GIL held for a scope (RAII): released when the scope ends, also on an early return out of it.
InitGILScope opens the scope and takes the GIL; ExitGILScope closes it. A return between the two
releases the GIL too (with a plain Ensure/Release pair it did not).
*/
struct LibM00nnyGILScope
{
    PyGILState_STATE state;
    LibM00nnyGILScope() : state(PyGILState_Ensure()) {}
    ~LibM00nnyGILScope() { PyGILState_Release(this->state); }
    LibM00nnyGILScope(const LibM00nnyGILScope&) = delete;
    LibM00nnyGILScope& operator=(const LibM00nnyGILScope&) = delete;
};
#define InitGILScope { LibM00nnyGILScope _gil_scope;
#define ExitGILScope }

/*
A module of this library: PyModule_Create and, on a free-threaded CPython, the declaration that the module runs
without the GIL (without it, importing the library turns the GIL back on for the whole process).
The functions keep no shared mutable state. An object (KMeans, TSNE) is not synchronized: use one object from one
thread at a time.
*/
inline PyObject*
LibM00nnyCreateModule(PyModuleDef* def)
{
    PyObject* m = PyModule_Create(def);
#ifdef Py_GIL_DISABLED
    if (m != NULL && PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED) < 0) { Py_DECREF(m); return NULL; }
#endif
    return m;
}

/*
Set an attribute to the object with a string name.
This macro is used when setting an attribute with C-style string.
*/
#define PyObject_GenericSetAttrWithString(obj, name, value) \
{\
    PyObject* _name = PyUnicode_FromString(name);\
    PyObject_GenericSetAttr(obj, _name, value);\
    Py_XDECREF(_name);\
}

#endif // Py_PYTHON_H
#endif // LIBM00NNY_MACROS_H