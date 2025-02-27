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
Initialize the GIL scope.
This macro is used when parsing Python arguments.
This macro must be used with ExitGILScope.
*/
#define InitGILScope \
{\
PyGILState_STATE _gstate = PyGILState_Ensure();

/*
Release the GIL scope.
This macro is used when acess to Python objects.
This macro must be used with InitGILScope.
*/
#define ExitGILScope \
PyGILState_Release(_gstate);\
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