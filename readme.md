Define a class in Python C API
```cxx
struct PyTypeObject{
    PyObject_VAR_HEAD ob_base;              // PyVarObject_HEAD_INIT
    const char *      tp_name;              // class name
    Py_ssize_t        tp_basicsize;         // class size
    Py_ssize_t        tp_itemsize;          // class item size
    destructor        tp_dealloc;           // destructor function
    Py_ssize_t        tp_vectorcall_offset; // vectorcall offset
    getattrfunc       tp_getattr;           // getattr function (deprecated)
    setattrfunc       tp_setattr;           // setattr function (deprecated)
    inquiry           tp_as_async;          // as_async function
    reprfunc          tp_repr;              // repr function
    numbermethods*    tp_as_number;         // as_number function
    sequencemethods*  tp_as_sequence;       // as_sequence function
    mappingmethods*   tp_as_mapping;        // as_mapping function
    hashfunc          tp_hash;              // hash function
    ternaryfunc       tp_call;              // call function
    reprfunc          tp_str;               // str function
    getattrofunc      tp_getattro;          // getattro function
    setattrofunc      tp_setattro;          // setattro function
    PyBufferProcs*    tp_as_buffer;         // as_buffer function
    unsigned long     tp_flags;             // flags
    const char *      tp_doc;               // docstring
    traverseproc      tp_traverse;          // traverse function
    inquiry           tp_clear;             // clear function
    richcmpfunc       tp_richcompare;       // richcompare function
    Py_ssize_t        tp_weaklistoffset;    // weaklist offset
    getiterfunc       tp_iter;              // iter function
    iternextfunc      tp_iternext;          // iternext function
    PyMethodDef*      tp_methods;           // methods {const char* method_name, PyCFunction method_func, int method_flags, const char* method_doc}
    struct PyMemberDef* tp_members;         // members {const char* name, int type, Py_ssize_t offset, int flags, const char* doc}
    struct PyGetSetDef* tp_getset;          // getset  {const char* name, getter, setter, const char* doc, void* closure}
    struct _typeobject* tp_base;            // base
    PyObject*         tp_dict;              // dict
    descrgetfunc      tp_descr_get;         // descr_get
    descrsetfunc      tp_descr_set;         // descr_set
    Py_ssize_t        tp_dictoffset;        // dict offset
    initproc          tp_init;              // init function
    allocfunc         tp_alloc;             // alloc function
    newfunc           tp_new;               // new function
    freefunc          tp_free;              // free function
    inquiry           tp_is_gc;             // is_gc function
    PyObject*         tp_bases;             // bases
    PyObject*         tp_mro;               // mro (method resolution order)
    PyObject*         tp_cache;             // cache (no longer used)
    PyObject*         tp_subclasses;        // subclasses (for builtin types this is an index)
    PyObject*         tp_weaklist;          // weaklist (not used for builtin types)
    destructor        tp_del;               // del function (deprecated)
    unsigned int      tp_version_tag;       // version tag
    destructor        tp_finalize;          // finalize function
    vectorcallfunc    tp_vectorcall;        // vectorcall function
    unsigned int      tp_watched;           // watched
};
```

Define a module in Python C API
```cxx
struct PyModuleDef{
    PyModuleDef_Base  m_base,     // PyModuleDef_HEAD_INIT
    const char *      m_name,     // module name
    const char *      m_doc,      // module docstring 
    Py_ssize_t        m_size,     // module state size  -1
    PyMethodDef*      m_methods,  // module methods     {const char* method_name, PyCFunction method_func, int method_flags, const char* method_doc}
    PyModuleDef_Slot* m_slots,    // module slots       {int slot, void *value}
    traverseproc      m_traverse, // traverse function  int  tp_traverse (PyObject *self, visitproc visit, void *arg)
    inquiry           m_clear,    // clear function     int  tp_clear    (PyObject *self)
    freefunc          m_free,     // free function      void tp_free     (void *self)
};
```