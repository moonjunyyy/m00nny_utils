#include "algorithm/algorithm.h"

PyMODINIT_FUNC
PyInit_algorithm
(void)
{
    PyObject *m;
    m = PyModule_Create(&Algorithm_Module);
    if (m == NULL)
        return NULL;
    for (PyTypeObject *type : Algorithm_Types)
        PyModule_AddType(m, type);
    return m;
}
