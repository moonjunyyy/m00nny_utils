#include "m00nny.h"

PyMODINIT_FUNC
PyInit_algorithm
(void)
{
    PyObject *m;
    m = PyModule_Create(&Algorithm_Module);
    if (m == NULL)
        return NULL;
    PyModule_AddType(m, &TSNEType);
    PyModule_AddType(m, &KMeansType);
    return m;
}

PyMODINIT_FUNC
PyInit_linalg
(void)
{
    PyObject *m;
    m = PyModule_Create(&Linalg_Module);
    if (m == NULL)
        return NULL;
    return m;
}

PyMODINIT_FUNC
PyInit_m00nny_utils
(void)
{
    PyObject *m;
    PyObject *algorithm_module, *linalg_module;

    m = PyModule_Create(&M00NNY_Utils_Module);
    if (m == NULL) return NULL;

    algorithm_module = PyInit_algorithm();
    if (algorithm_module == NULL) return NULL;
    PyModule_AddObject(m, "algorithm", algorithm_module);
    
    linalg_module = PyInit_linalg();
    if (linalg_module == NULL) return NULL;
    PyModule_AddObject(m, "linalg", linalg_module);
    
    return m;
}