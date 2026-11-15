#include "m00nny.h"

PyMODINIT_FUNC
PyInit_m00nny_utils
(void)
{
    PyObject *m;

    m = PyModule_Create(&M00NNY_Utils_Module);
    if (m == NULL) return NULL;

    for (const auto &submodule : M00NNY_Utils_Submodules)
    {
        PyObject *module = submodule.init();
        if (module == NULL) return NULL;
        PyModule_AddObject(m, submodule.name, module);
    }

    return m;
}
