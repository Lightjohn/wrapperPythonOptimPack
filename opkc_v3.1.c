// File indented with: indent -i4 -nut -br -ce -npcs NomFichier.c
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <optimpack.h>          // HERE we will implement the simple driver version of optimpack  l1560

#define TRUE  1
#define FALSE 0

// Destructor for cleaning up Point objects
void destructor_opk(PyObject * obj) {
    opk_optimizer_t *opt =
        (opk_optimizer_t *) PyCapsule_GetPointer(obj, "opkc_v3_1._optimizer");
    if (opt != NULL) {
        opk_destroy_optimizer(opt);
    }
    PyErr_Print();
}


static PyObject *
initialisation(PyObject * self, PyObject * args, PyObject * keywds)
{
    void *x;
    PyObject *x_obj = NULL;
    PyArrayObject *x_arr = NULL;

    PyObject *bound_up_obj = NULL;
    void *bound_up_arr = NULL;

    PyObject *bound_low_obj = NULL;
    void *bound_low_arr = NULL;

    char *algorithm_name = "vmlmb";
    int algorithm_method = OPK_ALGORITHM_VMLMB;

    // Preparing and settinf arguments and keywords
    static char *kwlist[] =
        { "x", "algorithm", "upper_bound", "lower_bound", NULL };

    if (!PyArg_ParseTupleAndKeywords
        (args, keywds, "O|sOO", kwlist, &x_obj, &algorithm_name,
         &bound_up_obj, &bound_low_obj)) {
        return NULL;
    }

    if (x_obj == NULL) {
        return NULL;
    }
    if (!PyArray_Check(x_obj)) {
        PyErr_SetString(PyExc_TypeError, "Input is not a scipy array object");
        return NULL;
    }
    // Is the array of type CFloat and has the good shape/DIM
    int single = (PyArray_TYPE((PyArrayObject *) x_obj) == NPY_FLOAT);
    // printf("Scalar: %d\n", PyArray_IsPythonScalar(x_obj));
    x_arr =
        (PyArrayObject *) PyArray_FROM_OT(x_obj,
                                          (single ? NPY_FLOAT : NPY_DOUBLE));
    if (x_arr == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(x_arr) != 1) {
        PyErr_SetString(PyExc_TypeError, "We need 1D arrays");
        return NULL;
    }
    npy_intp *shape = PyArray_DIMS(x_arr);
    int n = shape[0];

    // If everything is good gettinfg the data from the PyArray
    x = PyArray_DATA(x_arr);

    if (x == NULL) {
        return NULL;
    }
    int type = (single ? OPK_FLOAT : OPK_DOUBLE);

    // What is the algorithm we should use
    if (strcasecmp(algorithm_name, "vmlmb") == 0) {
        algorithm_method = OPK_ALGORITHM_VMLMB;
    } else if (strcasecmp(algorithm_name, "nlcg") == 0) {
        algorithm_method = OPK_ALGORITHM_NLCG;
    } else {
        PyErr_SetString(PyExc_TypeError, "Unknown algorithm method");
        return NULL;
    }
    // Testing Bounds
    opk_bound_type_t bound_up, bound_low;
    // Upper bound test
    if (bound_up_obj == NULL || bound_up_obj == Py_None) {
        bound_up = OPK_BOUND_NONE;
    } else if (PyFloat_Check(bound_up_obj) || PyLong_Check(bound_up_obj)) {
        bound_up = (single ? OPK_BOUND_SCALAR_FLOAT : OPK_BOUND_SCALAR_DOUBLE);
        double tmp_up = PyFloat_AsDouble(bound_up_obj);
        bound_up_arr = (void *) &tmp_up;        // Should work in float and double case ....
        /* This does not work, because it discards the information that the bounds are
        scalar, which results in treating the &tmp_up as an array, which in turn leads to
        reading garbage in the OptimPack code. Same in the second block below */
    } else if (PyArray_Check(bound_up_obj)) {
        bound_up = (single ? OPK_BOUND_STATIC_FLOAT : OPK_BOUND_STATIC_DOUBLE);
        PyArrayObject *bound_up_arr_tmp;
        bound_up_arr_tmp =
            (PyArrayObject *) PyArray_FROM_OT(bound_up_obj,
                                              (single ? NPY_FLOAT :
                                               NPY_DOUBLE));
        bound_up_arr = PyArray_DATA(bound_up_arr_tmp);
        if (bound_up_arr == NULL) {
            PyErr_SetString(PyExc_TypeError, "Failed to convert upper bound");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Unknown upper bound type");
        return NULL;
    }
    // Lower bound test
    if (bound_low_obj == NULL || bound_low_obj == Py_None) {
        bound_low = OPK_BOUND_NONE;
    } else if (PyFloat_Check(bound_low_obj) || PyLong_Check(bound_low_obj)) {
        bound_low = (single ? OPK_BOUND_STATIC_FLOAT : OPK_BOUND_SCALAR_DOUBLE);
        double tmp_low = PyFloat_AsDouble(bound_low_obj);
        bound_low_arr = (void *) &tmp_low;      // Should work in float and double case ....
    } else if (PyArray_Check(bound_low_obj)) {
        bound_low = (single ? OPK_BOUND_STATIC_FLOAT : OPK_BOUND_STATIC_DOUBLE);
        PyArrayObject *bound_low_arr_tmp;
        bound_low_arr_tmp =
            (PyArrayObject *) PyArray_FROM_OT(bound_low_obj,
                                              (single ? NPY_FLOAT :
                                               NPY_DOUBLE));
        bound_low_arr = PyArray_DATA(bound_low_arr_tmp);
        if (bound_low_arr == NULL) {
            PyErr_SetString(PyExc_TypeError, "Failed to convert lower bound");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Unknown lower bound type");
        return NULL;
    }
    opk_optimizer_t *opt = opk_new_optimizer(algorithm_method, NULL, type, n,
                                             bound_low, bound_low_arr,
                                             bound_up, bound_up_arr, NULL);
    if (opt == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to create optimizer");
        return NULL;
    }
    opk_task_t task = opk_start(opt, x);
    // We store in the python object the optimizer
    PyObject *c_api_object =
        PyCapsule_New((void *) opt, "opkc_v3_1._optimizer", destructor_opk);
    // and add it to the module
    int err = PyModule_AddObject(self, "_optimizer", c_api_object);
    err += PyModule_AddObject(self, "single", Py_BuildValue("i", single));
    if (err != 0) {
        return NULL;
    }
    // TEST
    PyObject_SetAttrString(self, "counter", Py_BuildValue("i", 0));
    return Py_BuildValue("i", task);
}

static PyObject *
iterate(PyObject * self, PyObject * args)
{
    // TEST
    PyObject *v = PyObject_GetAttrString(self, "counter");
    int foobar = PyLong_AsLong(v);
    PyObject_SetAttrString(self, "counter", Py_BuildValue("i", foobar + 1));

    // Testing inputs, type and converting them to their true type
    PyObject *x_obj = NULL;
    PyArrayObject *x_arr = NULL;
    double fx = 1;
    PyObject *g_obj = NULL;
    PyArrayObject *g_arr = NULL;
    void *x = NULL, *g = NULL;

    if (!PyArg_ParseTuple(args, "OdO", &x_obj, &fx, &g_obj)) {
        return NULL;
    }

    if (x_obj == NULL) {
        return NULL;
    }
    if (!PyArray_Check(x_obj)) {
        PyErr_SetString(PyExc_TypeError, "Input is not a scipy array object");
        return NULL;
    }
    int single = (PyArray_TYPE((PyArrayObject *) x_obj) == NPY_FLOAT);
    PyObject *single_obj = PyObject_GetAttrString(self, "single");
    if (((int) PyLong_AsLong(single_obj)) != single) {
        PyErr_SetString(PyExc_TypeError,
                        "Input is not the same type as initialization");
        return NULL;
    }
    if (single) {
        x_arr = (PyArrayObject *) PyArray_FROM_OT(x_obj, NPY_FLOAT);
        g_arr = (PyArrayObject *) PyArray_FROM_OT(g_obj, NPY_FLOAT);
    } else {
        x_arr = (PyArrayObject *) PyArray_FROM_OT(x_obj, NPY_DOUBLE);
        g_arr = (PyArrayObject *) PyArray_FROM_OT(g_obj, NPY_DOUBLE);
    }
    if (x_arr == NULL || g_arr == NULL) {
        return NULL;
    }
    x = PyArray_DATA(x_arr);
    g = PyArray_DATA(g_arr);

    if (x == NULL || g == NULL) {
        return NULL;
    }
    // npy_intp *shape = PyArray_DIMS(x_arr);
    // int n = shape[0];
    // int type = (single ? OPK_FLOAT : OPK_DOUBLE);
    // Getting the optimizer stored in the python object
    opk_optimizer_t *opt =
        (opk_optimizer_t *) PyCapsule_Import("opkc_v3_1._optimizer", 0);
    if (opt == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to import optimizer");
        return NULL;
    }
    opk_task_t task = opk_iterate(opt, x, fx, g);;
    return Py_BuildValue("i", task);
}

static PyObject *
opk_close(PyObject * self)
{

    opk_optimizer_t *opt =
        (opk_optimizer_t *) PyCapsule_Import("opkc_v3_1._optimizer", 0);
    if (opt != NULL) {
        opk_destroy_optimizer(opt);
    }
    PyErr_Print();
    Py_RETURN_NONE;
}

static PyObject *
get_gnorm(PyObject * self)
{
    double gnorm = 0.0;
    opk_optimizer_t *opt =
        (opk_optimizer_t *) PyCapsule_Import("opkc_v3_1._optimizer", 0);
    if (opt != NULL) {
        gnorm = opk_get_gnorm(opt);
    }
    return Py_BuildValue("d", gnorm);
}

static PyObject *
get_reason(PyObject * self)
{
    const char *reason;
    opk_optimizer_t *opt =
        (opk_optimizer_t *) PyCapsule_Import("opkc_v3_1._optimizer", 0);
    if (opt != NULL) {
        reason = opk_get_reason(opk_get_status(opt));
    } else {
        reason = "";
    }
    return Py_BuildValue("s", reason);
}

// link the function that will be seen by the user to the c code
static PyMethodDef Methods[] = {
    {"initialisation", (PyCFunction) initialisation,
     METH_VARARGS | METH_KEYWORDS, "Create the optimizer with the right inputs"},
    {"iterate", (PyCFunction) iterate, METH_VARARGS, "Once the function has been initialized, we can iterate"},
    // {"TaskInfo", (PyCFunction)TaskInfo, METH_VARARGS, "lala"},
    {"close", (PyCFunction) opk_close, METH_NOARGS, "Free optimizer and so memory"},
    {"get_gnorm", (PyCFunction) get_gnorm, METH_NOARGS, "getter"},
    {"get_reason", (PyCFunction) get_reason, METH_NOARGS, "getter"},
    {NULL, NULL, 0, NULL}
};



#if PY_MAJOR_VERSION >= 3   // Python 3+
    // module initialization
    static struct PyModuleDef optimpack_module =
        { PyModuleDef_HEAD_INIT, "opkc_v3_1", NULL, -1, Methods };

    PyMODINIT_FUNC
    PyInit_opkc_v3_1(void)
    {
        import_array();
        return PyModule_Create(&optimpack_module);
    }
#else                       // PYTHON2
    PyMODINIT_FUNC
    initopkc_v3_1(void)
    {
        (void) Py_InitModule("opkc_v3_1", Methods);
        import_array();
    }
#endif
