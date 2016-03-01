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


opk_optimizer_t *opt;

static PyObject* initialisation(PyObject * self, PyObject * args, PyObject * keywds)
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

    // opk_bound_t *lower;
    // opk_bound_t *upper;

    // Preparing and settinf arguments and keywords
    static char *kwlist[] = { "x", "algorithm", "upper_bound", "lower_bound", NULL };

    if (!PyArg_ParseTupleAndKeywords
        (args, keywds, "O|sOO", kwlist, &x_obj, &algorithm_name, &bound_up_obj, &bound_low_obj)) {
        return NULL;
    }

    if (x_obj == NULL) {
        return NULL;
    }
    // Is the array of type CFloat and has the good shape/DIM
    int single = PyArray_IsScalar(x_obj, CFloat);
    // printf("Scalar: %d\n", PyArray_IsPythonScalar(x_obj));
    if (single) {
        x_arr = (PyArrayObject*) PyArray_FROM_OT(x_obj, NPY_FLOAT);
    } else {
        x_arr = (PyArrayObject*) PyArray_FROM_OT(x_obj, NPY_DOUBLE);
    }
    if (x_arr == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(x_arr) != 1) {
        printf("We need 1D arrays");
        return NULL;
    }
    npy_intp *shape = PyArray_DIMS( x_arr);
    int n = shape[0];

    // If everything is good gettinfg the data from the PyArray
    x = PyArray_DATA( x_arr);

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
        printf("# Unknown algorithm method\n");
        return NULL;
    }

    // Testing Bounds
    opk_bound_type_t bound_up, bound_low;
     // Upper bound test
    if(bound_up_obj == NULL || bound_up_obj == Py_None) {
        bound_up = OPK_BOUND_NONE;
    } else if(PyFloat_Check(bound_up_obj)) {
        bound_up = OPK_BOUND_SCALAR;
        double tmp_up = PyFloat_AsDouble(bound_up_obj);
        bound_up_arr = (void*)&tmp_up;  // Should work in float and double case ....
    } else {
        bound_up = OPK_BOUND_VECTOR;
        PyArrayObject* bound_up_arr_tmp;
        if (single) {
            bound_up_arr_tmp = (PyArrayObject*) PyArray_FROM_OT(x_obj, NPY_FLOAT);
        } else {
            bound_up_arr_tmp = (PyArrayObject*) PyArray_FROM_OT(x_obj, NPY_DOUBLE);
        }
        bound_up_arr = PyArray_DATA(bound_up_arr_tmp);
    }
    // Lower bound test
    if(bound_low_obj == NULL || bound_low_obj == Py_None) {
        bound_low = OPK_BOUND_NONE;
    } else if(PyFloat_Check(bound_low_obj)) {
        bound_low = OPK_BOUND_SCALAR;
        double tmp_low = PyFloat_AsDouble(bound_low_obj);
        bound_low_arr = (void*)&tmp_low; // Should work in float and double case ....
    } else {
        bound_low = OPK_BOUND_VECTOR;
        PyArrayObject* bound_low_arr_tmp;
        if (single) {
            bound_low_arr_tmp = (PyArrayObject*) PyArray_FROM_OT(x_obj, NPY_FLOAT);
        } else {
            bound_low_arr_tmp = (PyArrayObject*) PyArray_FROM_OT(x_obj, NPY_DOUBLE);
        }
        bound_low_arr = PyArray_DATA(bound_low_arr_tmp);
    }
    printf("BOUNDS %d %d\n", bound_up, bound_low);
    opt = opk_new_optimizer(algorithm_method, type, n, 0, 0,
                                              bound_up, bound_up_arr,
                                              bound_low, bound_low_arr,
                                              NULL);
    opk_task_t task = opk_start(opt, type, n, x);
    return Py_BuildValue("i", task);
}

static PyObject* iterate(PyObject * self, PyObject * args)
{
    // Testing inputs, type and converting them to their true type
    PyObject *x_obj = NULL;
    PyArrayObject* x_arr = NULL;
    double fx = 1;
    PyObject *g_obj = NULL;
    PyArrayObject* g_arr = NULL;
    void *x = NULL, *g = NULL;

    if (!PyArg_ParseTuple(args, "OdO", &x_obj, &fx, &g_obj)) {
        return NULL;
    }

    if (x_obj == NULL) {
        return NULL;
    }
    int single = PyArray_IsScalar(x_obj, CFloat);
    if (single == OPK_DOUBLE) {
        x_arr = (PyArrayObject*) PyArray_FROM_OT(x_obj, NPY_DOUBLE);
        g_arr = (PyArrayObject*) PyArray_FROM_OT(g_obj, NPY_DOUBLE);
    } else {
        x_arr = (PyArrayObject*) PyArray_FROM_OT(x_obj, NPY_FLOAT);
        g_arr = (PyArrayObject*) PyArray_FROM_OT(g_obj, NPY_FLOAT);
    }
    if (x_arr == NULL || g_arr == NULL) {
        return NULL;
    }
    x = PyArray_DATA( x_arr);
    g = PyArray_DATA( g_arr);

    if (x == NULL || g == NULL) {
        return NULL;
    }
    npy_intp *shape = PyArray_DIMS( x_arr);
    int n = shape[0];
    int type = (single ? OPK_FLOAT : OPK_DOUBLE);
    opk_task_t task = opk_iterate(opt, type, n, x, fx, g);
    return Py_BuildValue("i", task);
}

static PyObject* opk_close(PyObject * self)
{
    if(opt != NULL) {
        opk_destroy_optimizer(opt);
    }
    Py_RETURN_NONE;
}

static PyObject* get_gnorm(PyObject * self) {
    double gnorm = 0.0;
    if(opt != NULL) {
        gnorm = opk_get_gnorm(opt);
    }
    return Py_BuildValue("d", gnorm);
}

static PyObject* get_reason(PyObject * self) {
    const char *reason;
    if(opt != NULL) {
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

// module initialization
static struct PyModuleDef optimpack_module =
    { PyModuleDef_HEAD_INIT, "opkc_v3_1", NULL, -1, Methods };

#if PY_MAJOR_VERSION >= 3   // Python 3+
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
