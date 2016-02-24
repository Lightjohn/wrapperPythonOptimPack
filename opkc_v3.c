// File indented with: indent -i4 -nut -br -ce -npcs NomFichier.c
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <optimpack.h>


/*  -------------------------- DEFINTION OF GLOBAL VARIABLES --------------------------- */
// Creation of vspace
opk_vspace_t *_vspace;

// vspace vectors: they keep on changing but are declared only once.
opk_vector_t *_vgp;                     // Projected Gradient Vector
opk_vector_t *_vx;                      // Current point
opk_vector_t *_vg;                      // Current point gradient

// The task: it is always changing and tells the algorithm what must be done next.
opk_task_t _task;

// Optimizers: Contain the information of the initialiation step 
const char *_algorithm_name = "vmlmb"; 	// Name of the algorithm
int _limited = 0;			// whether it uses a limited memory minimization process
opk_nlcg_t *_nlcg;        		// non-linear conjugate gradient optimizer 
opk_vmlmb_t *_vmlmb;     	        // quasi-newton optimizer
opk_optimizer_t *_limited_optimizer;    // limited memory optimizer

// Options
opk_nlcg_options_t _options_nlcg;	// Options relative to the nlcg optimizer
opk_vmlmb_options_t _options_vmlmb;     // Options relative to the vmlmb optimizer

// Type and size
opk_type_t _type = OPK_DOUBLE;		// Type of the variables (x, f and g). Double or float
opk_index_t _problem_size;		// Number of variables of the problem


/* ------------------------------------------------------------------------------------------
------------------------------- FONCTION INITIALISATION -------------------------------------
--------------------------------------------------------------------------------------------- */
static PyObject *
opk_initialisation(PyObject * self, PyObject * args, PyObject * keywds)
{
    //  ------------------------------------- DECLARATIONS ----------------------------------
    #define TRUE  1
    #define FALSE 0

    // Parameters of the minimization problem. They will receive values according to what the user asks
    opk_lnsrch_t *lnsrch = NULL;
    int autostep = 0;
    unsigned int nlcg_method = OPK_NLCG_DEFAULT;
    unsigned int vmlmb_method = OPK_LBFGS;    
    unsigned int limited_method = nlcg_method;

    // Input arguments in c
    void *x;
    char *linesrch_name = "quadratic";
    char *autostep_name = "ShannoPhua";
    char *nlcgmeth_name = "FletcherReeves";
    char *vmlmbmeth_name = "OPK_LBFGS";
    double delta;               // No need for default value
    double epsilon;             // No need for default value
    int delta_given;            // No need for default value
    int epsilon_given;          // No need for default value
    double gatol = 1.0e-6;
    double grtol = 0.0;
    double bl = 0;              // Value if user doesn't specifie anything and algorithm is vmlmb
    double bu = 1e6;            // Value if user doesn't specifie anything and algorithm is vmlmb
    int bound_given = 0;        // No Default boundries
    int mem = 5;                // Value if user doesn't specifie anything and algorithm is vmlmb
    int powell = FALSE;
    int single = FALSE;

    // Input x as a PyObject
    PyObject *x_obj = NULL;
    PyObject *x_arr = NULL;

    // Other local variables
    opk_vector_t *vbl;          // bounds       
    opk_vector_t *vbu;          // bounds
    double sftol = 1e-4;        // Values used in linesearch setting
    double sgtol = 0.9;         // Values used in linesearch setting
    double sxtol = 1e-15;       // Values used in linesearch setting
    double samin = 0.1;         // Values used in linesearch setting           
    opk_bound_t *lower;         // bounds   
    opk_bound_t *upper;         // bounds   

    // Returned value
    PyObject *task_py;

    // Print the error messages in an external file "text.txt"
    FILE* fichier = NULL;
    fichier = fopen("text.txt", "w");
    if (fichier == NULL)
    {
	return Py_None;
    }
    //  -------------------------------------------------------------------------------------


    //  --------------------- PYTHON OBJECT ARE CONVERTED TO C OBJECT ----------------------- 
    /*
     i = int
     s = string
     f = float
     d = double
     p = predicate (bool) = int
     | = other arguments are optionals
    */
    static char *kwlist[] =
        { "x", "algorithm", "linesearch", "autostep", "_nlcg", "_vmlmb",
        "delta", "epsilon", "delta_given", "epsilon_given", "gatol", "grtol",
            "bl", "bu",
        "bound_given", "mem", "powell", "single", "_limited", NULL
    };

    if (!PyArg_ParseTupleAndKeywords
        (args, keywds, "Osssssddiiddddiiiii", kwlist, &x_obj, &_algorithm_name,
         &linesrch_name, &autostep_name, &nlcgmeth_name, &vmlmbmeth_name,
         &delta, &epsilon, &delta_given, &epsilon_given, &gatol, &grtol, &bl,
         &bu, &bound_given, &mem, &powell, &single, &_limited)) {
        return NULL;
    }
    //  -------------------------------------------------------------------------------------


    //  ------------------------------ VSPACE IS DESCRIBED VIA X ---------------------------- 
    if (x_obj == NULL) {
        return NULL;
    }
    // FIXME get single value trough  PyArray_IsPythonNumber(NPY_FLOAT), do not ask user for this value
    if (single == FALSE) {
        x_arr = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    } else {
        x_arr = PyArray_FROM_OTF(x_obj, NPY_FLOAT, NPY_IN_ARRAY);
    }
    if (x_arr == NULL) {
        return NULL;
    }

    int nd = PyArray_NDIM(x_arr);
    npy_intp *shape = PyArray_DIMS(x_arr);
    _problem_size = shape[0];
    if (nd != 1) {
        printf("We need 1D arrays");
        return NULL;
    }
    if (single == FALSE) {
        x = (double *) PyArray_DATA(x_arr);
    } else {
        x = (float *) PyArray_DATA(x_arr);
    }

    if (x == NULL) {
        return NULL;
    }
    //  -------------------------------------------------------------------------------------


    //  ------------------- SETTING OF LNSRCH, AUTOSTEP, NLCG AND VMLMB ---------------------
    // linesearch
    if (strcasecmp(linesrch_name, "quadratic") == 0) {
        lnsrch = opk_lnsrch_new_backtrack(sftol, samin);
    } else if (strcasecmp(linesrch_name, "Armijo") == 0) {
        lnsrch = opk_lnsrch_new_backtrack(sftol, 0.5);
    } else if (strcasecmp(linesrch_name, "cubic") == 0) {
        lnsrch = opk_lnsrch_new_csrch(sftol, sgtol, sxtol);
    } else if (strcasecmp(linesrch_name, "nonmonotone") == 0) {
        lnsrch = opk_lnsrch_new_nonmonotone(mem, 1e-4, 0.1, 0.9);
    } else {
        printf("# Unknown line search method\n");
        lnsrch = NULL;
        return NULL;
    }

    // autstep
    if (strcasecmp(autostep_name, "ShannoPhua") == 0) {
        autostep = OPK_NLCG_SHANNO_PHUA;
    }

    /*
    else if (strcasecmp (autostep_name, "OrenSpedicato") == 0) 
        {autostep = OPK_NLCG_OREN_SPEDICATO;}
    else if (strcasecmp (autostep_name, "BarzilaiBorwein") == 0) 
        {autostep = OPK_NLCG_BARZILAI_BORWEIN;}
    */
    else {
        printf("# Unknown line search method\n");
        return NULL;
    }

    // nlcg method
    if (strcasecmp(nlcgmeth_name, "FletcherReeves") == 0) {
        nlcg_method = OPK_NLCG_FLETCHER_REEVES;
    } else if (strcasecmp(nlcgmeth_name, "HestenesStiefel") == 0) {
        nlcg_method = OPK_NLCG_HESTENES_STIEFEL;
    } else if (strcasecmp(nlcgmeth_name, "PolakRibierePolyak") == 0) {
        nlcg_method = OPK_NLCG_POLAK_RIBIERE_POLYAK;
    } else if (strcasecmp(nlcgmeth_name, "Fletcher") == 0) {
        nlcg_method = OPK_NLCG_FLETCHER;
    } else if (strcasecmp(nlcgmeth_name, "LiuStorey") == 0) {
        nlcg_method = OPK_NLCG_LIU_STOREY;
    } else if (strcasecmp(nlcgmeth_name, "DaiYuan") == 0) {
        nlcg_method = OPK_NLCG_DAI_YUAN;
    } else if (strcasecmp(nlcgmeth_name, "PerryShanno") == 0) {
        nlcg_method = OPK_NLCG_PERRY_SHANNO;
    } else if (strcasecmp(nlcgmeth_name, "HagerZhang") == 0) {
        nlcg_method = OPK_NLCG_HAGER_ZHANG;
    } else {
        printf("# Unknown line search method for nlcg\n");
        return NULL;
    }

    // vmlmb method
    if (strcasecmp(vmlmbmeth_name, "blmvm") == 0) {
        vmlmb_method = OPK_BLMVM;
    } else if (strcasecmp(vmlmbmeth_name, "vmlmb") == 0) {
        vmlmb_method = OPK_VMLMB;
    } else if (strcasecmp(vmlmbmeth_name, "lbfgs") == 0) {
        vmlmb_method = OPK_LBFGS;
    } else {
        printf("# Unknown line search method for vmlmb\n");
        return NULL;
    }

    // Type of the variable
    if (single == TRUE) {
        _type = OPK_FLOAT;
    } else {
        _type = OPK_DOUBLE;
    }

    //  -------------------------------------------------------------------------------------


    //  ----------------------------- BIT TO BIT COMPARISON ---------------------------------
    if (powell) {
        nlcg_method |= OPK_NLCG_POWELL;
    }
    if (autostep != 0) {
        nlcg_method |= autostep;
    }
    //  -------------------------------------------------------------------------------------


    //  --------------------------------- CONSTRUCTION OF VSPACE ---------------------------- 
    // _vspace only relies on the size of x
    _vspace = opk_new_simple_double_vector_space(_problem_size);
    if (_vspace == NULL) {
        printf("# Failed to allocate vector space\n");
        return NULL;
    }
    // x is transformed into a vector of the new space vector vspace
    // Should be free instead of NULL
    _vx = opk_wrap_simple_double_vector(_vspace, x, NULL, x);     
    if (_vx == NULL) {
        printf("# Failed to wrap vectors\n");
        return NULL;
    }
    //  -------------------------------------------------------------------------------------


    //  ----------------------------- CONSTRUCTION OF THE BOUNDS ---------------------------- 
    // These are declared anyway because they are dropped later on (avoids a loop)
    if (single == TRUE) {
        bl = (float) (bl);
        bu = (float) (bu);
    }
    vbl = opk_wrap_simple_double_vector(_vspace, &bl, NULL, &bl);
    vbu = opk_wrap_simple_double_vector(_vspace, &bu, NULL, &bu);
    // If used asked for bounds: 1 = lower, 2 = upper, 3 = both
    if (bound_given == 1) {
        lower = opk_new_bound(_vspace, OPK_BOUND_VECTOR, vbl);
        if (lower == NULL) {
            printf("# Failed to wrap lower bounds\n");
            return NULL;
        }
        upper = NULL;
    } else if (bound_given == 2) {
        upper = opk_new_bound(_vspace, OPK_BOUND_VECTOR, vbu);
        if (upper == NULL) {
            printf("# Failed to wrap upper bounds\n");
            return NULL;
        }
        lower = NULL;
    } else if (bound_given == 3) {
        lower = opk_new_bound(_vspace, OPK_BOUND_VECTOR, vbl);
        upper = opk_new_bound(_vspace, OPK_BOUND_VECTOR, vbu);
        if (lower == NULL) {
            printf("# Failed to wrap lower bounds\n");
            return NULL;
        }
        if (upper == NULL) {
            printf("# Failed to wrap upper bounds\n");
            return NULL;
        }
    } else {
        lower = NULL;
        upper = NULL;
    }
    //  -------------------------------------------------------------------------------------


    //  ------------------------------ CREATION OF THE OPTIMIZER ---------------------------- 
    // The optimizer and task are created depending on the chosen algorithm.
    if (_limited == 0) {

        // VMLMB --------------------------------
        if (strcasecmp(_algorithm_name, "vmlmb") == 0) {
            // Creation of the optimizer
            _vmlmb =
                opk_new_vmlmb_optimizer(_vspace, mem, vmlmb_method, lower,
                                        upper, lnsrch);
            if (_vmlmb == NULL) {
                printf("# Failed to create VMLMB optimizer\n");
                return NULL;
            }
            // creation of _vgp
            _vgp = opk_vcreate(_vspace);
            if (_vgp == NULL) {
                printf("# Failed to create projected gradient vector\n");
                return NULL;
            }
            // option.gatol = gatol? Otherwise the received value "gatol" is lost?
            opk_get_vmlmb_options(&_options_vmlmb, _vmlmb);
            _options_vmlmb.gatol = grtol;
            _options_vmlmb.grtol = gatol;
            if (delta_given) {
                _options_vmlmb.delta = delta;
            }
            if (epsilon_given) {
                _options_vmlmb.epsilon = epsilon;
            }
            if (opk_set_vmlmb_options(_vmlmb, &_options_vmlmb) != OPK_SUCCESS) {
                printf("# Bad VMLMB options\n");
                return NULL;
            }
            //_task = opk_start_vmlmb (_vmlmb, _vx);

        }
        // NLCG ---------------------------------  
        else if (strcasecmp(_algorithm_name, "nlcg") == 0) {
            _nlcg = opk_new_nlcg_optimizer(_vspace, nlcg_method, lnsrch);
            if (_nlcg == NULL) {
                printf("# Failed to create NLCG optimizer\n");
                return NULL;
            }
            opk_get_nlcg_options(&_options_nlcg, _nlcg);
            _options_nlcg.gatol = gatol;
            _options_nlcg.grtol = grtol;
            if (delta_given) {
                _options_nlcg.delta = delta;
            }
            if (epsilon_given) {
                _options_nlcg.epsilon = epsilon;
            }
            if (opk_set_nlcg_options(_nlcg, &_options_nlcg) != OPK_SUCCESS) {
                printf("# Bad NLCG options\n");
                return NULL;
            }
            _task = opk_start_nlcg(_nlcg, _vx);
        }

        else {
            printf("# Bad algorithm\n");
            return NULL;
        }
    }
    // Limited Memory ---------------------------------  

    else {
        opk_algorithm_t limited_algorithm;
        if (strcasecmp(_algorithm_name, "nlcg") == 0) {
            limited_algorithm = OPK_ALGORITHM_NLCG;
            limited_method = nlcg_method;
        } else if (strcasecmp(_algorithm_name, "vmlmb") == 0) {
            limited_algorithm = OPK_ALGORITHM_VMLMB;
            limited_method = vmlmb_method;
        } else {
            printf("# Bad algorithm\n");
            return NULL;
        }
        // optimizer
        if (bound_given == 1) {
            _limited_optimizer =
                opk_new_optimizer(limited_algorithm, _type, shape[0], mem,
                                  limited_method, _type, &bl, OPK_BOUND_NONE,
                                  NULL, lnsrch);
        } else if (bound_given == 2) {
            _limited_optimizer =
                opk_new_optimizer(limited_algorithm, _type, shape[0], mem,
                                  limited_method, OPK_BOUND_NONE, NULL, _type,
                                  &bu, lnsrch);
        } else if (bound_given == 3) {
            _limited_optimizer =
                opk_new_optimizer(limited_algorithm, _type, shape[0], mem,
                                  limited_method, _type, &bl, _type, &bu,
                                  lnsrch);
        } else {
            _limited_optimizer =
                opk_new_optimizer(limited_algorithm, _type, shape[0], mem,
                                  limited_method, OPK_BOUND_NONE, NULL,
                                  OPK_BOUND_NONE, NULL, lnsrch);
        }

        if (_limited_optimizer == NULL) {
            printf("# Failed to create limited optimizer\n");
            return NULL;
        }

        _task = opk_start(_limited_optimizer, _type, shape[0], x);
    }

    // Free workspace 
    OPK_DROP(vbl);
    OPK_DROP(vbu);
    fclose(fichier);

    // Return value is OPK_TASK_COMPUTE_FG
    if (_task == OPK_TASK_COMPUTE_FG) {
        task_py = Py_BuildValue("s", "OPK_TASK_COMPUTE_FG");
    } else if (_task == OPK_TASK_START) {
        task_py = Py_BuildValue("s", "OPK_TASK_START");
    } else if (_task == OPK_TASK_NEW_X) {
        task_py = Py_BuildValue("s", "OPK_TASK_NEW_X");
    } else if (_task == OPK_TASK_FINAL_X) {
        task_py = Py_BuildValue("s", "OPK_TASK_FINAL_X");
    } else if (_task == OPK_TASK_WARNING) {
        task_py = Py_BuildValue("s", "OPK_TASK_WARNING");
    } else if (_task == OPK_TASK_ERROR) {
        task_py = Py_BuildValue("s", "OPK_TASK_ERROR");
    }

    else {
        task_py = Py_BuildValue("s", "INITIALIZATION_ERROR");
    }
    return task_py;
}
// ------------------------------------------------------------------------------------------ 


/* ------------------------------------------------------------------------------------------
----------------------------------- FONCTION ITERATE ----------------------------------------
--------------------------------------------------------------------------------------------- */
static PyObject *
opk_iteration(PyObject * self, PyObject * args)
{
    // Input arguments
    PyObject *x_obj = NULL, *x_arr = NULL;
    double f_c;
    PyObject *g_obj, *g_arr;
    // Output arguments
    opk_task_t task_locale = OPK_TASK_ERROR;
    char *task_c;
    // Other arguments
    void *x, *g;

    // Conversion depending on the type
    if (_type == OPK_DOUBLE) {
        if (!PyArg_ParseTuple(args, "OdO", &x_obj, &f_c, &g_obj)) {
            return NULL;
        }
    } else {
        f_c = (float) (f_c);
        if (!PyArg_ParseTuple(args, "OfO", &x_obj, &f_c, &g_obj)) {
            return NULL;
        }
    }

    // Values are passed to x, f and g depending on their type
    if (x_obj == NULL) {
        return NULL;
    }
    if (_type == OPK_DOUBLE) {
        x_arr = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
        g_arr = PyArray_FROM_OTF(g_obj, NPY_DOUBLE, NPY_IN_ARRAY);
        x = (double *) PyArray_DATA(x_arr);
        g = (double *) PyArray_DATA(g_arr);
    } else {
        x_arr = PyArray_FROM_OTF(x_obj, NPY_FLOAT, NPY_IN_ARRAY);
        g_arr = PyArray_FROM_OTF(g_obj, NPY_FLOAT, NPY_IN_ARRAY);
        x = (float *) PyArray_DATA(x_arr);
        g = (float *) PyArray_DATA(g_arr);
    }
    if (x_arr == NULL || g_arr == NULL) {
        return NULL;
    }
    if (x == NULL || g == NULL) {
        return NULL;
    }
    // x and g are converted to vspace vectors
    _vx = opk_wrap_simple_double_vector(_vspace, x, NULL, x);
    _vg = opk_wrap_simple_double_vector(_vspace, g, NULL, g);

    // iterate is called
    if (_limited == 0) {
        if (strcasecmp(_algorithm_name, "nlcg") == 0) {
            task_locale = opk_iterate_nlcg(_nlcg, _vx, f_c, _vg);
        } else if (strcasecmp(_algorithm_name, "vmlmb") == 0) {
            task_locale = opk_iterate_vmlmb(_vmlmb, _vx, f_c, _vg);
        }
    } else {
        task_locale =
            opk_iterate(_limited_optimizer, _type, _problem_size, x, f_c, g);
    }

    // The value is converted to a char string to be transfered in python
    if (task_locale == OPK_TASK_START) {
        task_c = "OPK_TASK_START";
    } else if (task_locale == OPK_TASK_COMPUTE_FG) {
        task_c = "OPK_TASK_COMPUTE_FG";
    } else if (task_locale == OPK_TASK_NEW_X) {
        task_c = "OPK_TASK_NEW_X";
    } else if (task_locale == OPK_TASK_FINAL_X) {
        task_c = "OPK_TASK_FINAL_X";
    } else if (task_locale == OPK_TASK_WARNING) {
        task_c = "OPK_TASK_WARNING";
    } else if (task_locale == OPK_TASK_ERROR) {
        task_c = "OPK_TASK_ERROR";
    } else {
        task_c = "OPK_TASK_UNKNOWN";
    }

    // task value is returned
    return Py_BuildValue("s", task_c);
}
// ------------------------------------------------------------------------------------------ 


/* ------------------------------------------------------------------------------------------
------------------------------------- FONCTION TASKINFO -------------------------------------
--------------------------------------------------------------------------------------------- */
static PyObject *
opk_taskInfo (PyObject * self, PyObject * args)
{
    // Input arguments
    char *QuelleFonction;
    // Output arguments
    char * return_c = "FAILURE : No value returned for TaskInfo";
    // Output arguments	is their local type		
    opk_task_t local_task=OPK_TASK_WARNING;
    opk_status_t local_status=OPK_TASK_WARNING;
    opk_index_t local_iteration = -1;
    opk_index_t local_evaluation = -1;
    opk_index_t local_restart = -1;
    char local_reason[80];
   // double local_step = -1;
   // opk_status_t local_get_options;
   // opk_status_t local_set_options;   
    // Other arguments
    double Value_d = -1;
    char Value_c[80];

    // If need be, required values can be printed in "text.txt"
    FILE* fichier = NULL;
    fichier = fopen("text.txt", "w");
    if (fichier == NULL)
    {
	return Py_None;
    }
    fprintf(fichier, "valeur de limited %i \n", _limited);

    // Conversion
    if (!PyArg_ParseTuple (args, "s",&QuelleFonction))
       {return NULL;}


    // TASK ------------ 
    if (strcmp(QuelleFonction, "Get_task") == 0)
    {
        // The asked value is stored in the appropriate variable
	if (_limited == 0)
	{
	     if (strcasecmp (_algorithm_name, "nlcg") == 0) 
	         {local_task = opk_get_nlcg_task(_nlcg);}
	     else if (strcasecmp (_algorithm_name, "vmlmb") == 0) 
	         {local_task = opk_get_vmlmb_task(_vmlmb);}
	}
	else
	    {local_task = opk_get_task(_limited_optimizer);}

        // Then it is converted in char string to be returned
        if (local_task == OPK_TASK_START)
    	    {return_c = "OPK_TASK_START";}   
        else if (local_task == OPK_TASK_COMPUTE_FG)
   	    {return_c = "OPK_TASK_COMPUTE_FG";}
        else if (local_task == OPK_TASK_NEW_X)
    	    {return_c = "OPK_TASK_NEW_X";}
        else if (local_task == OPK_TASK_FINAL_X)
    	    {return_c = "OPK_TASK_FINAL_X";}
        else if (local_task == OPK_TASK_WARNING)
    	    {return_c = "OPK_TASK_WARNING";}
        else if (local_task == OPK_TASK_ERROR)
    	    {return_c = "OPK_TASK_ERROR";}
        else
    	    {return_c = "OPK_GET_TASK_FAILURE";}
    }

    // STATUS ------------ 
    else if (strcmp(QuelleFonction, "Get_status") == 0)
    {
	if (_limited == 0)
	{
	     if (strcasecmp (_algorithm_name, "nlcg") == 0) 
	         {local_status = opk_get_nlcg_status(_nlcg);}
	     else if (strcasecmp (_algorithm_name, "vmlmb") == 0) 
	         {local_status = opk_get_vmlmb_status(_vmlmb);}
	}
	else
	    {local_status = opk_get_status(_limited_optimizer);}
        // Case might be more subtle here, but consistency won't be achieved
        if (local_status == OPK_SUCCESS)
    	    {return_c = "OPK_SUCCESS";}   
	else if (local_status == OPK_INVALID_ARGUMENT)
	    {return_c = "OPK_INVALID_ARGUMENT";}
	else if (local_status == OPK_INSUFFICIENT_MEMORY)
	    {return_c = "OPK_INSUFFICIENT_MEMORY";}
	else if (local_status == OPK_ILLEGAL_ADDRESS)
	    {return_c = "OPK_ILLEGAL_ADDRESS";}
	else if (local_status == OPK_NOT_IMPLEMENTED)
	    {return_c = "OPK_NOT_IMPLEMENTED";}
	else if (local_status == OPK_CORRUPTED_WORKSPACE)
	    {return_c = "OPK_CORRUPTED_WORKSPACE";}
	else if (local_status == OPK_BAD_SPACE)
	    {return_c = "OPK_BAD_SPACE";}
	else if (local_status == OPK_OUT_OF_BOUNDS_INDEX)
	    {return_c = "OPK_OUT_OF_BOUNDS_INDEX";}
	else if (local_status == OPK_NOT_STARTED)
	    {return_c = "OPK_NOT_STARTED";}
	else if (local_status == OPK_NOT_A_DESCENT)
	    {return_c = "OPK_NOT_A_DESCENT";}
	else if (local_status == OPK_STEP_CHANGED)
	    {return_c = "OPK_STEP_CHANGED";}
	else if (local_status == OPK_STEP_OUTSIDE_BRACKET)
	    {return_c = "OPK_STEP_OUTSIDE_BRACKET";}
	else if (local_status == OPK_STPMIN_GT_STPMAX)
	    {return_c = "OPK_STPMIN_GT_STPMAX";}
	else if (local_status == OPK_STPMIN_LT_ZERO)
	    {return_c = "OPK_STPMIN_LT_ZERO";}
	else if (local_status == OPK_STEP_LT_STPMIN)
	    {return_c = "OPK_STEP_LT_STPMIN";}
	else if (local_status == OPK_STEP_GT_STPMAX)
	    {return_c = "OPK_STEP_GT_STPMAX";}
	else if (local_status == OPK_FTOL_TEST_SATISFIED)
	    {return_c = "OPK_FTOL_TEST_SATISFIED";}
	else if (local_status == OPK_GTOL_TEST_SATISFIED)
	    {return_c = "OPK_GTOL_TEST_SATISFIED";}
	else if (local_status == OPK_XTOL_TEST_SATISFIED)
	    {return_c = "OPK_XTOL_TEST_SATISFIED";}
	else if (local_status == OPK_STEP_EQ_STPMAX)
	    {return_c = "OPK_STEP_EQ_STPMAX";}
	else if (local_status == OPK_STEP_EQ_STPMIN)
	    {return_c = "OPK_STEP_EQ_STPMIN";}
	else if (local_status == OPK_ROUNDING_ERRORS_PREVENT_PROGRESS)
	    {return_c = "OPK_ROUNDING_ERRORS_PREVENT_PROGRESS";}
	else if (local_status == OPK_NOT_POSITIVE_DEFINITE)
	    {return_c = "OPK_NOT_POSITIVE_DEFINITE";}
	else if (local_status == OPK_BAD_PRECONDITIONER)
	    {return_c = "OPK_BAD_PRECONDITIONER";}
	else if (local_status == OPK_INFEASIBLE_BOUNDS)
	    {return_c = "OPK_INFEASIBLE_BOUNDS";}
	else if (local_status == OPK_WOULD_BLOCK)
	    {return_c = "OPK_WOULD_BLOCK";}
	else if (local_status == OPK_UNDEFINED_VALUE)
	    {return_c = "OPK_UNDEFINED_VALUE";}
	else if (local_status == OPK_TOO_MANY_EVALUATIONS)
	    {return_c = "OPK_TOO_MANY_EVALUATIONS";}
	else if (local_status == OPK_TOO_MANY_ITERATIONS)
	    {return_c = "OPK_TOO_MANY_ITERATIONS";}
        else
    	    {return_c = "OPK_GET_STATUS_FAILURE";}
    }

    // ITERATION ------------ 
    else if (strcmp(QuelleFonction, "Get_iterations") == 0)
    {
	if (_limited == 0)
	{
	     if (strcasecmp (_algorithm_name, "nlcg") == 0) 
	         {local_iteration = opk_get_nlcg_iterations(_nlcg);}
	     else if (strcasecmp (_algorithm_name, "vmlmb") == 0) 
	         {local_iteration = opk_get_vmlmb_iterations(_vmlmb);}
	}
	else
	    {local_iteration = opk_get_iterations(_limited_optimizer);}
        if (local_iteration != -1)
	{
	    Value_d = local_iteration;
            sprintf(Value_c, "%lf", Value_d);
	    return_c = Value_c;
	} 
        else
    	    {return_c = "OPK_GET_ITERATION_FAILURE";}
    }

    // EVALUATION ------------ 
    if (strcmp(QuelleFonction, "Get_evaluations") == 0)
    {
	if (_limited == 0)
	{
	     if (strcasecmp (_algorithm_name, "nlcg") == 0) 
	         {local_evaluation = opk_get_nlcg_evaluations(_nlcg);}
	     else if (strcasecmp (_algorithm_name, "vmlmb") == 0) 
	         {local_evaluation = opk_get_vmlmb_evaluations(_vmlmb);}
	}
	else
	    {local_evaluation = opk_get_evaluations(_limited_optimizer);}

        if (local_evaluation != -1)
	{
	    Value_d = local_evaluation;
            sprintf(Value_c, "%lf", Value_d);
	    return_c = Value_c;
	}  
        else
    	    {return_c = "OPK_GET_EVALUATION_FAILURE";}    
    }

    // RESTART ------------ 
    if (strcmp(QuelleFonction, "Get_restarts") == 0)
    {
	if (_limited == 0)
	{
	     if (strcasecmp (_algorithm_name, "nlcg") == 0) 
	         {local_restart = opk_get_nlcg_restarts(_nlcg);}
	     else if (strcasecmp (_algorithm_name, "vmlmb") == 0) 
	         {local_restart = opk_get_vmlmb_restarts(_vmlmb);}
	}
	else
	    {local_restart = opk_get_restarts(_limited_optimizer);}
 
        if (local_restart != -1)
	{
	    Value_d = local_restart;
            sprintf(Value_c, "%lf", Value_d);
	    return_c = Value_c;
	}  
        else
    	    {return_c = "OPK_GET_RESTART_FAILURE";} 
    }

    // REASON ------------ 
    if (strcmp(QuelleFonction, "Get_reason") == 0)
    {
	if (_limited == 0)
	    {sprintf(local_reason, "REASON_IRRELEVANT_FOR_NON_LIMITED_MINIMIZATION");}
	else
	{
	    sprintf(local_reason, "%s",opk_get_reason(opk_get_status(_limited_optimizer)));
	}
	return_c = local_reason;
    }
/*
// ------------ STEP
    else if (strcmp(QuelleFonction, "Get_step") == 0)
    {
 	if (strcasecmp (_algorithm_name, "nlcg") == 0) 
	    {local_step = opk_get_nlcg_step(nlcg);}
	else if (strcasecmp (_algorithm_name, "lbfgs") == 0)
	    {local_step = opk_get_lbfgs_step(lbfgs);}
	else if (strcasecmp (_algorithm_name, "vmlm") == 0)
	    {local_step = opk_get_vmlm_step(vmlm);}
	else if (strcasecmp (_algorithm_name, "vmlmn") == 0)
	    {local_step = opk_get_vmlmn_step(vmlmn);}
	else 
	    {local_step = opk_get_vmlmb_step(vmlmb);}
        if (local_step != -1)
	{
            sprintf(Value_c, "%lf", local_step);
	    return_c = Value_c;
	}   
        else
    	    {return_c = "OPK_GET_STEP_FAILURE";} 
    }
*/

/* // ------------ GET_OPTIONS
    else if (strcmp(QuelleFonction, "Get_options") == 0)
    {
 	if (strcasecmp (_algorithm_name, "nlcg") == 0) 
	    {local_step = opk_get_nlcg_options(nlcg);}
	else if (strcasecmp (_algorithm_name, "lbfgs") == 0)
	    {local_step = opk_get_lbfgs_options(lbfgs);}
	else if (strcasecmp (_algorithm_name, "vmlm") == 0)
	    {local_step = opk_get_vmlm_options(vmlm);}
	else if (strcasecmp (_algorithm_name, "vmlmn") == 0)
	    {local_step = opk_get_vmlmn_options(vmlmn);}
	else 
	    {local_step = opk_get_vmlmb_options(vmlmb);}
        if (local_step != -1)
	{
            sprintf(Value_c, "%lf", local_step);
	    return_c = Value_c;
	}   
        else
    	    {return_c = "OPK_GET_STEP_FAILURE";} 
    }
// ------------ SET_OPTIONS
    else if (strcmp(QuelleFonction, "Set_options") == 0)
    {
 	if (strcasecmp (_algorithm_name, "nlcg") == 0) 
	    {local_step = opk_set_nlcg_options(nlcg);}
	else if (strcasecmp (_algorithm_name, "lbfgs") == 0)
	    {local_step = opk_set_lbfgs_options(lbfgs);}
	else if (strcasecmp (_algorithm_name, "vmlm") == 0)
	    {local_step = opk_set_vmlm_options(vmlm);}
	else if (strcasecmp (_algorithm_name, "vmlmn") == 0)
	    {local_step = opk_set_vmlmn_options(vmlmn);}
	else 
	    {local_step = opk_set_vmlmb_options(vmlmb);}
        if (local_step != -1)
	{
            sprintf(Value_c, "%lf", local_step);
	    return_c = Value_c;
	}   
        else
    	    {return_c = "OPK_GET_STEP_FAILURE";} 
    }
 // ------------ s
    else if (strcmp(QuelleFonction, "Get_s") == 0)
    {
	if (strcasecmp (_algorithm_name, "vmlmn") != 0)
	    {return_c = "ERROR : Get_s is relevant only with vmlmn algorithm";}
	else 
	    {local_vmlmn = opk_get_vmlmn_s(vmlmb, k);}

        if (local_vmlmn == OPK_SUCCESS)
    	    {return_c = "OPK_SUCCESS";}   
        else
    	    {return_c = "OPK_GET_S_FAILURE";}
    }
// ------------ y
    else if (strcmp(QuelleFonction, "Get_y") == 0)
    {
	if (strcasecmp (_algorithm_name, "vmlmn") != 0)
	    {return_c = "ERROR : Get_y is relevant only with vmlmn algorithm";}
	else 
	    {local_vmlmn = opk_get_vmlmn_y(vmlmb, k);}

        if (local_vmlmn == OPK_SUCCESS)
    	    {return_c = "OPK_SUCCESS";}   
        else
    	    {return_c = "OPK_GET_Y_FAILURE";}
    }

// ------------ WRONG INPUT
    else
	{return_c = "WRONG INPUT, CHECK FUNCTION HELP FOR MORE DETAILS";}

// Pour nlcg, differentes fonctions
    if (strcmp(QuelleFonction, "Get_fmin") == 0)
    {
 	if (strcasecmp (_algorithm_name, "nlcg") != 0) 
	    {return_c = "ERROR : Get_fmin is relevant only with vmlmn algorithm";}
	else 
	    {local_status_fmin = opk_get_nlcg_fmin(nlcg, double* fmin);}
    else if (strcmp(QuelleFonction, "Set_fmin") == 0)
    {
 	if (strcasecmp (_algorithm_name, "nlcg") != 0) 
	    {return_c = "ERROR : Get_fmin is relevant only with vmlmn algorithm";}
	else 
	    {local_status_fmin = opk_set_nlcg_fmin(nlcg, double fmin);}
    else if (strcmp(QuelleFonction, "Unset_fmin") == 0)
    {
 	if (strcasecmp (_algorithm_name, "nlcg") != 0) 
	    {return_c = "ERROR : Get_fmin is relevant only with vmlmn algorithm";}
	else 
	    {local_status_fmin = opk_unset_nlcg_fmin(nlcg);}
    else if (strcmp(QuelleFonction, "Get_description") == 0)
    {
 	if (strcasecmp (_algorithm_name, "nlcg") != 0) 
	    {return_c = "ERROR : Get_fmin is relevant only with vmlmn algorithm";}
	else 
	    {local_status_fmin = opk_get_nlcg_description(nlcg, char* str);}
    
    if (local_status_fmin == OPK_SUCCESS)
        {return_c = "OPK_SUCCESS";}   
    else
        {return_c = "OPK_FMIN_FAILURE";}
*/


    fclose(fichier);
// On renvoit la valeur demandee sous forme de chaine de charactere
    return Py_BuildValue("s",return_c);
}
// ------------------------------------------------------------------------------------------ 


/* ------------------------------------------------------------------------------------------
------------------------------------ FONCTION CLOSE -----------------------------------------
--------------------------------------------------------------------------------------------- */
static PyObject *
opk_close(PyObject * self)
{
    // Free workspace 
    OPK_DROP(_vx);
    OPK_DROP(_vg);
    OPK_DROP(_vgp);
    OPK_DROP(_vspace);

    // Drop algorithm
    if (strcasecmp(_algorithm_name, "nlcg") == 0)
        OPK_DROP(_nlcg);
    else
        OPK_DROP(_vmlmb);

    // Function does not return anything
    Py_RETURN_NONE;
}
// ------------------------------------------------------------------------------------------ 



// ------------------------------- WRAPPER CREATION ----------------------------------------- 
//  Define functions in module
static PyMethodDef Methods[] = 
    {
    {"opk_initialisation", (PyCFunction)opk_initialisation, METH_VARARGS|METH_KEYWORDS, "lala"},
    {"opk_iteration", (PyCFunction)opk_iteration, METH_VARARGS, "lala"},
    {"opk_taskInfo", (PyCFunction)opk_taskInfo, METH_VARARGS, "lala"},
    {"opk_close", (PyCFunction)opk_close, METH_NOARGS, "lala"},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
//module initialization 
static struct PyModuleDef optimpack_module =
    { PyModuleDef_HEAD_INIT, "opkc_v3", NULL, -1, Methods };

    // Python 3+
PyMODINIT_FUNC
PyInit_opkc_v3(void)
{
    import_array();
    return PyModule_Create(&optimpack_module);
}
#else
    // PYTHON2
PyMODINIT_FUNC
initopkc_v3(void)
{
    (void) Py_InitModule("opkc_v3", Methods);
    import_array();
}
#endif
// ------------------------------------------------------------------------------------------ 
