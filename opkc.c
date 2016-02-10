#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <optimpack.h>

/*				Les Warning

	- MyOptimizer : Il nous le faut en global pour pouvoir l'appeler dans WhatTask mais comment le declarer sachant que son type detend de l'algo??
	- Plein de warning en decoule quand on fait des comparaisons et des egalites. On peut garder 5 variables globales pour l'optimiseur sinon mais pas pratique.

	-Using deprecated NumPy API : On s'en fiche
*/

/*  -------------------------- ON DEFINIT LES VARIABLES GLOBALES --------------------------- */
// L'Espace
opk_vspace_t *vspace;		

// Les Vecteurs d'espace : ils changent mais on ne les declare qu'une fois
opk_vector_t *vgp;		// Mon Projected Gradient Vector
opk_vector_t *vx; 	        // Mes vecteurs d'entree
opk_vector_t *vg;               // Le gradient en x

// La Tache : elle change tout le temps et determine ce qu'il faut faire
opk_task_t task;

// Les Bords		
opk_bound_t *lower;
opk_bound_t *upper;

// Les Optimiseurs 
opk_nlcg_t *MyOptimizer = NULL; 
opk_nlcg_t *nlcg = NULL;        // non-linear conjugate gradient optimizer 
opk_vmlm_t *vmlm = NULL;        // limited memory quasi-Newton optimizer 
opk_vmlmb_t *vmlmb = NULL;      // idem with bound constraints 
opk_vmlmn_t *vmlmn = NULL;      // idem with bound constraints 
opk_lbfgs_t *lbfgs = NULL;      // LBFGS optimizer 

// Autres variables globales potentiels
opk_lnsrch_t *lnsrch = NULL;    // Line search method 


/*  --------------------------------- ON DEFINIT GRADNORM ----------------------------------- */
/* Return the (projected) gradient norm as in CG_DESCENT or ASA_CG 
static double
gradnorm(const opk_vector_t* x, const opk_vector_t* g, opk_vector_t* gp,
         const opk_bound_t* xl, const opk_bound_t* xu) {
    if (gp != NULL) 
    {
        // compute ||Projection(x_k - g_k) - x_k||_infty 
        opk_vaxpby(gp, 1, x, -1, g);
        if (opk_box_project_variables(gp, gp, xl, xu) != OPK_SUCCESS)
	{
            printf("# Failed to project variables\n");
            exit(-1);
        }
        opk_vaxpby(gp, 1, gp, -1, x);
        return opk_vnorminf(gp);
     }
     else 
     {
        return opk_vnorminf(g);
     }
}
*/

/*  ------------------------------- FONCTION INITIALISATION --------------------------------- */
static PyObject *Initialisation (PyObject * self, PyObject * args, PyObject * keywds)
{

/*  ------------------------------------- DECLARATIONS -------------------------------------- */
#define TRUE  1
#define FALSE 0
#define NLCG  1
#define VMLM  2
#define LBFGS 3
#define VMLMB 4
#define BLMVM 5
#define VMLMN 6

// Si j'ai besoin de voir
 FILE* fichier = NULL;
    fichier = fopen("text.txt", "w");
    if (fichier == NULL)
    {
	return Py_None;
    }
    fprintf(fichier, "repere1 \n");

// A trier
    double DELTA_DEFAULT = 5e-2;
    double EPSILON_DEFAULT = 1e-2;
    unsigned int vmlmn_flags = 0;
    double sftol = 1e-4;
    double sgtol = 0.9;
    double sxtol = 1e-15;
    double samin = 0.1;
    int bounds = 0;
    opk_vector_t *vbl;		// lower
    opk_vector_t *vbu;		// upper

// Sert a reperer l'algo mais on a deja algorithm_name ??
    int algorithm = VMLMB;

// On declare les vrai valeur des entrees. Elles contiendront ce qui est demande par l'utilisateur
    int autostep = 0;
    unsigned int nlcg_method = OPK_NLCG_DEFAULT;
    int delta_given = FALSE;
    int epsilon_given = FALSE;

// On declare les variables c qui serviront d'argument d'entree
    double *x, *ff, *g;
// Pareil + on initialise les parametres facultatifs
    double bl = 0;
    double bu = 1e6;
    const char *algorithm_name = "nlcg";
    char *linesrch_name = "quadratic";
    char *autostep_name = "ShannoPhua";
    char *nlcgmeth_name = "FletcherReeves";
    double delta = DELTA_DEFAULT;
    double epsilon = EPSILON_DEFAULT;
    double gatol = 1.0e-6;     
    double grtol = 0.0;         
    int maxiter = -1;
    int maxeval = -1;
    int mem = 5;
    int powell = FALSE;
    int verbose = 0;

// On declare les entrees obligatoires (pourquoi PyObject?)
    PyObject *x_obj=NULL, *g_obj;
    PyObject *x_arr=NULL, *g_arr;

// On declare la valeur de retour
    PyObject * task_py;
    fprintf(fichier, "repere2 \n");

/*  --------------------- ON CONVERTIT LES OBJETS PYTHON EN OBJET C ------------------------ */
/*
 i = int
 s = string
 f = float
 d = double
 p = predicate (bool) = int
 | = other arguments are optionals
*/
    static char *kwlist[] = 
    {"x", "f", "g", "bl", "bu", "algorithm", "linesearch", "autostep", "nlcg", "delta", "epsilon", "gatol", "grtol", "maxiter", "maxeval", "mem", "powell", "verbose", NULL};

    if (!PyArg_ParseTupleAndKeywords (args, keywds, "OdO|ddssssffffiiiii", kwlist, &x_obj, &ff, &g_obj, &bl, &bu, &algorithm_name, &linesrch_name, &autostep_name, &nlcgmeth_name, &delta, &epsilon, &gatol, &grtol, &maxiter, &maxeval, &mem, &powell, &verbose))
        {return NULL;}
    fprintf(fichier, "repere3 \n");

/*  ------------ ON REMPLIT LES VARIABLES SELON LES DEMANDES DE L'UTILISATEUR --------------- */
// les _obj sont de type "PyObject" et _arr de type "ndarray" ??
    if (x_obj == NULL) 
        {return NULL;}
    x_arr  = PyArray_FROM_OTF(x_obj,  NPY_DOUBLE, NPY_IN_ARRAY);
    g_arr  = PyArray_FROM_OTF(g_obj,  NPY_DOUBLE, NPY_IN_ARRAY);
    if (x_arr == NULL || g_arr == NULL) 
	{return NULL;}

// Nombre de dimensions
    int nd = PyArray_NDIM(x_arr);   
    npy_intp *shape = PyArray_DIMS(x_arr);
    if(nd != 1) 
    {
        printf("We need 1D arrays");
        return NULL;
    }
    x  = (double*)PyArray_DATA(x_arr);
    g  = (double*)PyArray_DATA(g_arr);

    if (x == NULL || g == NULL) 
    	{return NULL;}

    if (verbose > 0) 
        {printf("ALGO %s LINE %s AUTO %s\n", algorithm_name, linesrch_name, autostep_name);}

// ALGORITHM
    if (strcasecmp (algorithm_name, "nlcg") == 0) 
   	{algorithm = NLCG;}
    else if (strcasecmp (algorithm_name, "vmlm") == 0) 
   	{algorithm = VMLM;}
    else if (strcasecmp (algorithm_name, "vmlmb") == 0) 
	{algorithm = VMLMB;}
    else if (strcasecmp (algorithm_name, "vmlmn") == 0) 
	{algorithm = VMLMN;}
    else if (strcasecmp (algorithm_name, "blmvm") == 0) 
    {
        algorithm = VMLMN;
        vmlmn_flags = OPK_EMULATE_BLMVM;
    }
    else if (strcasecmp (algorithm_name, "lbfgs") == 0) 
        {algorithm = LBFGS;}
    else 
    {
        printf ("# Unknown algorithm\n");
        return NULL;
    }
    fprintf(fichier, "repere4 \n");

// LINESEARCH
    if (strcasecmp (linesrch_name, "quadratic") == 0) 
        {lnsrch = opk_lnsrch_new_backtrack (sftol, samin);}
    else if (strcasecmp (linesrch_name, "Armijo") == 0) 
        {lnsrch = opk_lnsrch_new_backtrack (sftol, 0.5);}
    else if (strcasecmp (linesrch_name, "cubic") == 0)
        {lnsrch = opk_lnsrch_new_csrch (sftol, sgtol, sxtol);}
    else 
    {
        printf ("# Unknown line search method\n");
        return NULL;
    }
// AUTOSTEP
    if (strcasecmp (autostep_name, "ShannoPhua") == 0) 
        {autostep = OPK_NLCG_SHANNO_PHUA;}
    else if (strcasecmp (autostep_name, "OrenSpedicato") == 0) 
        {autostep = OPK_NLCG_OREN_SPEDICATO;}
    else if (strcasecmp (autostep_name, "BarzilaiBorwein") == 0) 
        {autostep = OPK_NLCG_BARZILAI_BORWEIN;}
    else 
    {
        printf ("# Unknown line search method\n");
        return NULL;
    }
// NLCG METHOD
    if (strcasecmp (nlcgmeth_name, "FletcherReeves") == 0) 
        {nlcg_method = OPK_NLCG_FLETCHER_REEVES;}
    else if (strcasecmp (nlcgmeth_name, "HestenesStiefel") == 0) 
        {nlcg_method = OPK_NLCG_HESTENES_STIEFEL;}
    else if (strcasecmp (nlcgmeth_name, "PolakRibierePolyak") == 0) 
        {nlcg_method = OPK_NLCG_POLAK_RIBIERE_POLYAK;}
    else if (strcasecmp (nlcgmeth_name, "Fletcher") == 0) 
        {nlcg_method = OPK_NLCG_FLETCHER;}
    else if (strcasecmp (nlcgmeth_name, "LiuStorey") == 0) 
        {nlcg_method = OPK_NLCG_LIU_STOREY;}
    else if (strcasecmp (nlcgmeth_name, "DaiYuan") == 0) 
        {nlcg_method = OPK_NLCG_DAI_YUAN;}
    else if (strcasecmp (nlcgmeth_name, "PerryShanno") == 0) 
        {nlcg_method = OPK_NLCG_PERRY_SHANNO;}
    else if (strcasecmp (nlcgmeth_name, "HagerZhang") == 0) 
        {nlcg_method = OPK_NLCG_HAGER_ZHANG;}
    else 
    {
        printf ("# Unknown line search method\n");
        return NULL;
    }
// DELTA
    if (delta != DELTA_DEFAULT) 
	{delta_given = TRUE;}
// EPSILON
    if (epsilon != EPSILON_DEFAULT)
        {epsilon_given = TRUE;}
// NCLG METHOD ? POWELL ? AUTOSTEP
    if (powell) 
        {nlcg_method |= OPK_NLCG_POWELL;}
    if (autostep != 0) 
        {nlcg_method |= autostep;}
// Testing some inputs
    if (gatol < 0) 
    {
        printf ("# Bad value for GATOL\n");
        return NULL;
    }
    if (grtol < 0) 
    {
        printf ("# Bad value for GRTOL\n");
        return NULL;
    }

// Check whether the problem has constraints 
    if (bounds != 0 && mem == 0) 
    {
        printf ("# Use VMLMB or BLMVM for bound constrained optimization\n");
        return NULL;
    }
    fprintf(fichier, "repere5 \n");

/*  --------------------------------- CONSTRUCTION DE VSPACE -------------------------------- */
// Shape ne depend que de x_arr
    vspace = opk_new_simple_double_vector_space (shape[0]);  // FIXME CUTEst nvar ?
    if (vspace == NULL) 
    {
        printf ("# Failed to allocate vector space\n");
        return NULL;
    }
// On transforme x et g en vecteur de l'espace qu'on cree. Ils ne changent pas (juste le type)
    vx = opk_wrap_simple_double_vector (vspace, x, free, x);
    vg = opk_wrap_simple_double_vector (vspace, g, free, g);
    vbl = opk_wrap_simple_double_vector (vspace, &bl, free, &bl);
    vbu = opk_wrap_simple_double_vector (vspace, &bu, free, &bu);
    if (vx == NULL || vg == NULL) 
    {
        printf ("# Failed to wrap vectors\n");
        return NULL;
    }
    if ((bounds & 1) == 0)
        {lower = NULL;}
    else 
    {
        lower = opk_new_bound (vspace, OPK_BOUND_VECTOR, vbl);
        if (lower == NULL)
        {
            printf ("# Failed to wrap lower bounds\n");
            return NULL;
        }
    }
    if ((bounds & 2) == 0) 
        {upper = NULL;}
    else 
    {
        upper = opk_new_bound (vspace, OPK_BOUND_VECTOR, vbu);
        if (upper == NULL) 
        {
            printf ("# Failed to wrap upper bounds\n");
            return NULL;
        }
    }
    fprintf(fichier, "repere6 \n");

/*  -------------------------------- CREATION DE L'OPTIMISEUR ------------------------------- */
// On cree notre vgp avec vspace
// On cree notre optimiseur et notre tache en fonction de l'algo choisi
    vgp = NULL;
    if (algorithm == VMLMB)
    {
        opk_vmlmb_options_t options;
        algorithm_name = "VMLMB-";
        if (lnsrch != NULL) 
            {vmlmb = opk_new_vmlmb_optimizer_with_line_search (vspace, mem, lnsrch);}
        else 
            {vmlmb = opk_new_vmlmb_optimizer (vspace, mem);}
        if (vmlmb == NULL) 
        {
            printf ("# Failed to create VMLMB optimizer\n");
            return NULL;
        }
        vgp = opk_vcreate (vspace);
        if (vgp == NULL) 
        {
            printf ("# Failed to create projected gradient vector\n");
            return NULL;
        }
        opk_get_vmlmb_options (&options, vmlmb);
        options.gatol = 0.0;
        options.grtol = 0.0;
        if (delta_given) 
            {options.delta = delta;}
        else 
            {delta = options.delta;}
        if (epsilon_given) 
	    {options.epsilon = epsilon;}
        else 
            {epsilon = options.epsilon;}
        if (opk_set_vmlmb_options (vmlmb, &options) != OPK_SUCCESS) 
	{
            printf ("# Bad VMLMB options\n");
            return NULL;
        }
        task = opk_start_vmlmb (vmlmb, vx, lower, upper);
    }
    else if (algorithm == VMLMN)
    {
        opk_vmlmn_options_t options;
        vmlmn = opk_new_vmlmn_optimizer (vspace, mem, vmlmn_flags,lower, upper, lnsrch);
        if (vmlmn == NULL) 
	{
            printf ("# Failed to create VMLMN optimizer\n");
            return NULL;
        }
        algorithm_name = opk_get_vmlmn_method_name(vmlmn);
        vgp = opk_vcreate (vspace);
        if (vgp == NULL) 
	{
            printf ("# Failed to create projected gradient vector\n");
            return NULL;
        }
        opk_get_vmlmn_options (&options, vmlmn);
        options.gatol = 0.0;
        options.grtol = 0.0;
        if (delta_given)  
	{options.delta = delta;}
        else  
	{delta = options.delta;}
        if (epsilon_given)  
	{options.epsilon = epsilon;}
        else  
	{epsilon = options.epsilon;}
        if (opk_set_vmlmn_options (vmlmn, &options) != OPK_SUCCESS) 
	{
            printf ("# Bad VMLMN options\n");
            return NULL;
        }
        task = opk_start_vmlmn (vmlmn, vx);
    }
    else if (algorithm == VMLM) 
    {
        opk_vmlm_options_t options;
        algorithm_name = "VMLM";
        if (bounds != 0) 
	{
            printf ("# Algorithm %s cannot be used with bounds\n",algorithm_name);
            return NULL;
        }
        if (lnsrch != NULL) 
	    {vmlm = opk_new_vmlm_optimizer_with_line_search (vspace, mem, lnsrch);}
        else 
	    {vmlm = opk_new_vmlm_optimizer (vspace, mem);}
        if (vmlm == NULL) 
	{
            printf ("# Failed to create VMLM optimizer\n");
            return NULL;
        }
        opk_get_vmlm_options (&options, vmlm);
        options.gatol = 0.0;
        options.grtol = 0.0;
        if (delta_given) 
	    {options.delta = delta;}
        else 
            {delta = options.delta;}
        if (epsilon_given) 
           { options.epsilon = epsilon;}
        else 
            {epsilon = options.epsilon;}
        if (opk_set_vmlm_options (vmlm, &options) != OPK_SUCCESS) 
	{
            printf ("# Bad VMLM options\n");
            return NULL;
        }
        task = opk_start_vmlm (vmlm, vx);
    }
    else if (algorithm == LBFGS) 
	{
        opk_lbfgs_options_t options;
        algorithm_name = "LBFGS";
        if (bounds != 0) 
	{
            printf ("# Algorithm %s cannot be used with bounds\n",algorithm_name);
            return NULL;
        }
        if (lnsrch != NULL) 
	    {lbfgs =opk_new_lbfgs_optimizer_with_line_search (vspace, mem,lnsrch);}
        else 
	    {lbfgs = opk_new_lbfgs_optimizer (vspace, mem);}
        if (lbfgs == NULL) 
	{
            printf ("# Failed to create LBFGS optimizer\n");
            return NULL;
        }
        opk_get_lbfgs_options (&options, lbfgs);
        options.gatol = 0.0;
        options.grtol = 0.0;
        if (delta_given) 
	    {options.delta = delta;}
        else 
	    {delta = options.delta;}
        if (epsilon_given) 
	    {options.epsilon = epsilon;}
        else 
            {epsilon = options.epsilon;}
        if (opk_set_lbfgs_options (lbfgs, &options) != OPK_SUCCESS) 
	{
            printf ("# Bad LBFGS options\n");
            return NULL;
        }
        task = opk_start_lbfgs (lbfgs, vx);
    }
    else if (algorithm == NLCG) 
	{
        opk_nlcg_options_t options;
        algorithm_name = "NLCG";
        if (bounds != 0) 
	{
            printf ("# Algorithm %s cannot be used with bounds\n",algorithm_name);
            return NULL;
	}
        if (lnsrch != NULL) 
	    {nlcg =opk_new_nlcg_optimizer_with_line_search (vspace, nlcg_method,lnsrch);}
        else 
 	    {nlcg = opk_new_nlcg_optimizer (vspace, nlcg_method);}
        if (nlcg == NULL) 
	{
            printf ("# Failed to create NLCG optimizer\n");
            return NULL;
	}
        opk_get_nlcg_options (&options, nlcg);
        options.gatol = 0.0;
        options.grtol = 0.0;
        if (delta_given) 
	    {options.delta = delta;}
        else 
            {delta = options.delta;}
        if (epsilon_given) 
            {options.epsilon = epsilon;}
        else 
            {epsilon = options.epsilon;}
        if (opk_set_nlcg_options (nlcg, &options) != OPK_SUCCESS)
	{
            printf ("# Bad NLCG options\n");
            return NULL;
        }
        task = opk_start_nlcg (nlcg, vx);
    }
    else 
    {
        printf ("# Bad algorithm\n");
        return NULL;
    }
    fprintf(fichier, "repere7 \n");

// On declare globalement quelle optimiseur on utilise
    if (vmlmb != NULL) 
   	 {MyOptimizer = vmlmb;}
    else if (vmlmn != NULL) 
   	 {MyOptimizer = vmlmn;}
    else if (vmlm != NULL) 
   	 {MyOptimizer = vmlm;}
    else if (lbfgs != NULL) 
   	 {MyOptimizer = lbfgs;}
    else 
   	 {MyOptimizer = nlcg;}

    // Free workspace 
/*    OPK_DROP (vx);
    OPK_DROP (vg);
    OPK_DROP (vgp);
    OPK_DROP (vbl);
    OPK_DROP (vbu);
    OPK_DROP (vspace);
    OPK_DROP (nlcg);
*/
    fprintf(fichier, "repere8 \n");

    fclose(fichier);

    // On est pret a demarrer l'algo
    task_py = Py_BuildValue("s","OPK_TASK_COMPUTE_FG"); 

    return task_py;
}
// ---------------------------------------------------------------------------------------- 




// ---------------------------------------------------------------------------------------- 
static PyObject *WhatTask (PyObject * self, PyObject * args)
{
// Si on veut afficher des trucs pour y voir plus clair
/*    FILE* fichier = NULL;
    fichier = fopen("text.txt", "w");
    if (fichier == NULL)
    {
	return Py_None;
    }
    fprintf(fichier, "s pour string \n");
*/

// arguments d'entree
    char *QuelleFonction;
    PyObject *x_obj=NULL, *x_arr=NULL;
    double f_c;
    PyObject *g_obj, *g_arr;
// arguments de sortie
    opk_task_t task_locale;
    char * task_c;
// Autres arguments
    double *x, *g;

// Conversion
    if (!PyArg_ParseTuple (args, "sOdO",&QuelleFonction, &x_obj, &f_c, &g_obj))
       {return NULL;}

// Je sais pas a quoi ca sert mais c'etait fait par Jonathan
    if (x_obj == NULL) 
       {return NULL;}
    x_arr  = PyArray_FROM_OTF(x_obj,  NPY_DOUBLE, NPY_IN_ARRAY);
    g_arr  = PyArray_FROM_OTF(g_obj,  NPY_DOUBLE, NPY_IN_ARRAY);
    if (x_arr == NULL || g_arr == NULL)
       {return NULL;}
    x  = (double*)PyArray_DATA(x_arr);
    g  = (double*)PyArray_DATA(g_arr);
    if (x == NULL || g == NULL)
       {return NULL;}

// On transforme x et g en vecteur de l'espace qu'on cree. 
    vx = opk_wrap_simple_double_vector (vspace, x, free, x);
    vg = opk_wrap_simple_double_vector (vspace, g, free, g);

// On appelle la fonction demandee
    if (strcmp(QuelleFonction, "Iterate") == 0)
    {
	if (MyOptimizer == nlcg)
	    {task_locale = opk_iterate_nlcg(MyOptimizer, vx, f_c, vg);}
	else if (MyOptimizer == lbfgs)
	    {task_locale = opk_iterate_lbfgs(MyOptimizer, vx, f_c, vg);}
	else if (MyOptimizer == vmlm)
	    {task_locale = opk_iterate_vmlm(MyOptimizer, vx, f_c, vg);}
	else if (MyOptimizer == vmlmn)
	    {task_locale = opk_iterate_vmlmn(MyOptimizer, vx, f_c, vg);}
	else if (MyOptimizer == vmlmb)
	    {task_locale = opk_iterate_vmlmb(MyOptimizer, vx, f_c, vg, lower, upper);}
    }

// On prend la valeur de "task"
    if (task_locale == OPK_TASK_START)
    {task_c = "OPK_TASK_START";}   
    else if (task_locale == OPK_TASK_COMPUTE_FG)
    {task_c = "OPK_TASK_COMPUTE_FG";}
    else if (task_locale == OPK_TASK_NEW_X)
    {task_c = "OPK_TASK_NEW_X";}
    else if (task_locale == OPK_TASK_FINAL_X)
    {task_c = "OPK_TASK_FINAL_X";}
    else if (task_locale == OPK_TASK_WARNING)
    {task_c = "OPK_TASK_WARNING";}
    else if (task_locale == OPK_TASK_ERROR)
    {task_c = "OPK_TASK_ERROR";}
    else
    {task_c = "OPK_TASK_UNKNOWN";}

   // fclose(fichier);

// On renvoit la valeur de "task"
    return Py_BuildValue("s",task_c);
}




// ---------------------------------------------------------------------------------------- 
//  Define functions in module
static PyMethodDef Methods[] = {
    {"Initialisation", Initialisation, METH_VARARGS|METH_KEYWORDS, "lala"},
    {"WhatTask", WhatTask, METH_VARARGS, "lala"},
    {NULL, NULL, 0, NULL}
};

//module initialization */
	// PYTHON2
PyMODINIT_FUNC
initopkc (void)
{
    (void) Py_InitModule("opkc", Methods);
    import_array();
}

/*
	// PYTHON3
static struct PyModuleDef optimpack_module = {
    PyModuleDef_HEAD_INIT, "optimpack_module", NULL, -1, Methods
};

PyMODINIT_FUNC
PyInit_optimpack_wrapper(void)
{
    return PyModule_Create(&optimpack_module);
    import_array();
}
*/
// ---------------------------------------------------------------------------------------- 

