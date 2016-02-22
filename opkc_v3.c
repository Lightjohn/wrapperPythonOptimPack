#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <optimpack.h>
// #include <optimpack-private.h>


/*  -------------------------- ON DEFINIT LES VARIABLES GLOBALES --------------------------- */
// L'Espace
opk_vspace_t *vspace;		

// Les Vecteurs d'espace : ils changent mais on ne les declare qu'une fois
opk_vector_t *vgp;		// Mon Projected Gradient Vector
opk_vector_t *vx; 	        // Mes vecteurs d'entree
opk_vector_t *vg;               // Le gradient en x

// La Tache : elle change tout le temps et determine ce qu'il faut faire
opk_task_t task;
		

// Les Optimiseurs 
const char *algorithm_name = "vmlmb"; 	// Le nom de l'optimiseur
opk_nlcg_t *nlcg;        // non-linear conjugate gradient optimizer 
opk_vmlmb_t *vmlmb;      // quasi-newton
opk_optimizer_t *limited_optimizer; // limited memory optimizer

// Les Options
opk_nlcg_options_t options_nlcg;
opk_vmlmb_options_t options_vmlmb;

// Le type et la taille 
opk_type_t type = OPK_DOUBLE;
opk_index_t problem_size;



/* ------------------------------------------------------------------------------------------
------------------------------- FONCTION INITIALISATION -------------------------------------
--------------------------------------------------------------------------------------------- */

static PyObject *Initialisation (PyObject * self, PyObject * args, PyObject * keywds)
{

//  ------------------------------------- DECLARATIONS --------------------------------------
#define TRUE  1
#define FALSE 0

// Si j'ai besoin de voir
 FILE* fichier = NULL;
    fichier = fopen("text.txt", "w");
    if (fichier == NULL)
    {
	return Py_None;
    }
    fprintf(fichier, "repere1 \n");


// On declare les vrai valeur des entrees. Elles contiendront ce qui est demande par l'utilisateur
    opk_lnsrch_t *lnsrch = NULL;   
    int autostep = 0;
    unsigned int nlcg_method = OPK_NLCG_DEFAULT; 
    unsigned int vmlmb_method = OPK_LBFGS; 	// 	OPK_LBFGS, OPK_VMLMB, OPK_BLMVM	
    unsigned int limited_method = nlcg_method;

// On declare les variables c qui serviront d'argument d'entree
    void *x;
    char *linesrch_name = "quadratic";
    char *autostep_name = "ShannoPhua";
    char *nlcgmeth_name = "FletcherReeves";
    char *vmlmbmeth_name = "OPK_LBFGS";
    double delta;					// pas besoin de valeur par defaut
    double epsilon;					// pas besoin de valeur par defaut
    int delta_given;					// pas besoin de valeur par defaut
    int epsilon_given;					// pas besoin de valeur par defaut
    double gatol = 1.0e-6;     
    double grtol = 0.0;         
    double bl = 0;					// 0        On prend la valeur 
    double bu = 1e6;					// 1e6     donnee si l'algorithm 
    int bound_given = 0;				// pas de bound de base
    int mem = 5;					// 5            est vmlmb
    int powell = FALSE;
    int single = FALSE;
    int limited = FALSE;

// On rajoute x
    PyObject *x_obj=NULL;
    PyObject *x_arr=NULL;

// On declare d'autres variables locales
    opk_vector_t *vbl;					// bounds	
    opk_vector_t *vbu;					// bounds
    double sftol = 1e-4;				// Valeurs pour linesearch
    double sgtol = 0.9;					// Valeurs pour linesearch
    double sxtol = 1e-15;				// Valeurs pour linesearch
    double samin = 0.1;					// Valeurs pour linesearch		
    opk_bound_t *lower;					// Les Bords
    opk_bound_t *upper;					// Les Bords
    

// On declare la valeur de retour
    PyObject * task_py;
    fprintf(fichier, "repere2 \n");
//  -----------------------------------------------------------------------------------------


//  --------------------- ON CONVERTIT LES OBJETS PYTHON EN OBJET C ------------------------- 
/*
 i = int
 s = string
 f = float
 d = double
 p = predicate (bool) = int
 | = other arguments are optionals
*/
    static char *kwlist[] = 
    {"x", "algorithm", "linesearch", "autostep", "nlcg", "vmlmb", "delta", "epsilon", "delta_given", "epsilon_given", "gatol", "grtol", "bl", "bu", "bound_given", "mem", "powell", "single", "limited", NULL};

    if (!PyArg_ParseTupleAndKeywords (args, keywds, "Osssssddiiddddiiiii", kwlist, &x_obj, &algorithm_name, &linesrch_name, &autostep_name, &nlcgmeth_name, &vmlmbmeth_name, &delta, &epsilon, &delta_given, &epsilon_given, &gatol, &grtol, &bl, &bu, &bound_given, &mem, &powell, &single, &limited))
    {
	return NULL;
    }
    fprintf(fichier, "repere3 \n");
//  -----------------------------------------------------------------------------------------


//  ----------------------------- ON DECRIT VSPACE GRACE A X -------------------------------- 
    if (x_obj == NULL) 
        {return NULL;}
    if (single == FALSE)
        {x_arr  = PyArray_FROM_OTF(x_obj,  NPY_DOUBLE, NPY_IN_ARRAY);}
    else 
        {x_arr  = PyArray_FROM_OTF(x_obj,  NPY_FLOAT, NPY_IN_ARRAY);}
    if (x_arr == NULL) 
	{return NULL;}

    int nd = PyArray_NDIM(x_arr);   
    npy_intp *shape = PyArray_DIMS(x_arr);
    problem_size = shape[0];
    if(nd != 1) 
    {
        printf("We need 1D arrays");
        return NULL;
    }
    if (single == FALSE)
        {x  = (double*)PyArray_DATA(x_arr);}
    else
        {x  = (float*)PyArray_DATA(x_arr);}

    if (x == NULL) 
    	{return NULL;}
//  -----------------------------------------------------------------------------------------


//  -------------------- ON REMPLIT LNSRCH, AUTOSTEP, NLCG ET VMLMB -------------------------
// LINESEARCH
    if (strcasecmp (linesrch_name, "quadratic") == 0) 
        {lnsrch = opk_lnsrch_new_backtrack (sftol, samin);}
    else if (strcasecmp (linesrch_name, "Armijo") == 0) 
        {lnsrch = opk_lnsrch_new_backtrack (sftol, 0.5);}
    else if (strcasecmp (linesrch_name, "cubic") == 0)
        {lnsrch = opk_lnsrch_new_csrch (sftol, sgtol, sxtol);}
    else if (strcasecmp (linesrch_name, "nonmonotone") == 0)
	{lnsrch = opk_lnsrch_new_nonmonotone(mem, 1e-4, 0.1, 0.9);}
    else 
    {
        printf ("# Unknown line search method\n");
	lnsrch = NULL;
        return NULL;
    }

// AUTOSTEP
    if (strcasecmp (autostep_name, "ShannoPhua") == 0) 
        {autostep = OPK_NLCG_SHANNO_PHUA;}
/*
    else if (strcasecmp (autostep_name, "OrenSpedicato") == 0) 
        {autostep = OPK_NLCG_OREN_SPEDICATO;}
    else if (strcasecmp (autostep_name, "BarzilaiBorwein") == 0) 
        {autostep = OPK_NLCG_BARZILAI_BORWEIN;}
*/
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
        printf ("# Unknown line search method for nlcg\n");
        return NULL;
    }

// VLMLB METHOD
    if (strcasecmp (vmlmbmeth_name, "blmvm") == 0) 
        {vmlmb_method = OPK_BLMVM;}
    else if (strcasecmp (vmlmbmeth_name, "vmlmb") == 0) 
        {vmlmb_method = OPK_VMLMB;}
    else if (strcasecmp (vmlmbmeth_name, "lbfgs") == 0) 
        {vmlmb_method = OPK_LBFGS;}
    else 
    {
        printf ("# Unknown line search method for vmlmb\n");
        return NULL;
    }

// type of the variable
if (single == TRUE)
	{type = OPK_FLOAT;}
else
	{type = OPK_DOUBLE;}

//  -----------------------------------------------------------------------------------------


//  ------------------------ COMPARAISON BIT A BIT, JE SAIS PAS TROP ------------------------
    if (powell) 
        {nlcg_method |= OPK_NLCG_POWELL;}
    if (autostep != 0) 
        {nlcg_method |= autostep;}
//  -----------------------------------------------------------------------------------------


//  --------------------------------- CONSTRUCTION DE VSPACE -------------------------------- 
// Shape ne depend que de x_arr
    vspace = opk_new_simple_double_vector_space (problem_size);  
    if (vspace == NULL) 
    {
        printf ("# Failed to allocate vector space\n");
        return NULL;
    }
// On transforme x en vecteur de l'espace qu'on cree. Ils ne changent pas (juste le type)
    vx = opk_wrap_simple_double_vector (vspace, x, NULL, x); // normalement free au lieu de NULL
    if (vx == NULL) 
    {
       	printf ("# Failed to wrap vectors\n");
        return NULL;
    }
//  -----------------------------------------------------------------------------------------


//  -------------------------------- CONSTRUCTION DES BOUNDS -------------------------------- 
// Declare quoi qu'il en soit parce qu'on le drop a la fin de la fonction
    if (single == TRUE)
	{
	bl = (float)(bl);
	bu = (float)(bu);
	}
    vbl = opk_wrap_simple_double_vector (vspace, &bl,NULL, &bl);
    vbu = opk_wrap_simple_double_vector (vspace, &bu, NULL, &bu);
// Si l'utilisateur veut des bounds (1 = lower, 2 = upper, 3 = les deux)
    if (bound_given == 1)
    {
        lower = opk_new_bound (vspace, OPK_BOUND_VECTOR, vbl);
        if (lower == NULL)
        {
            printf ("# Failed to wrap lower bounds\n");
            return NULL;
        }
	upper = NULL;
    }
    else if (bound_given == 2)
    {
        upper = opk_new_bound (vspace, OPK_BOUND_VECTOR, vbu);
        if (upper == NULL) 
        {
            printf ("# Failed to wrap upper bounds\n");
            return NULL;
        }
	lower = NULL;
    }
    else if (bound_given == 3)
    {
	lower = opk_new_bound (vspace, OPK_BOUND_VECTOR, vbl);
        upper = opk_new_bound (vspace, OPK_BOUND_VECTOR, vbu);
        if (lower == NULL)
        {
            printf ("# Failed to wrap lower bounds\n");
            return NULL;
        }
        if (upper == NULL) 
        {
            printf ("# Failed to wrap upper bounds\n");
            return NULL;
        }
    }
    else 
    {
	lower = NULL;
	upper = NULL;
    }

    fprintf(fichier, "repere6 \n");
//  -----------------------------------------------------------------------------------------


//  -------------------------------- CREATION DE L'OPTIMISEUR ------------------------------- 
// On cree notre optimiseur et notre tache en fonction de l'algo choisi
    if (limited == 0)
    {

// VMLMB --------------------------------
    	if (strcasecmp (algorithm_name, "vmlmb") == 0) 
    	{
	     // Creation de l'optimiseur
             vmlmb = opk_new_vmlmb_optimizer (vspace, mem, vmlmb_method, lower, upper, lnsrch);
             if (vmlmb == NULL) 
             {
                 printf ("# Failed to create VMLMB optimizer\n");
                 return NULL;
             }

          	// creation de vgp
             vgp = opk_vcreate (vspace);
             if (vgp == NULL) 
             {
                 printf ("# Failed to create projected gradient vector\n");
                 return NULL;
             }
          	// option.gatol = gatol? sinon la valeur recue de gatol est perdu?
             opk_get_vmlmb_options (&options_vmlmb, vmlmb);
             options_vmlmb.gatol = grtol;
             options_vmlmb.grtol = gatol;
             if (delta_given) 
                 {options_vmlmb.delta = delta;}
             if (epsilon_given) 
          	    {options_vmlmb.epsilon = epsilon;}
             if (opk_set_vmlmb_options (vmlmb, &options_vmlmb) != OPK_SUCCESS) 
          	{
                 printf ("# Bad VMLMB options\n");
                 return NULL;
             }

	     //const char *coucou = opk_get_vmlmb_method_name(vmlmb);
	     //fprintf(fichier, "Nom de la methode : %s \n", coucou );
             //task = opk_start_vmlmb (vmlmb, vx);

         }
// NLCG ---------------------------------  
         else if (strcasecmp (algorithm_name, "nlcg") == 0)  
         {
           	nlcg = opk_new_nlcg_optimizer (vspace, nlcg_method, lnsrch);
             if (nlcg == NULL) 
	     {
                 printf ("# Failed to create NLCG optimizer\n");
                 return NULL;
	     }

             opk_get_nlcg_options (&options_nlcg, nlcg);
             options_nlcg.gatol = gatol;
             options_nlcg.grtol = grtol;
             if (delta_given) 
	         {options_nlcg.delta = delta;}
             if (epsilon_given) 
                 {options_nlcg.epsilon = epsilon;}
             if (opk_set_nlcg_options (nlcg, &options_nlcg) != OPK_SUCCESS)
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
    }

// Limited Memory ---------------------------------  

    else
    {
	opk_algorithm_t limited_algorithm;
	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	{
	     limited_algorithm = OPK_ALGORITHM_NLCG;
	     limited_method = nlcg_method;
	}
	else if (strcasecmp (algorithm_name, "vmlmb") == 0) 
	{
	     limited_algorithm = OPK_ALGORITHM_VMLMB;
	     limited_method = vmlmb_method;
	}
        else 
        {
             printf ("# Bad algorithm\n");
             return NULL;
        }
	// optimiseur
	if (bound_given == 1)
	{
 		limited_optimizer = opk_new_optimizer (limited_algorithm, type, shape[0], mem, limited_method, type, &bl, OPK_BOUND_NONE, NULL, lnsrch);	
	}
	else if (bound_given == 2)
	{ 
		limited_optimizer = opk_new_optimizer (limited_algorithm, type, shape[0], mem, limited_method, OPK_BOUND_NONE, NULL, type, &bu, lnsrch);
	}
	else if (bound_given == 3)
	{ 
		limited_optimizer = opk_new_optimizer (limited_algorithm, type, shape[0], mem, limited_method, type, &bl, type, &bu, lnsrch);
	}
	else
	{
 		limited_optimizer = opk_new_optimizer (limited_algorithm, type, shape[0], mem, limited_method, OPK_BOUND_NONE, NULL, OPK_BOUND_NONE, NULL, lnsrch);
	}

        if (limited_optimizer == NULL) 
	{
            printf ("# Failed to create limited optimizer\n");
            return NULL;
	}

        task = opk_start(limited_optimizer, type, shape[0], x);
    }



    fprintf(fichier, "repere7 \n");

// Free workspace 
    OPK_DROP (vbl);
    OPK_DROP (vbu);

    fprintf(fichier, "repere8 \n");
    fclose(fichier);

// Return value is OPK_TASK_COMPUTE_FG
    if (task == OPK_TASK_COMPUTE_FG)
    	{task_py = Py_BuildValue("s","OPK_TASK_COMPUTE_FG");}
    else if (task == OPK_TASK_START)
    	{task_py = Py_BuildValue("s","OPK_TASK_START");}
    else if (task == OPK_TASK_NEW_X)
    	{task_py = Py_BuildValue("s","OPK_TASK_NEW_X");}
    else if (task == OPK_TASK_FINAL_X)
    	{task_py = Py_BuildValue("s","OPK_TASK_FINAL_X");}
    else if (task == OPK_TASK_WARNING)
    	{task_py = Py_BuildValue("s","OPK_TASK_WARNING");}
    else if (task == OPK_TASK_ERROR)
    	{task_py = Py_BuildValue("s","OPK_TASK_ERROR");}

    else {task_py = Py_BuildValue("s","INITIALIZATION_ERROR");}
    return task_py;
}
// ------------------------------------------------------------------------------------------ 




/* ------------------------------------------------------------------------------------------
----------------------------------- FONCTION ITERATE ----------------------------------------
--------------------------------------------------------------------------------------------- */
static PyObject *Iterate (PyObject * self, PyObject * args)
{
// arguments d'entree
    PyObject *x_obj=NULL, *x_arr=NULL;
    double f_c;
    PyObject *g_obj, *g_arr;
    int limited_c;
// arguments de sortie
    opk_task_t task_locale = OPK_TASK_ERROR;
    char * task_c;
// Autres arguments
    void *x, *g;

// Conversion selon le type
    if (type == OPK_DOUBLE)
    {
        if (!PyArg_ParseTuple (args, "OdOi",&x_obj, &f_c, &g_obj, &limited_c))
           {return NULL;}
    }
    else
    {
	f_c = (float)(f_c);
        if (!PyArg_ParseTuple (args, "OfOi",&x_obj, &f_c, &g_obj, &limited_c))
           {return NULL;}
    }

// On fait passer les valeurs dans x et g, selon le type
    if (x_obj == NULL) 
        {return NULL;}
    if (type == OPK_DOUBLE)
    {
        x_arr  = PyArray_FROM_OTF(x_obj,  NPY_DOUBLE, NPY_IN_ARRAY);
   	g_arr  = PyArray_FROM_OTF(g_obj,  NPY_DOUBLE, NPY_IN_ARRAY);
	x  = (double*)PyArray_DATA(x_arr);
        g  = (double*)PyArray_DATA(g_arr);
    }
    else
    {
        x_arr  = PyArray_FROM_OTF(x_obj,  NPY_FLOAT, NPY_IN_ARRAY);
   	g_arr  = PyArray_FROM_OTF(g_obj,  NPY_FLOAT, NPY_IN_ARRAY);
        x  = (float*)PyArray_DATA(x_arr);
        g  = (float*)PyArray_DATA(g_arr);
    }
    if (x_arr == NULL || g_arr == NULL)
       {return NULL;}
    if (x == NULL || g == NULL)
       {return NULL;}

// On transforme x et g en vecteur de l'espace qu'on cree. 
    vx = opk_wrap_simple_double_vector (vspace, x, NULL, x);
    vg = opk_wrap_simple_double_vector (vspace, g, NULL, g);


// On appelle la fonction iterate
    if (limited_c == 0)
    {
   	if (strcasecmp (algorithm_name, "nlcg") == 0) 
            {task_locale = opk_iterate_nlcg(nlcg, vx, f_c, vg);}
    	else if (strcasecmp (algorithm_name, "vmlmb") == 0)
            {task_locale = opk_iterate_vmlmb(vmlmb, vx, f_c, vg);}
    }
    else
    {task_locale = opk_iterate(limited_optimizer, type, problem_size, x, f_c, g);}


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

// On renvoit la valeur de "task"
    return Py_BuildValue("s",task_c);
}
// ------------------------------------------------------------------------------------------ 



/* ------------------------------------------------------------------------------------------
------------------------------------- FONCTION TASKINFO -------------------------------------
--------------------------------------------------------------------------------------------- */
/*
static PyObject *TaskInfo (PyObject * self, PyObject * args)
{
// Si on veut afficher des trucs pour y voir plus clair
    FILE* fichier = NULL;
    fichier = fopen("text.txt", "w");
    if (fichier == NULL)
    {
	return Py_None;
    }
    fprintf(fichier, "s pour string \n");

// Arguments d'entree
    char *QuelleFonction;
// Arguments de sortie
    opk_task_t local_task;
    opk_status_t local_status;
    opk_index_t local_iteration = -1;
    opk_index_t local_evaluation = -1;
    opk_index_t local_restart = -1;
    double local_step = -1;
    opk_status_t local_get_options;
    opk_status_t local_set_options;
    

// Arguments de sortie specifique a l'optimiseur
//    opk_vector_t* local_vmlmn; // vmlmn pour le get_s et le get_y

// Valeur de sortie
    char * return_c = "FAILURE : No value returned for TaskInfo";
// Autres declarations
    double Value_d = -1;
    char Value_c[80];

// Conversion
    if (!PyArg_ParseTuple (args, "s",&QuelleFonction))
       {return NULL;}

// On appelle la fonction demandee
// ------------ TASK
    if (strcmp(QuelleFonction, "Get_task") == 0)
    {
    // On store sa valeur dans l'objet approprie
 	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	    {local_task = opk_get_nlcg_task(nlcg);}
	else if (strcasecmp (algorithm_name, "lbfgs") == 0)
	    {local_task = opk_get_lbfgs_task(lbfgs);}
	else if (strcasecmp (algorithm_name, "vmlm") == 0)
	    {local_task = opk_get_vmlm_task(vmlm);}
	else if (strcasecmp (algorithm_name, "vmlmn") == 0)
	    {local_task = opk_get_vmlmn_task(vmlmn);}
	else 
	    {local_task = opk_get_vmlmb_task(vmlmb);}
    // Puis on converti la valeur de retour en chaine de charactere
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
// ------------ STATUS
    else if (strcmp(QuelleFonction, "Get_status") == 0)
    {
 	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	    {return_c = "ERROR : get_status is irrelevant for nlcg algorithm";}
	if (strcasecmp (algorithm_name, "lbfgs") == 0)
	    {local_status = opk_get_lbfgs_status(lbfgs);}
	else if (strcasecmp (algorithm_name, "vmlm") == 0)
	    {local_status = opk_get_vmlm_status(vmlm);}
	else if (strcasecmp (algorithm_name, "vmlmn") == 0)
	    {local_status = opk_get_vmlmn_status(vmlmn);}
	else 
	    {local_status = opk_get_vmlmb_status(vmlmb);}

        if (local_status == OPK_SUCCESS)
    	    {return_c = "OPK_SUCCESS";}   
        else
    	    {return_c = "OPK_GET_STATUS_FAILURE";}
    }
// ------------ ITERATION
    else if (strcmp(QuelleFonction, "Get_iteration") == 0)
    {
 	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	    {local_iteration = opk_get_nlcg_iterations(nlcg);}
	else if (strcasecmp (algorithm_name, "lbfgs") == 0)
	    {local_iteration = opk_get_lbfgs_iterations(lbfgs);}
	else if (strcasecmp (algorithm_name, "vmlm") == 0)
	    {local_iteration = opk_get_vmlm_iterations(vmlm);}
	else if (strcasecmp (algorithm_name, "vmlmn") == 0)
	    {local_iteration = opk_get_vmlmn_iterations(vmlmn);}
	else 
	    {local_iteration = opk_get_vmlmb_iterations(vmlmb);}

        if (local_iteration != -1)
	{
	    Value_d = local_iteration;
            sprintf(Value_c, "%lf", Value_d);
	    return_c = Value_c;
	} 
        else
    	    {return_c = "OPK_GET_ITERATION_FAILURE";}
    }
// ------------ EVALUATION
    else if (strcmp(QuelleFonction, "Get_evaluation") == 0)
    {
 	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	    {local_evaluation = opk_get_nlcg_evaluations(nlcg);}
	else if (strcasecmp (algorithm_name, "lbfgs") == 0)
	    {local_evaluation = opk_get_lbfgs_evaluations(lbfgs);}
	else if (strcasecmp (algorithm_name, "vmlm") == 0)
	    {local_evaluation = opk_get_vmlm_evaluations(vmlm);}
	else if (strcasecmp (algorithm_name, "vmlmn") == 0)
	    {local_evaluation = opk_get_vmlmn_evaluations(vmlmn);}
	else 
	    {local_evaluation = opk_get_vmlmb_evaluations(vmlmb);}

        if (local_evaluation != -1)
	{
	    Value_d = local_evaluation;
            sprintf(Value_c, "%lf", Value_d);
	    return_c = Value_c;
	}  
        else
    	    {return_c = "OPK_GET_EVALUATION_FAILURE";}    
    }
// ------------ RESTART
    else if (strcmp(QuelleFonction, "Get_restarts") == 0)
    {
 	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	    {local_restart = opk_get_nlcg_restarts(nlcg);}
	else if (strcasecmp (algorithm_name, "lbfgs") == 0)
	    {local_restart = opk_get_lbfgs_restarts(lbfgs);}
	else if (strcasecmp (algorithm_name, "vmlm") == 0)
	    {local_restart = opk_get_vmlm_restarts(vmlm);}
	else if (strcasecmp (algorithm_name, "vmlmn") == 0)
	    {local_restart = opk_get_vmlmn_restarts(vmlmn);}
	else 
	    {local_restart = opk_get_vmlmb_restarts(vmlmb);}
 
        if (local_restart != -1)
	{
	    Value_d = local_restart;
            sprintf(Value_c, "%lf", Value_d);
	    return_c = Value_c;
	}  
        else
    	    {return_c = "OPK_GET_RESTART_FAILURE";} 
    }
// ------------ STEP
    else if (strcmp(QuelleFonction, "Get_step") == 0)
    {
 	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	    {local_step = opk_get_nlcg_step(nlcg);}
	else if (strcasecmp (algorithm_name, "lbfgs") == 0)
	    {local_step = opk_get_lbfgs_step(lbfgs);}
	else if (strcasecmp (algorithm_name, "vmlm") == 0)
	    {local_step = opk_get_vmlm_step(vmlm);}
	else if (strcasecmp (algorithm_name, "vmlmn") == 0)
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
 	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	    {local_step = opk_get_nlcg_options(nlcg);}
	else if (strcasecmp (algorithm_name, "lbfgs") == 0)
	    {local_step = opk_get_lbfgs_options(lbfgs);}
	else if (strcasecmp (algorithm_name, "vmlm") == 0)
	    {local_step = opk_get_vmlm_options(vmlm);}
	else if (strcasecmp (algorithm_name, "vmlmn") == 0)
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
 	if (strcasecmp (algorithm_name, "nlcg") == 0) 
	    {local_step = opk_set_nlcg_options(nlcg);}
	else if (strcasecmp (algorithm_name, "lbfgs") == 0)
	    {local_step = opk_set_lbfgs_options(lbfgs);}
	else if (strcasecmp (algorithm_name, "vmlm") == 0)
	    {local_step = opk_set_vmlm_options(vmlm);}
	else if (strcasecmp (algorithm_name, "vmlmn") == 0)
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
	if (strcasecmp (algorithm_name, "vmlmn") != 0)
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
	if (strcasecmp (algorithm_name, "vmlmn") != 0)
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
 	if (strcasecmp (algorithm_name, "nlcg") != 0) 
	    {return_c = "ERROR : Get_fmin is relevant only with vmlmn algorithm";}
	else 
	    {local_status_fmin = opk_get_nlcg_fmin(nlcg, double* fmin);}
    else if (strcmp(QuelleFonction, "Set_fmin") == 0)
    {
 	if (strcasecmp (algorithm_name, "nlcg") != 0) 
	    {return_c = "ERROR : Get_fmin is relevant only with vmlmn algorithm";}
	else 
	    {local_status_fmin = opk_set_nlcg_fmin(nlcg, double fmin);}
    else if (strcmp(QuelleFonction, "Unset_fmin") == 0)
    {
 	if (strcasecmp (algorithm_name, "nlcg") != 0) 
	    {return_c = "ERROR : Get_fmin is relevant only with vmlmn algorithm";}
	else 
	    {local_status_fmin = opk_unset_nlcg_fmin(nlcg);}
    else if (strcmp(QuelleFonction, "Get_description") == 0)
    {
 	if (strcasecmp (algorithm_name, "nlcg") != 0) 
	    {return_c = "ERROR : Get_fmin is relevant only with vmlmn algorithm";}
	else 
	    {local_status_fmin = opk_get_nlcg_description(nlcg, char* str);}
    
    if (local_status_fmin == OPK_SUCCESS)
        {return_c = "OPK_SUCCESS";}   
    else
        {return_c = "OPK_FMIN_FAILURE";}
*/

/*
    fclose(fichier);
// On renvoit la valeur demandee sous forme de chaine de charactere
    return Py_BuildValue("s",return_c);
}
*/
// ------------------------------------------------------------------------------------------ 


/* ------------------------------------------------------------------------------------------
------------------------------------ FONCTION CLOSE -----------------------------------------
--------------------------------------------------------------------------------------------- */
static PyObject *Close (PyObject * self)
{

// Free workspace 
    OPK_DROP (vx);
    OPK_DROP (vg);
    OPK_DROP (vgp);
    OPK_DROP (vspace);

// On ferme l'algorithme utilise
    if (strcasecmp (algorithm_name, "nlcg") == 0) 
    	OPK_DROP (nlcg);
    else 
        OPK_DROP (vmlmb);

// Function does not return anything
  Py_RETURN_NONE;
}
// ------------------------------------------------------------------------------------------ 



// ------------------------------------------------------------------------------------------ 
//  Define functions in module
static PyMethodDef Methods[] = 
    {
    {"Initialisation", (PyCFunction)Initialisation, METH_VARARGS|METH_KEYWORDS, "lala"},
    {"Iterate", (PyCFunction)Iterate, METH_VARARGS, "lala"},
   // {"TaskInfo", (PyCFunction)TaskInfo, METH_VARARGS, "lala"},
    {"Close", (PyCFunction)Close, METH_NOARGS, "lala"},
    {NULL, NULL, 0, NULL}
    };

//module initialization */
	// PYTHON2
PyMODINIT_FUNC
initopkc_v3 (void)
{
    (void) Py_InitModule("opkc_v3", Methods);
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
// ------------------------------------------------------------------------------------------ 
