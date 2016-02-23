# * opkpy.py --
# *
# * OPKPY is OptimPack for PYTHON.
# *
# *-----------------------------------------------------------------------------
# *
# * This file is part of OptimPack (https://github.com/emmt/OptimPack).
# *
# * Copyright (C) 2014, 2015 Eric Thiebaut
# *
# * Permission is hereby granted, free of charge, to any person obtaining a copy
# * of this software and associated documentation files (the "Software"), to
# * deal in the Software without restriction, including without limitation the
# * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# * sell copies of the Software, and to permit persons to whom the Software is
# * furnished to do so, subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be included in
# * all copies or substantial portions of the Software.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# * IN THE SOFTWARE.
# *
# *-----------------------------------------------------------------------------

import numpy as np
import opkc_v3

#############################################################################
##  A FAIRE
# tester que ce qui est dans x est bien un float.

## avec verbose, ajouter les get_status

## Changer les valeurs de sortie pour get_s et get_y
# les get_options et set_options ont des arguments compliques, a voir
# pareil pour les fonctions specifiques d'nlcg

## Verifier les noms des algos, leurs parametres d'entree sortie, p-e plus de description
## Documenter mem, powell, verbose, linesearch autostep, nlcg

## dans Iterate et TaskInfo attention a nlcg pas comme les autres 

#############################################################################


#############################################################################
## DECLARATION ET DOCUMENTATION

# extern Initialisation
"""      ---------- DOCUMENT: Initialisation ----------
The function Initialisation performs the creation of vspace and the optimizer 
as requested by the user. It returns the value of the next task (supposedly
 OPK_TASK_COMPUTE_FG).
 
The input arguments are more or less the same than for opk_minimize. 
These cannot be optional arguments for some may be modified by the user.
    
"""

# extern Iterate
"""      ---------- DOCUMENT: Iterate ----------
The function Iterate performs an iteration of x given the algorithm,
linesearch, etc given in parameter of opk_minimize. It returns the value of
the next task.

The input arguments are :  
x --> the current point of the function.
fx --> the value of the function to minimize at current point.
g --> the function gradient at current point.
    
"""

# extern TaskInfo
"""      ---------- DOCUMENT: Taskinfo ----------
The function TaskInfo returns the information asked by the user in 
parameter, regarding the current optimizer.
The returned value is always a char string, even if it represents a double.

The input argument is the name of the action that needs to be 
performed. It is a char string. Possible values are :
    "Get_task" --> 
    "Get_status" -->  
    "Get_iteration" --> 
    "Get_evaluation" --> 
    "Get_restarts" -->          
    "Get_step" -->   
    
"""

# extern Close
"""      ---------- DOCUMENT: Close ----------
The function Close drops the object defined as global variables. A copy of x
needs to be done prior to the call of Close for x is lost afterwards.
No input, no output.
    
"""
#############################################################################
# Input values
DELTA_DEFAULT = 5e-2
EPSILON_DEFAULT = 1e-2
NULL = 0


def opk_minimize(x, fg, g, bl=NULL, bu=NULL, algorithm="nlcg", linesearch="quadratic", autostep="ShannoPhua",
                 nlcg="FletcherReeves", vmlmb="lbfgs", delta=DELTA_DEFAULT, epsilon=EPSILON_DEFAULT, gatol=1.0e-6,
                 grtol=0.0, maxiter=500, maxeval=500, mem=5, powell=False, verbose=0, limited=NULL):
    """      ---------- DOCUMENT: opk_minimize ----------
        
    The function opk_minimize minimizes a function passed by the user following 
    a given algorithm, linesearch, etc. It does not return anything but the 
    value of the current point x is constantly uploaded in parameter x.
    
    The input arguments are :
    x --> the starting point of the function. x contains all the variables of 
          the fonction to minimize in a numpy array of dimension 1, and type 
          double. Exemple : x = np.array([-1, 1], dtype="double")
    fg--> function which takes two arguments x and gx and which, for the given
          variables x, stores the gradient of the function in gx and returns 
          the value of the function to minimize. Exemple : fx= fg(x,gx) where
          fx = Rosenbrock(x) and gx is computed into grad(Rosenbrock(x))
    g --> the gradient of the function to minimize at x. g is of same size and 
          type as x. The input value does not matter as it will be calculated
          with fg. Exemple : g = np.array([-22.4 , -16.24], dtype="double")
          
    ------- Other arguments are optional 
    
    bl --> lower bound for "vmlmb" algorithm. Default value is 0.
    bu --> upper bound for "vmlmb" algorithm. Default value is 1e6.
    algorithm --> The name of the algorithm chosen to minimise the function.
                  It is a char string. Possible values are :
             "nlcg" --> Non Linear Conjugate Gradient
             "vmlmb" --> Variable Metric Method 
                Default value is "nlcg"
    linesearch --> Default value is "quadratic".
             "quadratic" --> 
             "Armijo" --> 
             "cubic" --> 
             "nonmonotone" -->
    autostep --> Default value is "ShannoPhua".
             "ShannoPhua" --> 
             "OrenSpedicato" --> 
             "BarzilaiBorwein" --> 
    nlcg --> Default value is "FletcherReeves".
             "FletcherReeves" --> 
             "HestenesStiefel" --> 
             "PolakRibierePolyak" --> 
             "Fletcher" --> 
             "LiuStorey" --> 
             "DaiYuan" --> 
             "PerryShanno" --> 
             "HagerZhang" --> 
    vmlmb --> Default value is "lbfgs".
             "blmvm" --> 
             "vmlmb" --> 
             "lbfgs" --> 
    delta --> Relative size for a small step. Default value is 5e-2.
    epsilon --> Threshold to accept descent direction. Default value is 1e-2.
    gatol --> Absolute threshold for the norm or the projected gradient for 
              convergence. Default value is 1.0e-6.
    grtol --> Relative threshold for the norm or the projected gradient 
              (relative to GPINIT the norm of the initial projected gradient)
              for convergence. Default value is 0.0.
    maxiter --> The maximum number of iteration. Default value is 500.
    maxeval --> The maximum number of evaluation. Default value is 500.
    mem --> Number of step memorized by the limited memory variable metric.
            Default value is 5.
    powell --> Default value is False.
    verbose --> To be set to a value other than NULL if the user wants more
                information about the minimization process. 
                Default value is 0.
    limited --> To be set to 1 if the required algorithm is of limited
                memory. Default value is 0.
    
             --------------------------------------------"""

    # Initialisation of the algorithm
    iteration = 0
    evaluation = 0
    task = "OPK_TASK_START"
    error = False
    x_final = x.copy()
    bound_given = 0
    single = 0

    # Tests to check the type of the entries
    # size of x
    if (isinstance(x, np.ndarray) == False) or (len(x.shape) != 1):
        print("ERROR : x must be of type numpy.ndarray and of dimension 1")
        task = "INPUT_ERROR"
    # type of x
    if isinstance(x[0], np.float32):
        single = 1
    elif isinstance(x[0], np.float64):
        single = 0
    else:
        print("ERROR :x elements must be of type float")
        task = "INPUT_ERROR"
    fx = fg(x, g)
    # type of f
    if (isinstance(fx, np.float) == False) and (isinstance(fx, np.float64) == False):
        print("ERROR :fg must return a float")
        task = "INPUT_ERROR"
    # size of g    
    if (isinstance(g, np.ndarray) == False) or (len(g.shape) != 1):
        print("ERROR :g must be of type numpy.ndarray  and of dimension 1")
        task = "INPUT_ERROR"

    # Input arguments
    # Delta
    if delta != DELTA_DEFAULT:
        delta_given = 1
    else:
        delta_given = 0
    # epsilon
    if epsilon != EPSILON_DEFAULT:
        epsilon_given = 1
    else:
        epsilon_given = 0
        # bl, bu, mem
    if (algorithm == "nlcg") and ((bl != NULL) or (bu != NULL)):
        print("WARNING: User specified a bound for algorithm nlcg : irrelevant")
    elif (algorithm == "nlcg") and (mem != 5):
        print("WARNING: User specified a mem for algorithm nlcg : irrelevant")
    elif (algorithm == "vmlmb") and (bl != NULL) and (bu == NULL):
        bound_given = 1
    elif (algorithm == "vmlmb") and (bl == NULL) and (bu != NULL):
        bound_given = 2
    elif (algorithm == "vmlmb") and (bl != NULL) and (bu != NULL):
        bound_given = 3
    else:
        bound_given = 0
    # verbose
    if verbose != NULL:
        print("ALGO: ", algorithm, "LINE: ", linesearch, "AUTO: ", autostep)
    # gatol
    if gatol < 0:
        print("ERROR: bad input for gatol")
        task = "INPUT_ERROR"
        # grtol
    elif grtol < 0:
        print("ERROR: bad input for grtol")
        task = "INPUT_ERROR"
        # nlcg
    if (algorithm != "nlcg") and (nlcg != "FletcherReeves"):
        print("WARNING: User specified a search direction for algorithm vmlmb")

    ## tests
    #    task = opkc_v3.Initialisation(x,algorithm,linesearch,autostep,nlcg,vmlmb,
    #       delta,epsilon,delta_given, epsilon_given, gatol,grtol,bl,bu,bound_given,mem,powell,single,limited)
    #    print "task = ",task
    #    task = opkc_v3.Iterate(x,fx,g,limited)
    #    print "task = ",task

    # Beginning of the algorithm
    while True:
        # Caller must call `start` method
        if task == "OPK_TASK_START":
            task = opkc_v3.Initialisation(x, algorithm, linesearch, autostep, nlcg, vmlmb, delta, epsilon, delta_given,
                                       epsilon_given, gatol, grtol, bl, bu, bound_given, mem, powell, single, limited)
            print("task = ", task)
            # Caller must compute f(x) and g(x).
        elif task == "OPK_TASK_COMPUTE_FG":
            fx = fg(x, g)  # Compute f and g
            evaluation += 1  # Increase evaluation
            task = opkc_v3.Iterate(x, fx, g, limited)  # Iterate
            print("task = ", task)
            # A new iterate is available
        elif task == "OPK_TASK_NEW_X":
            iteration += 1  # Increase iteration
            task = opkc_v3.Iterate(x, fx, g, limited)  # Iterate
            print("task = ", task)
            # Algorithm has converged, solution is available
        elif task == "OPK_TASK_FINAL_X":
            x_final = x.copy()
            opkc_v3.Close()
            print("Algorithm has converged, solution is available")
            print("iteration = ", iteration, "     evaluation = ", evaluation)
            break
            # Algorithm terminated with a warning
        elif task == "OPK_TASK_WARNING":
            print("Algorithm terminated with a warning")
            error = True
            # An error has ocurred
        elif task == "OPK_TASK_ERROR":
            print("ERROR :OPK_TASK_ERROR has occured")
            error = True
            # Error in the variable input
        elif task == "INPUT_ERROR":
            print("ERROR :Input error, check all the input of function optimize")
            error = True
            # Unknown task has been asked
        else:
            print("ERROR :Unknown task has been asked")
            error = True
            # An error has occured
        if error:
            break
            # Too much iterations, check OPK_TASK_NEW_X
        if iteration >= maxiter:
            print("Too much iteration\n")
            print("iteration = ", iteration, "     evaluation = ", evaluation)
            break
            # Too much evaluation of f and g, check OPK_TASK_COMPUTE_FG
        if evaluation >= maxeval:
            print("Too much evaluation\n")
            print("iteration = ", iteration, "     evaluation = ", evaluation)
            break

        print(" f(x) = ", fx)
    return x_final
