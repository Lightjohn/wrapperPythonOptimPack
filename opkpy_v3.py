# opkpy.py --
#
# OPKPY is OptimPack for PYTHON.
#
#-----------------------------------------------------------------------------
#
# This file is part of OptimPack (https://github.com/emmt/OptimPack).
# 
# Copyright (C) 2014, 2015 Eric Thiebaut
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
#------------------------------------------------------------------------------

import numpy as np

import opkc_v3

#############################################################################
##  A FAIRE

## Il y a des problemes au niveau du type, des var sont declares comme "double"
# mais selon le type ca change.

# fmin marche pas : undefined value

## linesearch : bien comprendre ce que c'est, si c'est un algo comme nlcg et
# vlmb. 
# Pareil pour bound

## voir comment tester un reel

## le get_method pour nlcg renvoit 520 qui ne correspond a aucune methode

##
# method et flag pareil? 
#pour vmlmb, il existe une structure "methode" mais pour les autres ce sont
# des unsigned int. Pourquoi pas de conflit entre les types?
# OPK_EMULATE_BLMVM est un flag
# opk_vmlmb_method_t est la methode.

## finir taskinfo

#############################################################################


# Documentation of function opk_initialisation.
"""Performs the creation of vspace and the optimizer as requested by the user.
 
The input arguments are more or less the same than for opk_minimize :
x, algorithm, linesearch, autostep, nlcg, vmlmb, delta, epsilon, delta_given,
epsilon_given, gatol, grtol, bl, bu, bound_given, mem, powell, single, limited
These cannot be optional arguments for some may be modified by the user.

It returns the value of the next task (supposedlyOPK_TASK_COMPUTE_FG).
"""

# Documentation of function opk_iteration.
"""Performs an iteration of x given the algorithm,linesearch, etc.

The input arguments are :  
x --> the current point of the function.
fx --> the value of the function to minimize at current point.
g --> the function gradient at current point.

It returns the value of the next task. 
"""

# Documentation of function opk_taskinfo.
"""Used to query some members of the optimizer instance.

The input argument is the name of the action that needs to be 
performed. It is a char string. Possible values are :
    "get_method" --> Returns the minimization method used.
                     See help - opk_minimize for the list of possible output.
    "get_size" --> Returns the number of variables of the problem.
    "get_type" --> Returns the type of the variables (float or double).
    "get_task" --> Returns the current task. Possible output are :
                   "OPK_TASK_START", "OPK_TASK_COMPUTE_FG", "OPK_TASK_NEW",
                   "OPK_TASK_FINAL_X", "OPK_TASK_WARNING", "OPK_TASK_ERROR".
    "get_status" --> Returns the current status. 29 possible values.
                     See optimpack.h for the list of possible output.
    "get_reason" --> Retrieves a textual description for a given status, the 
                     described status is obtained via Get_status. Relevant 
                     only for limited minimization.
    "get_iterations" --> Returns the number of iterations, supposeldy egal to
                         iteration.
    "get_evaluations" --> Returns the number of evaluations, supposeldy egal 
                          to evaluation.
    "get_restarts" --> Returns the number of times the algorithm has restarted.    
    "get_step" --> Returns the current step length.
    "get_gnorm" --> Returns the Euclidean norm of the gradient.
    "get_description" --> Returns a description of the minimization method.
                         Maximum size of the description is 255 bytes.
    "get_options" --> Returns the options (delta, epsilon, grtol, gatol, 
                      stpmin, stpmax) of the optimizer.
    "get_beta" --> Returns beta, the factor for the vector "y" of nlcg
                   algorithm. 
    "get_fmin" --> Returns fmin for the nlcg algorithm. 
    "get_mp" --> Returns the actual number of memorized steps for vmlmb.
    "get_s" --> Returns the variable difference "s[k-j]" where "k" is the
                current iteration number. See below.
    "get_y" --> Returns the variable difference "y[k-j]" where "k" is the
                current iteration number. See below.
                
The second (optional) argument is used to get a given memorized variable 
change. 
Variable metric methods store variable and gradient changes for the few last
steps to measure the effect of the Hessian.  Using pseudo-code notation the
following `(s,y)` pairs are memorized:
 * s[k-j] = x[k-j+1] - x[k-j]     // variable change
 * y[k-j] = g[k-j+1] - g[k-j]     // gradient change
with `x[k]` and `g[k]` the variables and corresponding gradient at `k`-th
iteration and `j=1,...,mp` the relative index of the saved pair.
See optimpack.h for help. 0 < j < mp
 
The returned value is always a char string, even if it represents a double.
"""

# Documentation of function opk_close.
"""Drops the object defined as global variables.

A copy of xneeds to be done prior to the call of Close for x is lost
afterwards. Function has no input and no output.
"""
#############################################################################
# Constant variables
DELTA_DEFAULT = 5e-2
EPSILON_DEFAULT = 1e-2
NULL = 0


def opk_minimize(x_in, fg, g, bl=NULL, bu=NULL, algorithm="nlcg", linesearch="quadratic", autostep="ShannoPhua",
                 nlcg="FletcherReeves", vmlmb="lbfgs", delta=DELTA_DEFAULT, epsilon=EPSILON_DEFAULT, gatol=1.0e-6,
                 grtol=0.0, maxiter=50, maxeval=50, mem=5, powell=False, verbose=0, limited=NULL):
    """Minimizes a function given a starting point and it's gradient.
    
    It keeps x_in untuched and returns the value of the final point x_out 
    so that f(x_out) = 0.    
    The input arguments are :
    x_in --> the starting point of the function. x contains all the variables 
          of the fonction to minimize in a numpy array of dimension 1, and type 
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
    algorithm --> The name of the algorithm chosen to minimize the function.
                  It is a char string. Possible values are :
             "nlcg" --> Non Linear Conjugate Gradient
             "vmlmb" --> Variable Metric Method 
                  Default value is "nlcg"
    linesearch --> The name of the linesearch chosen to minimize the function.
                   It is a char string. Possible values are :
             "quadratic" --> 
             "armijo" --> 
             "cubic" --> 
             "nonmonotone" -->
                  Default value is "quadratic".
    autostep --> The name of the autostep chosen to minimize the function.
                 It is a char string. Possible values are :
             "ShannoPhua" --> 
             "OrenSpedicato" --> 
             "BarzilaiBorwein" --> 
                 Default value is "ShannoPhua".
    nlcg -->  The name of the sub method chosen to minimize the function with 
              the nlcg algorithm. It is a char string. Possible values are :
             "FletcherReeves" --> 
             "HestenesStiefel" --> 
             "PolakRibierePolyak" --> 
             "Fletcher" --> 
             "LiuStorey" --> 
             "DaiYuan" --> 
             "PerryShanno" --> 
             "HagerZhang" --> 
              Default value is "FletcherReeves".
    vmlmb --> The name of the sub method chosen to minimize the function with 
              the vmlmb algorithm. It is a char string. Possible values are :
             "blmvm" --> 
             "vmlmb" --> 
             "lbfgs" --> 
              Default value is "lbfgs".             
    delta --> Relative size for a small step. Default value is 5e-2.
    epsilon --> Threshold to accept descent direction. Default value is 1e-2.
    gatol --> Absolute threshold for the norm or the projected gradient for 
              convergence. Default value is 1.0e-6.
    grtol --> Relative threshold for the norm or the projected gradient 
              (relative to GPINIT the norm of the initial projected gradient)
              for convergence. Default value is 0.0.
    maxiter --> The maximum number of iteration. Default value is 50.
                If no limit is required, set to None.
    maxeval --> The maximum number of evaluation. Default value is 50.
                If no limit is required, set to None.
    mem --> Number of step memorized by the limited memory variable metric.
            Default value is 5.
    powell --> Default value is False.
    verbose --> To be set to a value other than NULL if the user wants more
                information about the minimization process. 
                Default value is 0.
    limited --> To be set to 1 if the required algorithm is of limited
                memory. Default value is 0.
    """

    # Initialisation of the algorithm
    print "\n"
    iteration = 0
    evaluation = 0
    task = "OPK_TASK_START"
    x = x_in.copy()
    x_out = x.copy()
    bound_given = 0
    single = 0

    # Input arguments tests
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
    # bl, bu, mem
    if (isinstance(bl, float) == False) and (isinstance(bl, int) == False):
        print("ERROR :bl must be a real number")
    if (isinstance(bu, float) == False) and (isinstance(bu, int) == False):
        print("ERROR :bu must be a real number")
        task = "INPUT_ERROR"        
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
        print("WARNING: User specified a search direction for algorithm nlcg \
              but is running algorithm vmlmb")
    if (algorithm != "vmlmb") and (vmlmb != "lbfgs"):
        print("WARNING: User specified a search direction for algorithm vmlmb \
              but is running algorithm nlcg")
 
    # Initialization             
    task = opkc_v3.opk_initialisation(x, algorithm, linesearch, autostep, nlcg, 
                                  vmlmb, delta, epsilon, delta_given,
                                  epsilon_given, gatol, grtol, bl, bu, 
                                  bound_given, mem, powell, single, limited)          

    # Beginning of the algorithm
    while True:
        if (maxeval != None) and (evaluation > maxeval):
            break
        if (maxiter != None) and (iteration > maxiter):
            break
        if (task == "OPK_TASK_COMPUTE_FG"):
            # Caller must compute f(x) and g(x).
            evaluation += 1          
            fx = fg(x, g)  
        elif (task == "OPK_TASK_NEW_X") :
            # A new iterate is available                
            iteration += 1  
        else :
            break
        # Comment and iterate
        if verbose != NULL:
            print "-----------  iteration n",iteration, ", evaluation n", evaluation, "  -----------"
        print "f(x) = ", fx
            #print "x = ", x                
        task = opkc_v3.opk_iteration(x, fx, g)              
                              
    if task == "OPK_TASK_FINAL_X":
        # Algorithm has converged, solution is available  
        print"Algorithm has converged in",iteration,"iterations and",evaluation,"evaluation. Solution is available"
    elif task == "OPK_TASK_WARNING":
        # Algorithm terminated with a warning
        print"ERROR : Algorithm terminated with a warning"
        print "reason = ",opkc_v3.opk_taskInfo("get_reason")
    elif task == "OPK_TASK_ERROR":
        # An error has ocurred
        print "ERROR : OPK_TASK_ERROR has occured"
        print "reason = ",opkc_v3.opk_taskInfo("get_reason")
    elif task == "INPUT_ERROR":
        # Error in the variable input
        print"ERROR : Input error, check all the input of function optimize"
    elif iteration >= maxiter:
        # Too much iterations, check OPK_TASK_NEW_X
        print"WARNING : Too much iteration\n"
    elif evaluation >= maxeval:
        # Too much evaluation of f and g, check OPK_TASK_COMPUTE_FG
        print"WARNING : Too much evaluation\n"
    else:
        # Unknown problem has occured
        print"ERROR : Unknown problem has occured"
        print "reason = ",opkc_v3.opk_taskInfo("get_reason")   
    # Destruction of the optimizer 
    if verbose != NULL:
        info = opkc_v3.opk_taskInfo("get_method")    
        print "method = ",info
        info = opkc_v3.opk_taskInfo("get_size")    
        print "size = ",info
        info = opkc_v3.opk_taskInfo("get_type")    
        print "type = ",info   
        info = opkc_v3.opk_taskInfo("get_task")    
        print "tast = ",info   
        info = opkc_v3.opk_taskInfo("get_status")    
        print "status = ",info   
        info = opkc_v3.opk_taskInfo("get_iterations")    
        print "iteration = ",info, iteration
        info = opkc_v3.opk_taskInfo("get_evaluations")    
        print "evaluation = ",info, evaluation
        info = opkc_v3.opk_taskInfo("get_restarts")    
        print "restarts = ",info
        info = opkc_v3.opk_taskInfo("get_reason")    
        print "reason = ",info
        info = opkc_v3.opk_taskInfo("get_step")    
        print "step = ",info
        info = opkc_v3.opk_taskInfo("get_gnorm")    
        print "gnorm = ",info
        info = opkc_v3.opk_taskInfo("get_description")    
        print "description = ",info
        info = opkc_v3.opk_taskInfo("get_options")    
        print "options = ",info
        info = opkc_v3.opk_taskInfo("get_beta")    
        print "beta = ",info    
        info = opkc_v3.opk_taskInfo("get_fmin")    
        print "fmin = ",info    
        info = opkc_v3.opk_taskInfo("get_mp")    
        print "mp = ",info    
        info = opkc_v3.opk_taskInfo("get_s", 1)    
        print "s = ",info 
        info = opkc_v3.opk_taskInfo("get_y", 1)    
        print "y = ",info 
     
    x_out = x.copy()    
    opkc_v3.opk_close()   

    return x_out
