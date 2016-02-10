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

"""
Created on Fri Feb  5 16:15:52 2016

@author: gheurtier
"""
                                                                    
import numpy as np
import opkc

#############################################################################
##  A FAIRE
# Dans les tests d'input, tester si f(x) est un Reel! Pas necessairement un float
# Peut etre testersa dimension ?

## Changer les valeurs de sortie pour get_s et get_y
# les get_options et set_options ont des arguments compliques, a voir
# pareil pour les fonctions specifiques d'nlcg

## Verifier les noms des algos, leurs parametres d'entree sortie, p-e plus de description
## Documenter mem, powell, verbose, linesearch autostep, nlcg

## dans Iterate et TaskInfo attention a nlcg pas comme les autres 

#############################################################################


#############################################################################
## DECLARATION ET DOCUMENTATION

##extern Iterate          
"""      ---------- DOCUMENT: Iterate ----------
The function Iterate performs an iteration of x given the algorithm,
linesearch, etc given in parameter of pok_minimize. It returns the value of
the next task.

The input arguments are :  
x --> the current point of the function.
fx --> the value of the function to minimize at current point.
g --> the function gradient at current point.
    
"""

# extern TaskInfo
"""      ---------- DOCUMENT: Taskinfo ----------
The function TaskInfo performs return the information asked by the user in 
parameter, regarding the current optimizer.
The returned value is always a char string, even if it represents a double.

The input argument is fonctionName, the name of the action that needs to be 
performed. It is a char string. Possible values are :
    "Get_task" --> 
    "Get_status" -->  
    "Get_iteration" --> 
    "Get_evaluation" --> 
    "Get_restarts" -->          
    "Get_step" -->   
    
"""
#############################################################################


def opk_minimize(x,fg,g,bl=0,bu=1e6,algorithm="nlcg",linesearch="quadratic",autostep="ShannoPhua",nlcg="FletcherReeves",delta=5e-2,epsilon=1e-2,gatol=1.0e-6,grtol=0.0,maxiter=50,maxeval=50,mem=5,powell=False,verbose=0):
    
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
             "vmlm" --> Variable Metric Method with Limited Memory
             "vmlmb" --> Bounded Variable Metric Method with Limited Memory
             "blmvm" --> Same?
             "lbfgs" --> Broyden-Fletcher-Goldfarb-Shanno
                Default value is "nlcg"
    linesearch --> Default value is "quadratic".
             "quadratic" --> 
             "Armijo" --> 
             "cubic" --> 
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
    delta --> Relative size for a small step. Default value is 5e-2.
    epsilon --> Threshold to accept descent direction. Default value is 1e-2.
    gatol --> Absolute threshold for the norm or the projected gradient for 
              convergence. Default value is 1.0e-6.
    grtol --> Relative threshold for the norm or the projected gradient 
              (relative to GPINIT the norm of the initial projected gradient)
              for convergence. Default value is 0.0.
    maxiter --> The maximum number of iteration. Default value is 50.
    maxeval --> The maximum number of evaluation. Default value is 50.
    mem --> Default value is 5.
    powell --> Default value is False.
    verbose --> Default value is 0.
    
             --------------------------------------------"""

    # Initialisation of the algorithm
    iteration = 0
    evaluation = 0
    task = "OPK_TASK_START"
    error = False
    
# Tests to check the entrys of the algorithm
    if ( (isinstance(x,np.ndarray) == False) or (len(x.shape) != 1) ):
        print "ERROR : x must be of type numpy.ndarray and of dimension 1"
        task = "INPUT_ERROR"
    fx = fg(x,g)
    if (isinstance(fx,np.float64) == False):
        print "ERROR :fg must return a numpy.ndarray of size 1"
        task = "INPUT_ERROR"
    if ( (isinstance(g,np.ndarray) == False) or (len(g.shape) != 1) ):
        print "ERROR :g must be of type numpy.ndarray  and of dimension 1"    
        task = "INPUT_ERROR"
        
    while True:
    # Caller must call `start` method
        if (task == "OPK_TASK_START"):
           task = opkc.Initialisation(x,fx,g,bl,bu,algorithm,linesearch,autostep,nlcg,delta,epsilon,gatol,grtol,maxiter,maxeval,mem,powell,verbose)     
    # Caller must compute f(x) and g(x).
        elif (task == "OPK_TASK_COMPUTE_FG") :
           fx = fg(x, g)                                     # Compute f and g
           evaluation = evaluation+1                         # Increase evaluation
           task = opkc.Iterate(x, fx, g);    # Iterate
    # A new iterate is available
        elif (task == "OPK_TASK_NEW_X") :
           iteration = iteration+1                           # Increase iteration
           task = opkc.Iterate( x, fx, g);    # Iterate                                     
    # Algorithm has converged, solution is available
        elif (task == "OPK_TASK_FINAL_X"):
           print"Algorithm has converged, solution is available"
           print"iteration = ",iteration, "     evaluation = ",evaluation    
           break
    # Algorithm terminated with a warning
        elif (task == "OPK_TASK_WARNING"):
           print"Algorithm terminated with a warning"
           break       
    # An error has ocurred
        elif (task == "OPK_TASK_ERROR"):
           print "ERROR :OPK_TASK_ERROR has occured"    
           error = True   
    # Error in the variable input
        elif (task == "INPUT_ERROR"):
           error = True             
    # Unknown task has been asked
        else:
           print "ERROR :Unknown task has been asked"    
           error = True   
    # An error has occured
        if (error == True):
            break
    # Too much iterations, check OPK_TASK_NEW_X
        if (iteration >= maxiter):
            print("Too much iteration\n")
            print"iteration = ",iteration, "     evaluation = ",evaluation      
            break
    # Too much evaluation of f and g, check OPK_TASK_COMPUTE_FG
        if (evaluation >= maxeval):
            print("Too much evaluation\n")
            print"iteration = ",iteration, "     evaluation = ",evaluation  
            break 

       # print" f(x) = ",fx
      
        