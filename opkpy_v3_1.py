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
import opkc_v3_1


class Optimizer:
    # Using same convention as in OptimPack
    OPK_TASK_ERROR = -1  # An error has ocurred.
    OPK_TASK_START = 0  # Caller must call `start` method.
    OPK_TASK_COMPUTE_FG = 1  # Caller must compute f(x) and g(x).
    OPK_TASK_NEW_X = 2  # A new iterate is available.
    OPK_TASK_FINAL_X = 3  # Algorithm has converged, solution is available.
    OPK_TASK_WARNING = 4  # Algorithm terminated with a warning.
    OPK_TASKS = {-1: "OPK_TASK_ERROR", 0: "OPK_TASK_START", 1: "OPK_TASK_COMPUTE_FG", 2: "OPK_TASK_NEW_X",
                 3: "OPK_TASK_FINAL_X", 4: "OPK_TASK_WARNING"}

    def __init__(self):
        self.iteration = 0
        self.evaluation = 0

    def minimize(self, x, fg, g, bl=None, bu=None, algorithm="vmlmb", maxiter=500, maxeval=500, verbose=False):
        # Initialisation of the algorithm

        task = self.OPK_TASK_START
        error = False
        bound_given = 0
        single = 0
        fx = None

        # Beginning of the algorithm
        while True:
            if verbose:
                print("task = ", self.OPK_TASKS[task])
            if task == self.OPK_TASK_START:
                task = opkc_v3_1.initialisation(x, algorithm, bu, bl)
                # Caller must compute f(x) and g(x).
            elif task == self.OPK_TASK_COMPUTE_FG:
                fx = fg(x, g)  # Compute f and g
                self.evaluation += 1  # Increase evaluation
                task = opkc_v3_1.iterate(x, fx, g)  # Iterate
                # A new iterate is available
            elif task == self.OPK_TASK_NEW_X:
                self.iteration += 1  # Increase iteration
                task = opkc_v3_1.iterate(x, fx, g)  # Iterate
                # Algorithm has converged, solution is available
            elif task == self.OPK_TASK_FINAL_X:
                print("Algorithm has converged, solution is available")
                print("iteration = ", self.iteration, "     evaluation = ", self.evaluation)
                break
                # Algorithm terminated with a warning
            elif task == self.OPK_TASK_WARNING:
                print("Algorithm terminated with a warning")
                error = True
                # An error has ocurred
            elif task == self.OPK_TASK_ERROR:
                print("ERROR OPK_TASK_ERROR has occurred, reason:", opkc_v3_1.get_reason())
                error = True
                # Error in the variable input
            else:
                print("ERROR Unknown task has been asked:", task)
                error = True
                # An error has occured
            if error:
                break
                # Too much iterations, check OPK_TASK_NEW_X
            if self.iteration >= maxiter:
                print("Too much iteration\n")
                print("iteration = ", self.iteration, "     evaluation = ", self.evaluation)
                break
                # Too much evaluation of f and g, check OPK_TASK_COMPUTE_FG
            if self.evaluation >= maxeval:
                print("Too much evaluation\n")
                print("iteration = ", self.iteration, "     evaluation = ", self.evaluation)
                break
            if fx and verbose:
                print(" f(x) = ", fx)
        print("PY counter:", opkc_v3_1.counter," Single:", opkc_v3_1.single, opkc_v3_1.get_gnorm())
        # self.close()
        return x


    # No need fo close or __del__ ^^ look for destructor_opk in c code.
    def close(self):    # if done in __del__ the capsule was already liberated 
        opkc_v3_1.close()

    def __del__(self):
        pass


