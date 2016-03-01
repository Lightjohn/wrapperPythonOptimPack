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

    def minimize(self, x, fg, g, bl=None, bu=None, algorithm="vmlmb", maxiter=500, maxeval=500, verbose=False):
        # Initialisation of the algorithm
        iteration = 0
        evaluation = 0
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
                evaluation += 1  # Increase evaluation
                task = opkc_v3_1.iterate(x, fx, g)  # Iterate
                # A new iterate is available
            elif task == self.OPK_TASK_NEW_X:
                iteration += 1  # Increase iteration
                task = opkc_v3_1.iterate(x, fx, g)  # Iterate
                # Algorithm has converged, solution is available
            elif task == self.OPK_TASK_FINAL_X:
                print("Algorithm has converged, solution is available")
                print("iteration = ", iteration, "     evaluation = ", evaluation)
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
            if iteration >= maxiter:
                print("Too much iteration\n")
                print("iteration = ", iteration, "     evaluation = ", evaluation)
                break
                # Too much evaluation of f and g, check OPK_TASK_COMPUTE_FG
            if evaluation >= maxeval:
                print("Too much evaluation\n")
                print("iteration = ", iteration, "     evaluation = ", evaluation)
                break
            if fx and verbose:
                print(" f(x) = ", fx)
        return x


    def __del__(self):
        opkc_v3_1.close()
