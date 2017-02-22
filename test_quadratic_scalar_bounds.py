'''
This test case is meant to check the support for scalar bounds (as opposed to arrays of values)
It currently does not pass and I don't see how to make it work, because it is unclear how
to pass the information about the bounds being non-array to the
>opk_new_optimizer(algorithm_method, NULL, type, n,
>                  bound_low, bound_low_arr,
>                  bound_up, bound_up_arr, NULL);
API defined in driver.c
'''
import numpy as np
import opkpy_v3_1


target = np.array([5, 5, 5, 5, 5], dtype='float32')

def fg(hypothesis):
    global target
    f = ((hypothesis - target) ** 2).sum()
    gradient = 2 * (hypothesis - target)
    return f, gradient

x0 = np.array([7, 3, 5, 2, 3], dtype="float32")

opt = opkpy_v3_1.Optimizer()
solution = opt.minimize(x0, fg, bl=0, bu=10, maxiter=3, maxeval=3, verbose=False)

print(solution)
assert((solution == target).all())