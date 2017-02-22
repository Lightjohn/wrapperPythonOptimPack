import numpy as np
import opkpy_v3_1


target = np.array([5, 5, 5, 5, 5], dtype='float32')

def fg(hypothesis):
    global target
    f = ((hypothesis - target) ** 2).sum()
    gradient = 2 * (hypothesis - target)
    return f, gradient

bu= np.array([10, 10, 10, 10, 10], dtype="float32")
bl= np.array([0, 0, 0, 0, 0], dtype="float32")
x0 = np.array([7, 3, 5, 2, 3], dtype="float32")


opt = opkpy_v3_1.Optimizer()
solution = opt.minimize(x0, fg, bl=bl, bu=bu, maxiter=3, maxeval=3, verbose=False)

assert((solution == target).all())