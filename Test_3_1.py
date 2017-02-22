import numpy as np
import opkpy_v3_1

a = 1
b = 100

global_minimum = np.array([a, a ** 2])

def rosen_fun(vec):
    x, y = vec[0], vec[1]
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def rosen_grad(vec):
    x, y = vec[0], vec[1]
    df_dx = -2 * (a - 2 * b * x ** 3 + 2 * b * x * y - x)
    df_dy = b * (2 * y - 2 * x ** 2)
    return np.array([df_dx, df_dy]).astype('float32')

def fg_rosen(vec):
    return rosen_fun(vec), rosen_grad(vec)


if __name__ == "__main__":
    opt = opkpy_v3_1.Optimizer()
    bu = np.array([20, 10], dtype="float32")
    bl = np.array([0.2, 0.1], dtype="float32")
    x = np.array([-1.2, 1.0], dtype="float32")

    found_minimum = opt.minimize(x, fg_rosen, bu=bu, bl=bl, maxiter=50, maxeval=50)
    
    assert(np.isclose(found_minimum, global_minimum, atol=1e-4).all())

