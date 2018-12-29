# -*- coding: utf-8 -*-

import numpy as np
import inspect

def ReLU(x, differential=False):
    if differential:
        return 1.0 * (x >= 0)
    else:
        return x * (x >= 0)

def sigmoid(x, differential=False):
    sgm = 1.0 / ( 1 + np.exp(-1 * x))
    if differential:
        return (1 - sgm) * sgm
    else:
        return sgm

def cross_entropy_error(u, activation_function, T,  differential=False):
    N = 
    if differential:

    print("未実装 :" + inspect.currentframe().f_code.co_name)
