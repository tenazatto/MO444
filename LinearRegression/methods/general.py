import numpy as np


class CostCalculus:
    @staticmethod
    def cost(params, variables, axis=None):
        h = params*variables

        return np.sum(h, axis=axis)

    @staticmethod
    def compute_error(params, x, y, axis=None):
        m = x.shape[0]
        h = CostCalculus.cost(params, x, axis=axis)
        tmp = (h - y)*(h - y)

        return np.sum(tmp)/(2.0*m)
