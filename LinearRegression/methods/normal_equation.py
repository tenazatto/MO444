import numpy as np

from methods.general import CostCalculus


class NormalEquation:
    @staticmethod
    def normal_equation(train_x, train_y, val_x, val_y):
        # Define x0 = 1
        ones = np.ones((train_x.shape[0], 1))
        train_x = np.concatenate((ones, train_x), axis=1)
        val_x = np.concatenate((ones[0:val_x.shape[0]], val_x), axis=1)

        # Find parameters
        params = np.linalg.inv(train_x.T.dot(train_x)).dot(train_x.T).dot(train_y)

        # Compute error
        params = np.array([params, ] * train_x.shape[0])
        train_error = CostCalculus.compute_error(params, train_x, train_y, axis=1)
        val_error = CostCalculus.compute_error(params[0:val_x.shape[0], :], val_x, val_y, axis=1)

        return params[0, :], train_error, val_error
