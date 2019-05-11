import numpy as np

from methods.general import CostCalculus


class LinearRegressor:
    @staticmethod
    def BGDRegressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance):

        # Define x0 = 1
        ones = np.ones((train_x.shape[0], 1))
        train_x = np.concatenate((ones, train_x), axis=1)
        val_x = np.concatenate((ones[0:val_x.shape[0]], val_x), axis=1)

        # Data dimensions
        n = train_x.shape[1]
        m = train_x.shape[0]

        # Set random parameters values [0,1) to start
        params = np.random.rand(n)
        params = np.array([params, ] * m)

        # Temporary parameters
        tmp_params = np.zeros(n)

        # Array error per iteration
        train_error = np.zeros(max_iterations + 1)
        val_error = np.zeros(max_iterations + 1)

        # Do process
        k = 1
        while k <= max_iterations:
            # Compute model h_theta(x)
            h = CostCalculus.cost(params, train_x, axis=1)

            # For each variable Xn
            for j in range(0, n):
                tmp = (h - train_y) * train_x[:, j]
                tmp_params[j] = np.sum(tmp) / m

            # Update coefficients
            params[0, :] = params[0, :] - learning_rate * tmp_params
            params = np.array([params[0, :], ] * m)

            # Compute Error
            train_error[k] = CostCalculus.compute_error(params, train_x, train_y, axis=1)

            # Validation Error
            val_error[k] = CostCalculus.compute_error(params[0:val_x.shape[0], :], val_x, val_y, axis=1)

            print('Iteration:', k, ', ( Training Error:', train_error[k], ', Validation Error:', val_error[k]), ')'

            # Stop criterion
            if k >= 2:
                if abs(train_error[k-1] - train_error[k]) <= tolerance:
                    break

            k = k + 1

        return params[0, :], train_error, val_error, k-1

    @staticmethod
    def SGDRegressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance):

        # Define x0 = 1
        ones = np.ones((train_x.shape[0], 1))
        train_x = np.concatenate((ones, train_x), axis=1)
        val_x = np.concatenate((ones[0:val_x.shape[0]], val_x), axis=1)

        # Data dimensions
        n = train_x.shape[1]
        m = train_x.shape[0]

        # Set random parameters values [0,1) to start
        params = np.random.rand(n)

        # Array error per iteration
        train_error = np.zeros(max_iterations + 1)
        val_error = np.zeros(max_iterations + 1)

        # Do process
        k = 1
        while k <= max_iterations:

            for i in range(0, m):
                h = CostCalculus.cost(params, train_x[i, :])
                # For each variable Xn
                for j in range(0, n):
                    params[j] = params[j] - learning_rate * (h - train_y[i]) * train_x[i, j]

            params_array = np.array([params, ] * m)

            # Compute Error
            train_error[k] = CostCalculus.compute_error(params_array, train_x, train_y, axis=1)

            # Validation Error
            val_error[k] = CostCalculus.compute_error(params_array[0:val_x.shape[0], :], val_x, val_y, axis=1)

            print('Iteration:', k, ', ( Training Error:', train_error[k], ', Validation Error:', val_error[k]), ')'

            # Stop criterion
            if k >= 2:
                if abs(train_error[k-1] - train_error[k]) <= tolerance:
                    break

            k = k + 1

        return params, train_error, val_error, k-1

    @staticmethod
    def MBGDRegressor(train_x, train_y, val_x, val_y, b, max_iterations, learning_rate, tolerance):

        # Define x0 = 1
        ones = np.ones((train_x.shape[0], 1))
        train_x = np.concatenate((ones, train_x), axis=1)
        val_x = np.concatenate((ones[0:val_x.shape[0]], val_x), axis=1)

        # Data dimensions
        n = train_x.shape[1]
        m = train_x.shape[0]

        # Set random parameters values [0,1) to start
        params = np.random.rand(n)
        params = np.array([params, ] * b)

        # Temporary parameters
        tmp_params = np.zeros(n)

        # Array error per iteration
        train_error = np.zeros(max_iterations + 1)
        val_error = np.zeros(max_iterations + 1)

        # Do process
        k = 1
        while k <= max_iterations:

            for i in range(0, m, b):
                step = b
                if i+b >= m:
                    step = i+b - m
                h = CostCalculus.cost(params[0:step, :], train_x[i:i+step, :], axis=1)
                # For each variable Xn
                for j in range(0, n):
                    tmp = (h - train_y[i:i+step]) * train_x[i:i+step, j]
                    tmp_params[j] = np.sum(tmp) / step

                # Update coefficients
                params[0, :] = params[0, :] - learning_rate * tmp_params
                params = np.array([params[0, :], ] * step)

            params_array = np.array([params[0, :], ] * m)

            # Compute Error
            train_error[k] = CostCalculus.compute_error(params_array, train_x, train_y, axis=1)

            # Validation Error
            val_error[k] = CostCalculus.compute_error(params_array[0:val_x.shape[0], :], val_x, val_y, axis=1)

            print('Iteration:', k, ', ( Training Error:', train_error[k], ', Validation Error:', val_error[k]), ')'

            if k >= 2:
                if abs(train_error[k - 1] - train_error[k]) <= tolerance:
                    break

            k = k + 1

        return params[0, :], train_error, val_error, k-1
