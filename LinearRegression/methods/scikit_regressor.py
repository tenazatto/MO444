from sklearn import linear_model
from sklearn.metrics import mean_squared_error


class ScikitRegressor:
    @staticmethod
    def regressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance):

        model = linear_model.SGDRegressor(max_iter=max_iterations, learning_rate='constant', eta0=learning_rate,
                                          tol=tolerance)

        # Fit model
        model.fit(train_x, train_y)

        # Predict
        train_y_pred = model.predict(train_x)
        val_y_pred = model.predict(val_x)

        # Compute error
        train_error = mean_squared_error(train_y_pred, train_y)
        val_error = mean_squared_error(val_y_pred, val_y)

        return model, train_error, val_error
