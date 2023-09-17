import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """

        DiffSquared = (pred - label)**2

        return np.sqrt(np.mean(DiffSquared))

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        """
        N = x.shape[0]
        if (x.ndim == 2):
            D = x.shape[1]
        else:
            D = 1

        feat = np.zeros((N, degree + 1, D))

        if (D != 1):
            for i in range(degree + 1):
                feat[:, i] = (x ** i)
                #Note i = 0 will result in the bias term
        else :
            feat = np.zeros((N, degree + 1))
            for i in range(degree + 1):
                feat[:, i] = (x ** i)


        return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,D) numpy array, where N is the number
                   of instances and D is the dimensionality
                   of each instance
            weight: (D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """

        
        return np.matmul(xtest, weight)

    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
        """
        x_plus = np.linalg.pinv(xtrain)
        weight = np.matmul(x_plus, ytrain)
        return weight

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        N = xtrain.shape[0]
        D = xtrain.shape[1]

        I = np.diag(np.ones(D))

        term = c_lambda * I
        term[0] = 0

        z_plus = np.matmul(np.linalg.inv(np.matmul(xtrain.T, xtrain) + term), xtrain.T)

        weight = np.matmul(z_plus, ytrain)

        return weight

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> float:
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the mean RMSE across all kfolds

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: float, average rmse error
        """
        
        N = X.shape[0]
        D = X.shape[1]
        
        fold_size = int(N / kfold)
        rmse = 0

        # Slicing - x[i:j:k]
        # where i is the starting index, j is the stopping index, and k is the step
        for i in range(kfold) :
             xtrain_start = X[0:(i * fold_size)]
             xtrain_end = X[((i+1) * fold_size):N]
             xtrain = np.concatenate((xtrain_start, xtrain_end), axis = 0)
             
             ytrain_start = y[0:(i * fold_size)]
             ytrain_end = y[((i+1) * fold_size):N]
             ytrain = np.concatenate((ytrain_start, ytrain_end), axis = 0)

             xtest = X[(i * fold_size):((i+1) * fold_size)]
             ytest = y[(i * fold_size):((i+1) * fold_size)]

             weight = self.ridge_fit_closed(xtrain, ytrain, c_lambda)
             pred = self.predict(xtest, weight)

             rmse = rmse + self.rmse(pred, ytest)
             
        return (rmse / kfold)

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants to search from
            kfold: Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the RMSE error achieved using the best_lambda
            error_list: list[float] list of errors for each lambda value given in lambda_list
        """

        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            error_list.append(err)
            if best_error is None or err < best_error:
                best_error = err
                best_lambda = lm

        return best_lambda, best_error, error_list
