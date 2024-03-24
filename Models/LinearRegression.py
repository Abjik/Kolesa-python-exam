import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Optional

class MyLinearRegression:
    def __init__(self, fit_intercept: bool = True, lr: float = 0.01, max_iter: int = 100, sgd: bool = False, n_sample: int = 16):
        """
        Initialize the MyLinearRegression class.

        Parameters:
        fit_intercept (bool): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
        lr (float): The learning rate.
        max_iter (int): The maximum number of iterations.
        sgd (bool): Whether to use Stochastic Gradient Descent.
        n_sample (int): The number of samples to use for Stochastic Gradient Descent.
        """
        self.fit_intercept = fit_intercept
        self.w = None
        self.lr = lr
        self.max_iter = max_iter
        self.sgd = sgd
        self.n_sample = n_sample

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MyLinearRegression':
        """
        Fit the model according to the given training data.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        self: returns an instance of self.
        """
        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)

        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X

        self.losses = []

        for _ in range(self.max_iter):
            y_pred = self.predict(X)
            self.losses.append(mean_squared_error(y_pred, y))

            grad = self.__calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"
            self.w -= self.lr * grad

        return self

    def __calc_gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        y_pred (np.ndarray): Predicted values.

        Returns:
        grad (np.ndarray): The calculated gradient.
        """
        if self.sgd:
            inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)
            grad = 2 * (y_pred[inds] - y[inds])[:, np.newaxis] * X[inds]
        else:
            grad = 2 * (y_pred - y)[:, np.newaxis] * X
        return grad.mean(axis=0)

    def get_losses(self) -> list:
        """
        Get the losses.

        Returns:
        losses (list): The losses.
        """
        return self.losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters:
        X (np.ndarray): Samples.

        Returns:
        y_pred (np.ndarray): Returns predicted values.
        """
        n, k = X.shape
        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X
        return X_train @ self.w

    def get_weights(self) -> Optional[np.ndarray]:
        """
        Get the weights.

        Returns:
        w (np.ndarray): Returns the weights.
        """
        return self.w