import numpy as np

class MyDailyLinearRegression:
    def __init__(self, lr=0.01, max_iter=100, sgd=False, n_sample=30, use_gradient_descent=False):
        self.lr = lr
        self.max_iter = max_iter
        self.sgd = sgd
        self.n_sample = n_sample
        self.use_gradient_descent = use_gradient_descent
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.c_[X, np.ones(X.shape[0])]
        if self.use_gradient_descent:
            self.coef_ = np.zeros(X.shape[1])
            for _ in range(self.max_iter):
                if self.sgd:
                    # Stochastic Gradient Descent
                    random_index = np.random.randint(X.shape[0])
                    X_sample = X[random_index:random_index+1]
                    y_sample = y[random_index:random_index+1]
                else:
                    # Batch Gradient Descent
                    X_sample = X
                    y_sample = y
                    
                gradients = -2 * X_sample.T.dot(y_sample - X_sample.dot(self.coef_)) / X_sample.shape[0]
                self.coef_ -= self.lr * gradients
        else:
            self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def predict(self, X):
        X = np.c_[X, np.ones(X.shape[0])]
        return X @ np.append(self.coef_, self.intercept_)

class MyDailyLinearRegressionRamadan:
    def __init__(self, lr=0.01, max_iter=100, sgd=False, n_sample=30, use_gradient_descent=False):
        self.lr = lr
        self.max_iter = max_iter
        self.sgd = sgd
        self.n_sample = n_sample
        self.use_gradient_descent = use_gradient_descent
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, ramadan, covid):
        X = np.c_[X, np.ones(X.shape[0])]
        X = np.c_[X, ramadan, covid]
        if self.use_gradient_descent:
            self.coef_ = np.zeros(X.shape[1])
            for _ in range(self.max_iter):
                if self.sgd:
                    # Stochastic Gradient Descent
                    random_index = np.random.randint(X.shape[0])
                    X_sample = X[random_index:random_index+1]
                    y_sample = y[random_index:random_index+1]
                else:
                    # Batch Gradient Descent
                    X_sample = X
                    y_sample = y
                    
                gradients = -2 * X_sample.T.dot(y_sample - X_sample.dot(self.coef_)) / X_sample.shape[0]
                self.coef_ -= self.lr * gradients
        else:
            self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def predict(self, X, ramadan, covid):
        X = np.c_[X, np.ones(X.shape[0])]
        X = np.c_[X, ramadan, covid]
        return X @ np.append(self.coef_, self.intercept_)
