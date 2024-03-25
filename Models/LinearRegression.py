import numpy as np

class MyDailyLinearRegression:
    def __init__(self, lr=0.01, max_iter=100, sgd=False, n_sample=30):
        self.lr = lr
        self.max_iter = max_iter
        self.sgd = sgd
        self.n_sample = n_sample
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Добавляем столбец из единиц, чтобы учесть коэффициент смещения (intercept)
        X = np.c_[X, np.ones(X.shape[0])]
        
        # Вычисляем параметры модели аналитически используя метод наименьших квадратов
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def predict(self, X):
        # Добавляем столбец из единиц для учета коэффициента смещения (intercept)
        X = np.c_[X, np.ones(X.shape[0])]
        
        # Вычисляем предсказанные значения
        return X @ np.append(self.coef_, self.intercept_)
    
class MyDailyLinearRegressionRamadan:
    def __init__(self, lr, max_iter, sgd, n_sample):
        self.lr = lr
        self.max_iter = max_iter
        self.sgd = sgd
        self.n_sample = n_sample
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, ramadan, covid):
        # Добавляем столбец из единиц, чтобы учесть коэффициент смещения (intercept)
        X = np.c_[X, np.ones(X.shape[0])]
        
        # Вычисляем параметры модели аналитически используя метод наименьших квадратов
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def predict(self, X):
        # Добавляем столбец из единиц для учета коэффициента смещения (intercept)
        X = np.c_[X, np.ones(X.shape[0])]
        
        # Вычисляем предсказанные значения
        return X @ np.append(self.coef_, self.intercept_)