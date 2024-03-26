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
    # def __init__(self, learning_rate=0.01, max_iter=1000):
    #     self.learning_rate = learning_rate
    #     self.max_iter = max_iter
    #     self.coef_ = None
    #     self.intercept_ = None

    # def fit(self, X, y, ramadan, covid):
    #     n_samples, n_features = X.shape
        
    #     # Добавляем столбец из единиц, чтобы учесть коэффициент смещения (intercept)
    #     X = np.c_[X, np.ones(n_samples)]
        
    #     # Добавляем столбцы для учета Рамадана и COVID-19
    #     X = np.c_[X, ramadan, covid]

    #     # Инициализируем веса случайным образом
    #     np.random.seed(42)
    #     self.coef_ = np.random.randn(X.shape[1])

    #     # Градиентный спуск
    #     for _ in range(self.max_iter):
    #         # Вычисляем предсказания
    #         y_pred = self.predict(X)

    #         # Вычисляем градиент
    #         gradient = 2 * X.T @ (y_pred - y) / n_samples

    #         # Обновляем веса
    #         self.coef_ -= self.learning_rate * gradient

    # def predict(self, X, ramadan=None, covid=None):
    #     n_samples = X.shape[0]
        
    #     # Добавляем столбец из единиц для учета коэффициента смещения (intercept)
    #     X = np.c_[X, np.ones(n_samples)]

    #     # Добавляем столбцы для учета Рамадана и COVID-19, если они предоставлены
    #     if ramadan is not None and covid is not None:
    #         X = np.c_[X, ramadan, covid]

    #     # Вычисляем предсказания
    #     return X @ self.coef_

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, ramadan, covid):
        # Добавляем столбцы из единиц, чтобы учесть коэффициент смещения (intercept)
        X = np.c_[X, np.ones(X.shape[0])]
        
        # Добавляем столбцы для учета Рамадана и COVID-19
        X = np.c_[X, ramadan, covid]
        
        # Вычисляем параметры модели аналитически используя метод наименьших квадратов
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def predict(self, X, ramadan, covid):
        # Добавляем столбцы из единиц для учета коэффициента смещения (intercept)
        X = np.c_[X, np.ones(X.shape[0])]
        
        # Добавляем столбцы для учета Рамадана и COVID-19
        X = np.c_[X, ramadan, covid]
        
        # Вычисляем предсказанные значения
        return X @ np.append(self.coef_, self.intercept_)


