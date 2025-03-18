import numpy as np
import matplotlib.pyplot as plt

class KCircles:
     
    def __init__(self, k_circles, n_features=None) -> None:
        self.k_circles = k_circles
        self.n_features = n_features
        self.radius = None
        self.histo = None

    def init_paramters(self, X):
        self.histo = None
        std = X.std(axis=0).reshape(self.n_features)
        med = np.median(X, axis=0).reshape(self.n_features)
        centers = []
        for j in range(self.n_features):
            s, m = std[j], med[j]    
            centers.append(m + s - 2*s*np.random.random(self.k_circles))

        self.centers = np.column_stack(centers)
        self.radius = np.random.rand(self.k_circles) + 0.5
     
    def fit(self, X, iter=2000, init=True) -> None:
        samples, features = X.shape
        self.n_features = features
        if init:
            self.init_paramters(X)
        if self.histo == None :
            self.histo = [self.centers.copy()]
        for i in range(iter): 
            D = np.sqrt(((X.reshape(1, samples, features)-self.centers.reshape(self.k_circles, 1, samples))**2).sum(axis=2)).T
            L = (D.copy() - np.array(self.radius).reshape(1, self.k_circles))**2
            y = L.argmin(axis=1)
            for j in range(self.k_circles):
                Dj = D[y==j, j]
                size = len(Dj)
                grad = -(X[y==j]-self.centers[j])/Dj.reshape(size, 1)
                grad = np.hstack((-2*np.ones((grad.shape[0], 1)), grad))
                grad = grad * (Dj-self.radius[j]).reshape(size, 1) / size
                grad = grad.sum(axis=0)
                self.radius[j] -= 1/10**2 * grad[0]
                self.centers[j] = self.centers[j] - 1/10**1 * grad[1:]
            self.histo.append(self.centers.copy())

    def multi_fit(self, X, iter, rep):
        loss = 10**9
        for i in range(rep):
            self.fit(X, iter)
            y_, loss_ = self.predict(X, loss=True)
            if loss_<loss:
                loss = loss_
                centers = self.centers.copy()
                radius = self.radius
                histo = self.histo            
            self.histo = None
        self.centers = centers
        self.radius = radius
        self.histo = histo
     
    def predict(self, X, loss=False):
        samples, features = X.shape
        D = np.sqrt(((X.reshape(1, samples, features)-self.centers.reshape(self.k_circles, 1, samples))**2).sum(axis=2)).T
        L = (D.copy() - np.array(self.radius).reshape(1, self.k_circles))**2
        if loss:
            return L.argmin(axis=1), L.min(axis=1).sum()
        return L.argmin(axis=1)
        
    def fit_predict(self, X, iter):
        self.fit(X, iter)
        return self.predict(X)