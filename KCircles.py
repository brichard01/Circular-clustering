import numpy as np

class KCircles:
     
    def __init__(self, k_circles, n_features=None) -> None:
        self.k_circles = k_circles
        self.n_features = n_features
        self.centers = None
        self.radius = None
        self.histo = {}

    def init_paramters(self, X):
        self.histo = {k:[] for k in self.histo}
        std = X.std(axis=0).reshape(self.n_features)
        med = np.median(X, axis=0).reshape(self.n_features)
        centers = []
        for j in range(self.n_features):
            s, m = std[j], med[j]    
            centers.append(m + s - 2*s*np.random.random(self.k_circles))

        self.centers = np.column_stack(centers)
        self.radius = (std**2).mean()**(1/2)*(np.random.rand(self.k_circles) + 0.5)

    def histo_track(self, *args):
        for arg in args:
            assert arg in ['centers', 'radius', 'loss']
            self.histo = {**self.histo, arg: []}

    def _histo_update(self, centers, radius, L):
        if 'centers' in self.histo:
            self.histo['centers'].append(centers.copy())
        if 'radius' in self.histo:
            self.histo['radius'].append(radius.copy())
        if 'loss' in self.histo:
            self.histo['loss'].append(L.min(axis=1).sum().item())
     
    def fit(self, X, iter=2000, learning_rates=(1e-2, 1e-1), init=True) -> None:
        lr_radius, lr_centers = learning_rates
        samples, features = X.shape
        self.n_features = features
        if init:
            self.init_paramters(X)
        for i in range(iter):
            D = np.sqrt(((X.reshape(1, samples, features)-self.centers.reshape(self.k_circles, 1, features))**2).sum(axis=2)).T
            L = (D.copy() - np.array(self.radius).reshape(1, self.k_circles))**2
            y = L.argmin(axis=1)
            for j in range(self.k_circles):
                Dj = D[y==j, j]
                size = len(Dj)
                grad = -(X[y==j]-self.centers[j])/Dj.reshape(size, 1)
                grad = np.hstack((-2*np.ones((grad.shape[0], 1)), grad))
                grad = grad * (Dj-self.radius[j]).reshape(size, 1) / size
                grad = grad.sum(axis=0)
                self.radius[j] -= lr_radius * grad[0]
                self.centers[j] = self.centers[j] - lr_centers * grad[1:]
            self._histo_update(self.centers, self.radius, L)

    def multi_fit(self, X, iter, rep):
        loss = 10**12
        for i in range(rep):
            self.fit(X, iter, init=True)
            _, loss_ = self.predict(X, loss=True)
            print(loss_)
            if loss_<loss:
                loss = loss_
                centers = self.centers.copy()
                radius = self.radius
                histo = self.histo
        self.centers = centers
        self.radius = radius
        self.histo = histo
     
    def predict(self, X, loss=False):
        samples, features = X.shape
        D = np.sqrt(((X.reshape(1, samples, features)-self.centers.reshape(self.k_circles, 1, features))**2).sum(axis=2)).T
        L = (D - np.array(self.radius).reshape(1, self.k_circles))**2
        if loss:
            return L.argmin(axis=1), L.min(axis=1).sum().item()
        return L.argmin(axis=1)
        
    def fit_predict(self, X, iter):
        self.fit(X, iter)
        return self.predict(X)