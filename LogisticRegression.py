import numpy as np

class LogisticRegression(object):

    def __init__(self, eta = 0.01, n_iter = 1000):

        self.eta = eta
        self.n_iter = n_iter
        self.weight = None
        self.bias = None
        self.sigmoid = self.sigmoid
        self.n_samples = None
        
        
        


    def sigmoid(self, z):
        self.sig_function = 1/ (1 + np.exp(-z))
        self.prediction =  np.where(self.sig_function >= 0.5, 1, 0)
        return self
        
        

    def fit(self, X, y):

        self.n_samples, self.n_features =  X.shape
        self.weight = np.zeros(self.n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            
            z = np.dot(X, self.weight) + self.bias
            

            predicted_y = self.sigmoid(z).prediction

            dw = (1/(self.n_samples)) * (np.dot(X.T,predicted_y - y))
            db = (1/(self.n_samples)) * (np.sum(predicted_y - y))

            self.weight -= (self.eta * dw)
            self.bias -= (self.eta * db)
        



    def predict(self, X):
        z = np.dot(X, self.weight) + self.bias
        return self.sigmoid(z).prediction

    def estimate(self, X):
        z = np.dot(X, self.weight) + self.bias
        self.estimate_ = self.sigmoid(z).sig_function
        return self.estimate_
            

        

        
            
        

        
