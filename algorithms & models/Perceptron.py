import numpy as np

class Perceptron(object):

    def __init__(self, lr = 0.01, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weight = None
        self.bias = None
        self.activation_func = self.activation_function

    def activation_function(self, z):
        return np.where(z >= 0, 1, 0)



    def fit(self, X, y):
        n_samples, n_features = X.shape

        #weights initialization
        self.weight = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        
        for _ in range(self.n_iter):
            for index, x in enumerate(X):
                linear_output = np.dot(x, self.weight) + self.bias
                y_predicted = self.activation_func(linear_output)
                weight_update = self.lr * (y_[index] - y_predicted)
                self.weight += weight_update * x
                self.bias += weight_update
            
    def predict(self, X_test):
         linear_output = np.dot(X_test, self.weight) + self.bias
         return self.activation_func(linear_output)
         
