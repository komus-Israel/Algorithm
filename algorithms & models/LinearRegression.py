import numpy as np



class LinearRegression(object):


    '''
        def __init__(self, eta = 0.01, n_iter = 1000)

            self.eta = 0.01. Learning rate
            selg.n_iter = 1000. Number of iterations

        def fit(self, X, y)

            X = trainig samples
            y = training targets

        def predict(self, X)

            X = testing samples

        def accuracy(self, y_test, predicted_y)
            the r_squared error used compute the accuracy of the predicted values to the original values

        def slope(self, X, y)
            slope of the data set. This is supposed to return the an array of the slope of each features...but this will return the mean of the slope

        def intercept(self, X,y)
            intercept of the data set. This is supposed to return the an array of the intercept of y in each features...but this will return the mean of the intercept


        example

            from LinearRegression import LinearRegression()

            reg = LinearRegression(eta = 0.009)
            model = reg.fit(X_train, y_train)
            predict = reg.predict(X_test, y_test)
            accuracy = reg.score(X_test, y_test) or unsing sklearn,

            from sklearn.metrics import r2_score

            accuracy = r2_score(X_test, y_test)

            slope = reg.slope(X_train, y_train)

            slope = reg.intercept(X_train, y_train)
            
    '''

    def __init__(self, eta = 0.01, n_iter = 1000):

        self.eta = eta
        self.n_iter = n_iter
        self.weight = None
        self.bias = None
        self.intercept = self.intercept
        self.slope = self.slope
        

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #initializing the weights for each column(features) as 0
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):

            predicted_y = np.dot(X, self.weight) + self.bias

            dw = (1/n_samples) * (np.dot(X.T, predicted_y - y))
            db = (1/n_samples) * (np.sum(predicted_y - y))

            self.weight -= (self.eta * dw)
            self.bias -= (self.eta * db)


    def predict(self, X):
        predicted_y = np.dot(X, self.weight) + self.bias
        return predicted_y

    def score(self, y_test, predicted_y):
        mss_predicted = np.mean((y_test - predicted_y)**2)
        y_test_mean = np.mean(y_test)

        mss_y_mean = np.mean((y_test - y_test_mean) ** 2)

        squared_error = 1 - (mss_predicted/mss_y_mean)

        return squared_error
        

    def slope(self, X, y):
        n_samples, n_features = X.shape
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        xy_mean = (np.dot(X.T, y)/n_samples)
        m = ((x_mean * y_mean) - xy_mean)/((x_mean **2) - np.mean(X**2))
        return np.mean(m)
        
    def intercept(self, X,y):
        m = self.slope(X,y)
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        b = y_mean - (m * x_mean)
        return np.mean(b)

    
        
        
