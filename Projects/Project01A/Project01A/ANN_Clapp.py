# David Clapp
# 2-27-2019

import numpy as np
    
class ANN:
    # Constructor
    def __init__(self, X, y, layer_sizes, activation = 'relu', weights = None):
        # Store the variables as class attributes.
        self.X = np.array(X)
        self.y = np.array(y)
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        # Store the number of training samples.
        self.N = len(self.X)
        
        # Store the number of predictors.
        self.M = len(self.X[0])
        assert self.M == layer_sizes[0]
        
        # Store the depth of the network.
        self.D = len(layer_sizes)
        
        # Get the number of classes and store these classes.
        self.classes = np.unique(y)
        
        # Store the number of classes in K.
        self.K = len(self.classes)
        assert self.K == layer_sizes[-1]
        
        # If relu is specified, store the relu function in a.
        if (activation == 'relu'):
            self.a = lambda x: np.maximum(0, x)     
        # If sigmoid is specified, store the sigmoid function in a.
        else:
            self.a = lambda x:1/(1 + np.exp(-x))
        
        # Store the values of weights.
        # If no weights were provided by the user...
        if (weights == None):
            # ...randomly generate them.
            self.weights = []
            for i in range (0, self.D-1):
                # Get the number of rows for the matrix by looking at the
                # number of nodes in the current layer and adding 1 for the bias.
                rows = self.layer_sizes[i] + 1
                
                # Get the number of columns for the matrix by looking at the
                # number of nodes in the next layer.
                columns = self.layer_sizes[i+1]
                
                # Randomly generate the weights between -1 and 1 for the matrix.
                wts = np.random.uniform(low = -1, high = 1, size = (rows,columns))
                
                # Append the matrix of randombly generated weights to self.weights.
                self.weights.append(wts)
        # If weights are provided by the user...
        else:
            # ...set self.weights to the user provided weights.
            self.weights = weights
            
        # Create and store the indicator matrix.
        self.T = np.array(self.y[:,None]==self.classes).astype(int)
    
    # Predict the probability that elements of X are of each class.
    def predict_proba(self, X):
        X = np.array(X)
        
        # Set A equal to X.
        A = X
        
        # Initialize activations to only include A.
        self.activations = [A]
        
        # Create a column of ones that has the same number of rows as A.
        ones = np.ones(shape = (A.shape[0], 1))

        # For each non input layer...
        for i in range (self.D - 1):
            # ...add a layer of ones to the front of A.
            A = np.hstack((ones, A))
            
            # Z is the dot product of A and the weights for the current layer.
            Z = np.dot(A, self.weights[i])
            
            # If at the output layer...
            if (i == self.D - 2):
                # ...apply the softmax function to Z and store it in A.
                A = self.softmax(Z)
            # If not at the output layer...
            else:
                # ...apply the activation function to Z and store it in A.
                A = self.a(Z)
                
            # Append A to self.activations.
            self.activations.append(A)
            
        # Return A.
        return A
    
    # Predict what class each element in X is.    
    def predict(self, X):
        X = np.array(X)
        
        # Predict the probability for X.
        results = self.predict_proba(X)
        
        # Return the maximum values for the predictions for X.
        return np.argmax(results, axis = 1)
        
    # Determine the loss and accuracy of the model.
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # Calculate the accuracy.
        accuracy = np.mean(self.predict(X) == y)
        
        # Calculate the probability of elements in X.
        prob = self.predict_proba(X)
        
        # Calculate the loss with the calculated probability.
        loss = 0
        for i in range(len(y)):
            loss -= np.log(prob[i, y[i]])
            
        # Return the loss and accuracy
        return (loss, accuracy)
        
    # Softmax function.
    def softmax(self, x):
        return np.exp(x)/ np.sum(np.exp(x), axis = 1, keepdims = True)