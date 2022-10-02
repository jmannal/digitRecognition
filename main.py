import numpy as np
import matplotlib.pyplot as plt
import copy

def loadData():

    trainArray = np.loadtxt("mnist_train.csv", delimiter = ",")
    testArray = np.loadtxt("mnist_test.csv", delimiter = ",")

    Y_train = trainArray.T[0]
    X_train = (np.delete(trainArray, 0, 1)).T

    Y_test = testArray.T[0]
    X_test = (np.delete(testArray, 0, 1)).T

    '''for i in range(0, len(Y_train)):
        if Y_train[i] == 2:
            Y_train[i] = 1
        else:
            Y_train[i] = 0

    for i in range(0, len(Y_test)):
        if Y_test[i] == 2:
            Y_test[i] = 1
        else:
            Y_test[i] = 0'''

    return X_train, Y_train, X_test, Y_test

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    b = 0.0

    return w, b

def propagate(w, b, X, Y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)

    cost = (-1 / m) * np.sum(np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), (1 - (Y.T))))
    
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
                
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    
    A = sigmoid(np.dot(w.T, X) + b)
        
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
        
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = initialize_with_zeros(X_train.shape[0])
    
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = True)
    
    w = params["w"]
    b = params["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs, "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, "w" : w, "b" : b,
         "learning_rate" : learning_rate, "num_iterations": num_iterations}
    
    return d

X_train, Y_train, X_test, Y_test = loadData()

# Standardize data
X_train /= 255
X_test /= 255

model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.01, print_cost=True)