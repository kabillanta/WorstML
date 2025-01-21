import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = np.array([[0.5, 1.5], [1,0.5], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  
y = np.array([0, 0, 0, 1, 1, 1])           

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(y_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i) 
             
    cost = cost / m
    return cost

def initialize_weights(n_features):
    w = np.zeros(n_features)  
    b = 0                     
    return w, b

def compute_gradients(X, y, w, b):
    m = X.shape[0] 
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)    
    dw = 1/m * np.dot(X.T, (y_pred - y))
    db = 1/m * np.sum(y_pred - y)
    return dw, db

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    w, b = initialize_weights(X.shape[1])
    costs = []
    for epoch in range(epochs):
        dw, db = compute_gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db
        if epoch % 100 == 0:
            cost = compute_cost_logistic(X, y, w, b)
            costs.append(cost)
            print(f"Epoch {epoch}: Cost = {cost}")
    return w, b

def predict(X, w, b):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)

w, b = train_logistic_regression(X_train, y_train, lr=0.1, epochs=1000)

y_pred = predict(X_test, w, b)

print(y_pred)

# Above is the mathematical intituion for better understanding lol :((
