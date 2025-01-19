import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

dataset = {
    'Feature1': [2.7, 1.5, 3.1, 4.5, 3.8, 1.2, 3.6, 1.9],
    'Feature2': [1.3, 2.1, 1.8, 2.6, 3.2, 2.5, 2.7, 1.4],
    'Label': [0, 1, 0, 1, 1, 1, 0, 0]
}

data = pd.DataFrame(dataset)

X = data[['Feature1', 'Feature2']]
y = data['Label']                  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_weights(n_features):
    w = np.zeros(n_features)  
    b = 0                     
    return w, b

def compute_cost(X, y, w, b):
    m = X.shape[0] 
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    
    cost = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost

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
            cost = compute_cost(X, y, w, b)
            costs.append(cost)
            print(f"Epoch {epoch}: Cost = {cost}")
    return w, b

def predict(X, w, b):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)
