from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = (w * X[i]) + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    dw = 0
    db = 0
    for i in range(m):
        f_wb = (w * X[i]) + b
        dw += (f_wb - y[i]) * X[i]
        db += (f_wb - y[i])
    dw /= m
    db /= m
    return dw, db

def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    for i in range(num_iterations):
        dw, db = compute_gradient(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        cost = compute_cost(X, y, w, b)
    return w, b

w = 0
b = 0

learning_rate = 0.01
num_iterations = 1000

w, b = gradient_descent(X, y, w, b, learning_rate, num_iterations)

plt.scatter(X, y, label="Training data")
plt.plot(X, w * X + b, color="red", label="Linear regression line")
plt.legend()
plt.show()