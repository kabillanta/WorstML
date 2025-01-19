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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

w, b = train_logistic_regression(X_train_np, y_train_np, lr=0.1, epochs=1000)


X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

y_pred = predict(X_test_np, w, b)

accuracy = accuracy_score(y_test_np, y_pred)
print(f"Accuracy: {accuracy}")

y_pred_proba = sigmoid(np.dot(X_test_np, w) + b)
log_loss_value = log_loss(y_test_np, y_pred_proba)
print(f"Log Loss: {log_loss_value}")



# Above is the mathematical intituion for better understanding lol :((
# Here is the same implemented using sklearn :))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)
skpred = model.predict(X_test)
skaccuracy = accuracy_score(y_test, skpred)
print(skaccuracy)