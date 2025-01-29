import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
data = load_digits()
df = pd.DataFrame(data.data)
df['target'] = data.target
X = df.drop(['target'],axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators = 40)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

accuracyscore = accuracy_score(y_test, y_pred )
print(accuracyscore*100)
