import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


df = pd.read_csv('datasets/titanic_train.csv')
df = df.drop(['Cabin','Ticket','Name','PassengerId'], axis="columns")

encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['Embarked'] = encoder.fit_transform(df['Embarked'])


X = df.drop(['Survived'],axis="columns")
y = df[['Survived']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


result = accuracy_score(y_test,y_pred)

print(f'Accuracy Score is {result}')

# Got around 0.7988