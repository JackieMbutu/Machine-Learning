from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# load dataset
titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
df = titanic['data']
df['survived'] = titanic['target']
df['male'] = df['sex'] == 'male'
df = df.replace(np.nan,0) #Replaces all the blank values with 0

print(df.head()) #Returns 1st 5 rows
print(df.tail()) #Returns last 5 rows
print(df.info()) #Information about the dataset
print(df.describe()) #Summary statistics
print(df.keys()) #Returns 1st 5 rows

#Preparation of data
x = df[['pclass', 'male', 'age', 'sibsp', 'fare','parch']].values
y = df['survived'].values

#Creating model
model = LogisticRegression()

#Train-test-split
#Use random state to get the same split every time
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=26)

model.fit(x_train, y_train)

#Evaluate model using test set
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred,pos_label='1')) #%of positive results that are relevant
print(recall_score(y_test, y_pred,pos_label='1')) #% of positive cases correctly classified
print(f1_score(y_test, y_pred,pos_label='1')) #Average for precision and recall
print(model.predict_proba(x_test)) #outputs array of 0 class and 1 class.

