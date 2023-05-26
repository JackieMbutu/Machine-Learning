#To have multiple training and test sets

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
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

kfold = KFold(n_splits=3, shuffle=True)#Shuffle is f0r whether to randomize the order of the data
splits = list(kfold.split(x))

#Iterate through the splits to find the scores of the models built
scores = []
for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(x_train, y_train)
    scores.append(model.score(x_test, y_test))

#Find the split with the highest score
max_score = max(scores)
max_score_index = scores.index(max_score)

#Build the final model
train_indices, test_indices = splits[max_score_index]
x_train = x[train_indices]
x_test = x[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

final_model = LogisticRegression()
final_model.fit(x_train, y_train)
print(model.score(x_test, y_test))

#Evaluate model using test set
y_pred =final_model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred,pos_label='1')) #%of positive results that are relevant
print(recall_score(y_test, y_pred,pos_label='1')) #% of positive cases correctly classified
print(f1_score(y_test, y_pred,pos_label='1')) #Average for precision and recall
print(model.predict_proba(x_test)) #outputs array of 0 class and 1 class.

