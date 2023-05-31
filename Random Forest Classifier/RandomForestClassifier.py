#Random Forest: Multiple decision trees, less prone to overfitting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
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
x = df[['pclass', 'male', 'age', 'sibsp', 'parch', 'fare']].values
y = df['survived'].values

#Creating model
model = RandomForestClassifier()

#Train-test-split
#Use random state to get the same split every time
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=26)

model.fit(x_train, y_train)
print(model.score(x_test, y_test))

#Tuning using n_estimators and max_features. 
#N_estimators are the no of trees.
#Max_features are the no of trees to consider at each split.
#Finding the best fit using GridSearch

param_grid = {'n_estimators': [10,25,50,75,100]}
gs = GridSearchCV(model, param_grid, cv=5)
gs.fit(x,y)
print(gs.best_params_) #Find out which parameters work best

