from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import graphviz
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
model = DecisionTreeClassifier()

#Train-test-split
#Use random state to get the same split every time
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=26)

model.fit(x_train, y_train)

#predict using a test set
print(model.predict([[3, True, 22, 1, 0, 7.25]]))

#Evaluating the model
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred,pos_label='1')) #%of positive results that are relevant
print(recall_score(y_test, y_pred,pos_label='1')) #% of positive cases correctly classified
print(f1_score(y_test, y_pred,pos_label='1')) #Average for precision and recall
print(model.predict_proba(x_test)) #outputs array of 0 class and 1 class for survived and didn't survive. 

#Gini vs Entropy
#dt = DecisionTreeClassifier(criterion = 'entropy')
#Gini is the default
#Comparing the two using KFold
kfold = KFold(n_splits=3, shuffle=True)#Shuffle is f0r whether to randomize the order of the data
splits = list(kfold.split(x))

criterion = ['gini', 'entropy']
#Iterate through the splits to find the scores of the models built from both criterion
#Compare their values.
for i in criterion:
    print("Decision Tree - {}".format(i))
    scores = []
    accuracy= []
    precision = []
    recall = []
    f1Score = []
    for train_index, test_index in kfold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = DecisionTreeClassifier(criterion= i)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        scores.append(model.score(x_test, y_test))
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, pos_label='1'))
        recall.append(recall_score(y_test, y_pred, pos_label='1'))
        f1Score.append(f1_score(y_test, y_pred, pos_label='1'))
    print(np.mean(accuracy))
    print(np.mean(precision))
    print(np.mean(recall))
    print(np.mean(f1Score))

#Pre pruning: Limiting depth, no of leaf nodes and leaves with few data points 
# Example: model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)
#Finding the best parameters using GridSearch
param_grid= {'max_depth': [5,15,25],
              'min_samples_leaf': [1,3],
              'max_leaf_nodes': [10,20,35,50]
    }

gs = GridSearchCV(model, param_grid, scoring='f1_micro', cv = 5)#f1_micro is used for multi-class
gs.fit(x,y)
print(gs.best_params_) #To determine which model was better
print(gs.best_score_) #The score of the better model.

#Visualizing the decision tree
#Outputs a png of the decision tree
feature_names = ['pclass', 'male', 'age', 'sibsp', 'parch', 'fare']
file_ = export_graphviz(model, feature_names=feature_names)
file = graphviz.Source(file_)
file.render(filename = 'Decision Tree', format = 'png', cleanup = True)