from sklearn.datasets import load_iris #Load iris dataseet used for pattern recognition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


iris_dataset = load_iris(as_frame=True)
iris = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names) #Convert to dataframe
iris['Target'] = iris_dataset.target #Add target colunmn

print(iris.head()) #Returns 1st 5 rows
print(iris.tail()) #Returns last 5 rows
print(iris.info()) #Information about the dataset
print(iris.describe()) #Summary statistics
print(iris.keys()) #Returns 1st 5 rows

#Class distribution
print(iris.groupby('Target').size()) #Groups by target and outpts no of datapoints per group
print(iris['Target'].value_counts()) #Works similar to groupby

#Correlation matrix shows the relationship among the features
corr_matrix = iris.corr() #Last row/column shows which is most correlated with the target
print(corr_matrix) 

#From the corelation matrix we can see that the values most related are petal length and petal width
#Data preparation
x = iris[['petal length (cm)', 'petal width (cm)']]
y = iris['Target']

#Train-test-split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3, random_state=1, stratify=y)
#Stratified by y to ensure that the distribution of labels remains similar in both test and train tests

#Modelling
knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(X_train, Y_train)

#prediction
y_pred = knc.predict(X_test)
print(y_pred[:5])

#Probability prediction
#This outputs the probability for the target in array form
y_pred_prob = knc.predict_proba(X_test)
print(y_pred_prob[10:12]) 
#The probability of the 5 Nearest Neighbors of the 12th flower: 1 of them is 1 and the rest is 2

#Model Evaluation
print((y_pred==Y_test.values).sum())
print(Y_test.size)
print(knc.score(X_test, Y_test))#Accuracy

#Confusion matrix
#Summary of the counts of correct and incorrest predictions broken down by each class.
print(confusion_matrix(Y_test, y_pred))

#Using K-fold cross validation
#Each and every sample is used in the test and train set
knc_cv = KNeighborsClassifier(n_neighbors = 3)#Create a new model
cv_scores = cross_val_score(knc_cv, x,y,cv = 5)
print(cv_scores)
print(cv_scores.mean())

#Grid Search: Using Gridserch to find the optimal K
knc_ = KNeighborsClassifier() 
param_grid = {'n_neighbors': np.arange(2,10)} #Dict for all n_neighbors values\
knc_GS = GridSearchCV(knc_, param_grid, cv=5)
knc_GS.fit(x,y)
print(knc_GS.best_params_)#To check best performing k
print(knc_GS.best_score_)#Accuracy of top n_neighbor

#Building the final model with best performing n_neighbor
knc_final = KNeighborsClassifier(n_neighbors=knc_GS.best_params_['n_neighbors'])
knc_final.fit(x.values,y.values)
y_pred = knc_final.predict(x.values)
print(knc_final.score(x.values,y.values))

#Predicting using new data
new_data = np.array([3.76, 1.20])
new_data = new_data.reshape(1, -1)
print(knc_final.predict(new_data))
