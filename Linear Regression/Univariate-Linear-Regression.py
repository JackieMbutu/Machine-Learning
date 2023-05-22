from sklearn.datasets import fetch_california_housing #California housing dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset as a dataframe
housing = fetch_california_housing(as_frame = True) 
print(housing.DESCR) #Description of the dataset

#Convert the data into dataframe
c_housing = pd.DataFrame(housing.data, columns = housing.feature_names)

#Add the target to the created dataframe
c_housing['Target'] = housing.target

print(c_housing.head()) #Returns 1st 5 rows
print(c_housing.tail()) #Returns last 5 rows
print(c_housing.info()) #Information about the dataset
print(c_housing.describe()) #Summary statistics
print(c_housing.keys()) #Returns 1st 5 rows


#Correlation matrix shows the relationship among the features
corr_matrix = c_housing.corr() #Last row/column shows which is most correlated with the target
print(corr_matrix)

#Median Income seems most related to the target
#Building a linear regression model
model = LinearRegression()

#Train-Test split
X = c_housing[['MedInc']]
Y= c_housing['Target']
#Split data 70-30
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

#Check dimensions of split
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Train model using split data
model.fit(X_train, Y_train)
print(model.intercept_.round(2)) #Intercept
print(model.coef_.round(2)) #Slope
y_predicted = model.predict(X_test)

#How good is the prediction? 
#Residuals:The closer to 0 the better.  
#Should symmetrically abd randomly sapced around the horizontal axis.
#If residual plot shows a pattern, linear or nonlinear, model should be improved. 
residuals = Y_test - y_predicted
plt.scatter (X_test, residuals)
plt.hlines(y =0, xmin = X_test.min(), xmax = X_test.max(), linestyle = '-----')
plt.xlim([4,9])
plt.show

#Mean squared error: The smaller the better
print(mean_squared_error(Y_test, y_predicted))
#R-squared: Proportion of total variation explained by model
print(model.score(X_test, Y_test))


