#Multilayer perception using MNIST database of handwritten digits dataset.
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split



#To set the no of parameters to limit the no of target values:
    #x,y = load_digits(n_class =2, return_X_y = True)
x,y = load_digits(return_X_y=True)

#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=23)

mlp = MLPClassifier(random_state=23)
mlp.fit(x_train, y_train)
print(mlp.score(x_test, y_test))

#Using the MNIST_84: A more granular version
#from sklearn.datasets import fetch_openml
#x,y = fetch_openml('mnist_84', version = 1, return_X_y=True)

#Finding coefficients for the hidden layer and the output layer
print(mlp.coefs_)
print(len(mlp.coefs_))
#[0] for hidden and [1] for output