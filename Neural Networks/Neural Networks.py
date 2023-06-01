from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#Create an artificial dataset
x,y = make_classification(n_features = 2, n_redundant = 0, n_informative = 2, random_state = 3)

#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 3)

#MLP-Multilayer Perception
#The no of iterations determine the convergence of a network. More data less iterations
#Alpha is the step size. How much neural netweork changes coefficients at each iteration.
#Decreasing alpha requires an increase in max iterations.
#Solver is the algorithm used to find the optimum solution.
mlp = MLPClassifier(max_iter= 1000, hidden_layer_sizes = (100,50), alpha=0.0001, solver= 'adam', random_state=3)
mlp.fit(x_train, y_train)
print(mlp.score(x_test, y_test))