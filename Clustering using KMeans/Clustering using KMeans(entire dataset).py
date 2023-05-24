#Clustering using KMeans for an entire dataset rather than select groups
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

wine_dataset = load_wine()
wine = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)

print(wine.head()) #Returns 1st 5 rows
print(wine.tail()) #Returns last 5 rows
print(wine.info()) #Information about the dataset
print(wine.describe()) #Summary statistics
print(wine.keys()) #Returns 1st 5 rows

X = wine #Use entire dataset

#Preprocessing: To standardise the data
#Scaling x such that it is centered at 0 and it has a unit standard deviation
scale = StandardScaler() #Compute mean and std to be used later for scaling
scale.fit(X)
print(scale.mean_)
print(scale.scale_)#Standard deviation

#Fit to the training data and transform it
x = scale.transform(X)

#Check if data is centered at 0 and has an std of 1
print(x.mean(axis = 0))
print(x.std(axis = 0))

#Finding optimum k using elbow method
elbow = []
for i in np.arange(1,11):
    km = KMeans(n_clusters = i, n_init = 10)
    km.fit(x)
    elbow.append(km.inertia_)
    
print(elbow)
#From the data, k=3 seems the most favorable
#Optimal k is where inertia no longer decreases as rapidly

#Modelling: 
kmeans = KMeans(n_clusters=3, n_init= 10)
kmeans.fit(x)
y_pred = kmeans.predict(x)
print(y_pred)

#Inspecting the coordinates of all the centroids
print(kmeans.cluster_centers_)
#To predict the input should have as many values as the data