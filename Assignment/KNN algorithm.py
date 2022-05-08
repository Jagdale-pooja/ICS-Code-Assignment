import numpy as np
import pandas as pd

#read_dataset
data=pd.read_csv("KNN_dataset.csv")
x=data.iloc[:,:-1].values
print(x)
y=data.iloc[:,2].values
print(y)

#import KneighborsClassfier and create object of it
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(x,y)


#predict class for the points(6,6)
x_test=np.array([6,6])
y_pred=classifier.predict([x_test])
print("general KNN",y_pred)

classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(x,y)
#predict class for the points(6,6)
x_test=np.array([6,6])
y_pred=classifier.predict([x_test])
print("distance weighted KNN",y_pred)
