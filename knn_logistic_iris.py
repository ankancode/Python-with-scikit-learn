import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

logreg = LogisticRegression()
type(iris)
print (iris.data)
print (iris.target)
print (type(iris.data))
print (type(iris.target))
print (iris.data.shape)
print (iris.target.shape)

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=4)

k_range = list(range(1,31))
weight_option = ['uniform', 'distance']
scores=[]
knn = KNeighborsClassifier()
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    knn.fit(x_train, y_train)
#    y_pred_knn = knn.predict(x_test)
#    scores.append(metrics.accuracy_score(y_test, y_pred_knn))

param_grid = dict(n_neighbors=k_range, weights=weight_option)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(x,y)
print(grid.grid_scores_)  
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]  

'''plt.plot(k_range, grid_mean_scores) 
plt.xlabel('Value of for KNN')
plt.ylabel('Testing Accuracy')  ''' 

print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
#knn.fit(x_train,y_train)
k_scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    k_scores.append(score.mean())
print (k_scores)    

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

logreg.fit(x_train,y_train)
#print(knn.predict([1, 2, 3, 4]))
x_new = [[3, 5, 4, 2], [5, 4, 3, 2]]

y_pred_logreg = logreg.predict(x_test)


print (metrics.accuracy_score(y_test, y_pred_logreg))

knn = KNeighborsClassifier(n_neighbors=20)
print (cross_val_score(knn, x, y, cv=10, scoring='accuracy').mean())
print (cross_val_score(logreg, x, y, cv=10, scoring='accuracy').mean())
knn.fit(x,y)
print (knn.predict(x_new))










