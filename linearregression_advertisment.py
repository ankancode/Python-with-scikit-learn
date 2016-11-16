import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('C:/Users/Ankan/Downloads/Advertising.csv', index_col = 0)
print(data.head())
print(data.tail())
print(data.shape)

sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=4, aspect=0.7, kind='reg')

feature_cols = ['TV','Radio', 'Newspaper']
x = data[feature_cols]
print(x.head())
print(type(x))
print(x.shape)

y = data['Sales']
print(y.head())
print(type(y))
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

linreg = LinearRegression()
scores = cross_val_score(linreg, x, y, cv=10, scoring='mean_squared_error')
mse_scores = -scores
rmse_scores = np.sqrt(mse_scores)

print(rmse_scores)
print(rmse_scores.mean())

linreg.fit(x_train, y_train)

print (linreg.intercept_)
print (linreg.coef_)

zip(feature_cols, linreg.coef_)

y_pred = linreg.predict(x_test)
print (np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


