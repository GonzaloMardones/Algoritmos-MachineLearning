import pandas as pd

from sklearn.linear_model import (RANSACRegressor,HuberRegressor) 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad_corrupt.csv')
    print(dataset.head(5))

    X = dataset.drop(['country','score'], axis=1) #axis = 1, we mean the columns, axis = 0, the rows
    y  = dataset[['score']] #target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

   #define pro estimators using a dictionary
    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(), #is a metaestimator
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        print('-'*64)
        print(name)
        print('Error Medio Cuadraro (MSE): ', mean_squared_error(y_test,predictions))
