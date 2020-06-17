import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')
    #print(dataset)

    X = dataset.drop(['country','rank','score'],axis=1)
    y = dataset[['score']]

    regresor = RandomForestRegressor()

    #define the grid of parameters that the regressor will use to define
    #the best combination of parameters

    '''
         n_estimators: defines the number of trees that will make up
             my random forest RandomForestRegressor
         criterion: measure of caldiad of the splits that the tree makes, it tells me how
             good it was, or so bad it was
         max_depth: how deep the tree will be
     '''
    parametros = {
        'n_estimators': range(4,16),
        'criterion':['mse','mae'],
        'max_depth': range(2,11)
    }

    rand_est = RandomizedSearchCV(regresor,parametros,n_iter=10, cv=3,scoring='neg_mean_absolute_error').fit(X,y)
    print('-'*64)    
    print(rand_est.best_estimator_) 
    print('-'*64)    
    print(rand_est.best_params_)
    print('-'*64)    
    print(rand_est.predict(X.loc[[0]]))