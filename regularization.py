import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#herramientas add
from sklearn.model_selection import train_test_split #para generar datos de entrenamiento y test
from sklearn.metrics import mean_squared_error #error medio cuadrado

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    X = dataset[['gdp','family','lifexp','freedom','corruption','generosity','dystopia']]#definimos los features
    y = dataset[['score']]
    '''print(X.shape)
    print(y.shape)'''
    X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size= 0.25)#25% para datos de test

    #revisaremos nuestros regresores
    modelLinear = LinearRegression().fit(X_train,y_train)
    y_predict_linear = modelLinear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train,y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train,y_train)
    y_predict_Ridge = modelRidge.predict(X_test)

    #perdida para cada modelo entrenado
    linear_loss = mean_squared_error(y_test,y_predict_linear)
    print("Linear Loss:",linear_loss)
    
    lasso_loss = mean_squared_error(y_test,y_predict_lasso)
    print("Lasso Loss: ",lasso_loss)
    
    ridge_loss = mean_squared_error(y_test,y_predict_Ridge)
    print("Ridge Loss: ",ridge_loss)

    print("-"*32)
    print("Coef Lasso")
    print(modelLasso.coef_)

    print("-"*32)
    print("Coef Ridge")
    print(modelRidge.coef_)