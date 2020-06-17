import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

#we use logistic regression: used to classify
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['target'],axis=1)
    dt_target = dt_heart['target']

    #For PCA we need to normalize our data with some function
    dt_features = StandardScaler().fit_transform(dt_features)

    X_train,X_test, y_train, y_test = train_test_split(dt_features,dt_target,test_size=0.3,random_state=42)
    #train_test_split part of a set of tests (30% of tests), random_state, says that by giving it a value, the model will always start from the same point
    
    kpca = KernelPCA(n_components=4, kernel='poly')#n_components (optional),tells us to look for the 4 variables that provide the most amount of info
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    logistic  = LogisticRegression(solver='lbfgs')
    logistic.fit(dt_train,y_train)
    print("SCORE KPCA: ",logistic.score(dt_test,y_test))


    print(X_train.shape) #Table shape
    print(y_train.shape) #target data, (0-1): there is presence of heart disease or there is no
    #n_components = min(n_samples, n_features)

    pca = PCA(n_components=3)
    pca.fit(X_train)

    #comparison with IPCA
    ipca = IncrementalPCA(n_components=3, batch_size=10) #send blocks where you will train and combine until the end to generate the total result

    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show() #values between 0-2, how much% provide more information to the model
    
    #regression to normalize PCA
    logistic = LogisticRegression(solver='lbfgs')

    #PCA
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train,y_train)
    
    print("SCORE PCA:",logistic.score(dt_test, y_test))

    #IPCA
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train,y_train)
    print("SCORE IPCA:",logistic.score(dt_test, y_test))

