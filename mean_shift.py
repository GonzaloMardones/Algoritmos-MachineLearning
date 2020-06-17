#serves a moderate amount of data
import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(5))

    X = dataset.drop('competitorname',axis=1) #we eliminate the first column, because this cannot be trained and it is in column 1

    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_)) #he gives us 3 labels

    print('-'*64)
    print(meanshift.cluster_centers_) #we have centers

    dataset['meanshift'] = meanshift.labels_
    print('-'*64)
    print(dataset)
