#When we know how many groups we need
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt


from sklearn.cluster import MiniBatchKMeans #we use this type because it uses fewer resources

if __name__ == "__main__":
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(10))
 
    X = dataset.drop(['competitorname'],axis=1)
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X) #batch_size, it will go through groups to review, from 8 data we will be formed the algorithm and the result
    print('Total of centers: ', len(kmeans.cluster_centers_))
    print('-'*64)
    print(kmeans.predict(X))

    
    dataset['group'] = kmeans.predict(X)  #the new column must have the same amount of values as the original bbdd
    #sns.pairplot(dataset, hue='group')
    #sns.pairplot(dataset[['sugarpercent','pricepercent','winpercent','group']], hue = 'group')
    #print(dataset)
    #plt.savefig('save_as_a_png.png')
    # Now I send the data to an excel file :)
    writer = pd.ExcelWriter('./data/candy_user.xlsx', engine='xlsxwriter')
    dataset.to_excel(writer, sheet_name='user')
    writer.save()