import numpy as np
import pandas as pd
from sklearn.cluster._kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("winequality-red.csv")
df.replace({'white': 1, 'red': 0}, inplace=True)

#Score col for classification
df['score'] = 0

#Get rid of missing data, fill with mean
for x in df.columns:
    if(df[x].isnull().sum() > 0):
        df[x].fillna(df[x].mean(), inplace=True)
'''
df.hist(bins=20, figsize=(10, 10))
plt.show()

plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()
'''
#get rid of corralated data
rem_features = []
cor_matrix = df.corr().abs()
#inspect the upper triangle as ths matrix is symmetric
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
to_drop.append('quality')
df = df.drop(columns=to_drop, axis=1)

#train test split
X = df.drop(columns=['score'], axis=1)
y = df['score']
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, test_size=0.2)

#norm data
norm = StandardScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)
#run kmeans
kmeans = KMeans(n_clusters=4, init="k-means++", max_iter=100)
kmeans.fit(xtrain, ytrain)

y = kmeans.fit_predict(X)
unique_labels = np.unique(y)
X['score'] = y
plt.scatter(X['alcohol'], y)
plt.xlabel("Alcohol level")
plt.ylabel("Quality")
plt.show()