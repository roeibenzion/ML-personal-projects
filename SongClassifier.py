import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("Songs.csv", encoding = 'latin-1')

df['target'] = 0
df = df.drop(df.columns[0], axis=1)
df = df.drop(columns = ['title', 'artist', 'top genre', 'year'])

#normalize data
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
#new df for normalized features
scaled_features = scaler.transform(df.drop('target',axis=1))
scaled_features_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])

#split the data set to tain and test
X = scaled_features_df
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

#train 
knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
print(classification_report(y_test, prediction))