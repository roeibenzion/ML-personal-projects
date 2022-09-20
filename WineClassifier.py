import numpy as np
import pandas as pd
from sklearn.cluster._kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("winequality-red.csv")

#Score col
df['score'] = 0

#Normalize data
scaler = StandardScaler()
scaler.fit(X = df.drop(columns=['score']))
#New df with normalied features
scaled_features = scaler.transform(X=df.drop(columns=['score']))
scaled_features_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])

#Split data to train and test
X = scaled_features_df
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

'''
Run the kmeans algorithm on train.
We get 10 centeroids that will represent quality.
'''
kmeans = KMeans(n_clusters=10, init="k_means++", max_iter=300, algorithm="lloyd")
y = kMeans.fit(X=X_train, y=y_train)
y_test = kmeans.predict(X_test)
