import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import math
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from warnings import filterwarnings
filterwarnings('ignore')



#Q1-)we use this to see our first 5 rows.
df = pd.read_csv('/Users/esra/Library/Containers/com.microsoft.Excel/Data/Downloads/YemekhaneRezervasyon.csv')
#print(df.head())
#Q2-)we use this to see which cells are null.
#print(df.info())

#df.hist(bins=50,figsize=(20,15),color= '#ee104e',rwidth=2)
#print(plt.show()) #we use these coding line for image processing

#print(df.isnull().sum()) #we controled our data for if it contains null values

#Q3-) Make your train and test sets. I used the knn way.
X= df.drop(["People_count","Date"],axis=1)
y= df["People_count"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=190)
#knn_model = KNeighborsRegressor(n_neighbors=5)
#model = knn_model.fit(X_train,y_train)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import time
from sklearn.ensemble import RandomForestRegressor

time_start = time.time()

# Training model
regressor = RandomForestRegressor(n_estimators=100, max_features = 0.5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

time_end = time.time() # Estimating running time of the model
print(f'Run time : {time_end - time_start}')

from sklearn import metrics

print('Training score: ', regressor.score(X_train, y_train))
print('Testing score:  ', regressor.score(X_test, y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(
