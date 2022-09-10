import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
import numpy as np
import math

df = pd.read_csv('/Users/esra/Library/Containers/com.microsoft.Excel/Data/Downloads/bluebook-for-bulldozers/Machine_Appendix.csv')
print(df.head()) #Q1-)we use this to see our first 5 rows.

print(df.info()) #Q2-)we use this to see which cells are null.

df.hist(bins=50,figsize=(20,15),color= '#ee104e',rwidth=2)
print(plt.show()) #we use these coding line for image processing

print(df.isnull().sum()) #we controled our data for if it contains null values