import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
import joblib

data = pd.read_csv('G:/JS/ITI/MLEM/final/data/flights.csv')
data2 = data.dropna()
x = data2[['month', 'day','sched_dep_time', 'sched_arr_time', 'flight', 'distance','dep_time','air_time','dep_delay']]
y = data2['arr_delay']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.3)
model = KNeighborsRegressor()
model.fit(x_train,y_train)

joblib.dump(model, "rf_model.sav")