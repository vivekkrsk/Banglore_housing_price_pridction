import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle
data= pd.read_pickle('Cleaned_data.pkl')

X=data.drop(columns=['price'])
y=data['price']

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)

# Apply Linear Regression

column_trans = make_column_transformer((OneHotEncoder(sparse=False),['location']), remainder='passthrough')

scaler = StandardScaler() 
lr=LinearRegression()
pipe=make_pipeline(column_trans,scaler,lr)
pipe.fit(X_train, y_train)

y_pred_lr=pipe.predict(X_test)
r2_score(y_test, y_pred_lr)

#  Applying Lasso

lasso=Lasso()
pipe = make_pipeline(column_trans,scaler, lasso)
pipe.fit(X_train, y_train)
y_pred_lasso=pipe.predict(X_test)
r2_score(y_test, y_pred_lasso)

# Apply Ridge
ridge=Ridge()
pipe = make_pipeline(column_trans,scaler, ridge)
pipe.fit(X_train, y_train)
y_pred_ridge=pipe.predict(X_test)
r2_score(y_test, y_pred_ridge)

# compare all three model

print("NO Regularization: ", r2_score(y_test, y_pred_lr))
print("Lasso: ", r2_score(y_test, y_pred_lasso))
print("Ridge: ", r2_score(y_test, y_pred_ridge))

pickle.dump(pipe, open('RidgeModel.pkl','wb'))
