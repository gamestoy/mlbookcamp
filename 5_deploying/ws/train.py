#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import pickle

c = 1
n_split = 5
output_file = f'model_C={c}.bin'

print("Loading data...")
df = pd.read_csv('../data/lesson3.csv')

print("Analyzing data...")
df.columns = df.columns.str.lower().str.replace(' ','_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype('int')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_full_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

print(f"Validating model with C={c}...")

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['seniorcitizen','gender','partner','dependents','phoneservice','multiplelines','internetservice','onlinesecurity','onlinebackup','deviceprotection','techsupport','streamingtv','streamingmovies','contract','paperlessbilling','paymentmethod']

def train(df, y, c=1.0):
    dicts = df[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(max_iter=1000, C=c)
    model.fit(X_train, y)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.fit_transform(dicts)
    y_pred = model.predict_proba(X)[:,1]

    return y_pred

def kfold_scores(kfold, df_full_train, c):
    scores = []
    fold = 0
    for train_idxs, val_idxs in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idxs]
        df_val = df_full_train.iloc[val_idxs]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, c)
        y_pred = predict(df_val, dv, model)
        auc = roc_auc_score(y_val, y_pred)
        print(f'AUC in fold {fold}: {auc}')
        fold = fold + 1
        scores.append(auc)
    return scores


kfold = KFold(n_splits=n_split, shuffle=True, random_state=1)
scores = kfold_scores(kfold, df_full_train, c)
print('C = %s -> %.3f +/- %.3f' % (c, np.mean(scores), np.std(scores)))


print("Training the final model...")
y_full_train = df_full_train.churn.values
del df_full_train['churn']

dv, model = train(df_full_train, y_train, c)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)

print(f'final AUC: {auc}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

