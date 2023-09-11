#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:45:21 2023

@author: colinbehr
"""
#import os
#os.chdir('/Users/colinbehr/Downloads')

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier


# Load consumer data
df = pd.read_csv("all_data.csv")

#feature engineering (dispoable income & individual season)
df['Disposable Income'] = df['Income'] - df['Cost of Living']
#df['Disposable Income Significance'] = df['Disposable Income'] * df['Economy']

df['Fall'] = (df['Season'] == 'Fall').astype(int)
df['Winter'] = (df['Season'] == 'Winter').astype(int)
df['Spring'] = (df['Season'] == 'Spring').astype(int)
df['Summer'] = (df['Season'] == 'Summer').astype(int)
df = df.drop('Season', axis=1)


# impute
imp = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
df = df.drop('data_source', axis=1)
# non-numeric -> numeric
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == object:
        df[column] = le.fit_transform(df[column])
# impute
impute_strategies = ['mean', 'median', 'most_frequent']
best_score = -1
best_imputer = None
for strategy in impute_strategies:
    imp = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

    # numerical columns
    numerical_cols = df_imputed.select_dtypes(include=['int64', 'float64']).columns

    # impute numerical columns with mean or median
    imp_num = SimpleImputer(strategy='median')
    df_imputed[numerical_cols] = imp_num.fit_transform(df_imputed[numerical_cols])

    # non-numeric -> numeric

#    df_imputed = pd.get_dummies(df_imputed, columns=[column for column in df_imputed.columns if df_imputed[column].dtype == object])

    le = LabelEncoder()
    for column in df_imputed.columns:
        if df_imputed[column].dtype == object:
            df_imputed[column] = le.fit_transform(df_imputed[column])

    # split data
    train_df = df_imputed[0:7000]
    val_df = df_imputed[7000:8500]
    test_df = df_imputed[8500:]

    # X and y
    X = train_df.drop("Click", axis=1)
    y = train_df['Click']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
    # XGB
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'n_estimators': [200, 500],
        'max_depth': [1, 3, 5],
        'subsample': [0.5, 1],
        'reg_alpha': [0.1, 1],
        'gamma': [0, 0.1, 0.2],
        'colsample_bytree': [0.5, 0.8, 1]
    }


    xgb_model = XGBClassifier(
        objective='binary:logistic',
        nthread=4,
        seed=27
    )

    xgb_grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=6,
        scoring='roc_auc',
        n_jobs=10,
        verbose=True
    )

    xgb_grid.fit(X_train, y_train)

    # Check if the current imputation strategy gives a better score
    if xgb_grid.best_score_ > best_score:
        best_score = xgb_grid.best_score_
        best_imputer = imp
        
print("Best Score: ", xgb_grid.best_score_)
importances = xgb_grid.best_estimator_.feature_importances_

# Create a dataframe with feature importances
importances_df = pd.DataFrame({'Features': X_train.columns, 'Importance': importances})

# Sort the dataframe in descending order of feature importance
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Print the top 10 most important features
print("Most significant features:\n", importances_df.head(10))

df = pd.DataFrame({'innum':test_df['innum'], 'probabilities':xgb_grid.predict_proba(test_df.drop('Click', axis=1))[:, 1]})
df.to_csv('testpredict.csv')

#df = pd.DataFrame({'innum':val_df['innum'], 'probabilities':xgb_grid.predict_proba(val_df.drop('Click', axis=1))[:, 1]})
#df.to_csv('val.csv')
