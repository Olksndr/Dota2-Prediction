# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 07:40:32 2017

@author: Olksndr
"""

def df_creation():
    import pandas as pd
    df = pd.read_csv("D:/features.csv", index_col = "match_id")


    X = df.drop(['duration','radiant_win', 'tower_status_radiant',\
             'tower_status_dire', 'barracks_status_radiant', \
             'barracks_status_dire'], axis=1)
 
    y = df[['duration','radiant_win', 'tower_status_radiant',\
             'tower_status_dire', 'barracks_status_radiant', \
             'barracks_status_dire']]
    return X, y
#reading data

class data_to_analyze:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
def train_test_spliter(data_to_analyze):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5, shuffle = True)
    for train, test in kf.split(data_to_analyze.X):
        X_train, X_test = data_to_analyze.X.iloc[train], data_to_analyze.X.iloc[test]    
        y_train, y_test = data_to_analyze.y.iloc[train], data_to_analyze.y.iloc[test]
    return X_train, X_test, y_train, y_test   

def modelBuilt_and_score():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    
    
    dataSet = data_to_analyze(df_creation()[0], df_creation()[1])
    dataSet.X = dataSet.X.fillna(value = 0)
    
    X_train, X_test, y_train, y_test = train_test_spliter(dataSet)
    
    n_estimators = [10, 20, 30]
    scores = []
    for i in n_estimators:
        GBC = GradientBoostingClassifier(n_estimators = i)
        GBC.fit(X_train , y_train['radiant_win'])
        pred = GBC.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(y_test['radiant_win'], pred))
        
    print ('Gradient Boosting Best Accuracy: ', max(scores),';',' estimators = ',\
           n_estimators[scores.index(max(scores))],';')                    
    return scores
    
def time_measuring(function_to_measure_time):
    import time
    import datetime
    start_time = datetime.datetime.now()
    
    function_to_measure_time
    
    time.sleep(3)
    print ('Time elapsed:',\
                        datetime.datetime.now() - start_time)
    return function_to_measure_time

def train_test_spliter_1(data_to_analyze):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5, shuffle = True)
    for train, test in kf.split(data_to_analyze.X):
        X_train, X_test = data_to_analyze.X[train], data_to_analyze.X[test]    
        y_train, y_test = data_to_analyze.y.iloc[train], data_to_analyze.y.iloc[test]
    return X_train, X_test, y_train, y_test   


def logRegModelBuilt_and_measuring():
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    dataSet = data_to_analyze(df_creation()[0], df_creation()[1])
    
    scaler = StandardScaler() 
    
    dataSet.X = dataSet.X.drop(['lobby_type','r1_hero', 'r2_hero', 'r3_hero',\
    'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
        
    scaler = StandardScaler()    
    dataSet.X = scaler.fit_transform(dataSet.X.fillna(value=0))
    
    
    X_train, X_test, y_train, y_test = train_test_spliter_1(dataSet)
    
    scores = []
    C_range = [10.0 ** i for i in range(-5, 6)]
    for i in C_range:
        log_reg = LogisticRegression(C = i)
        log_reg.fit(X_train, y_train['radiant_win'])
        pred = log_reg.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(y_test['radiant_win'], pred))
        
    print ('Logistic Regression best score: ', max(scores),'; '\
           ,'inverse of regularization strength = ',C_range[scores.index(max(scores))])
    return scores

def categ_features_encoding(data_to_analyze):
    import numpy as np
    import pandas as pd
    heroesAmmount = 112
    X_pick = np.zeros((data_to_analyze.X.shape[0], heroesAmmount))
    for i, match_id in enumerate(data_to_analyze.X.index):
        for p in range(5):
            X_pick[i, data_to_analyze.X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data_to_analyze.X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    
    #The thing is, that dataset contains unreleased heroes                                  
    emptyColumnsindex = np.where(~X_pick.any(axis=0))[0]
    unclearedEncodedFeatures = pd.DataFrame(X_pick)
    clearedEncodedFeatures = unclearedEncodedFeatures.drop(emptyColumnsindex, axis = 1)
    encodedFeatures = pd.DataFrame(data = clearedEncodedFeatures, \
                                   index = data_to_analyze.X.index)
    
    return encodedFeatures.fillna(value=0)


def logRegModelBuiltPreprocessed_and_measuring():
    "categorial features processed"
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    dataSet = data_to_analyze(df_creation()[0], df_creation()[1])
    
    scaler = StandardScaler() 
    
    encodedFeatures = categ_features_encoding(dataSet)
    
    dataSet.X = dataSet.X.join(encodedFeatures)

    dataSet.X = dataSet.X.drop(['lobby_type','r1_hero', 'r2_hero', 'r3_hero',\
    'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
    
    scaler = StandardScaler()    
    dataSet.X = scaler.fit_transform(dataSet.X.fillna(value=0))
    
    
    X_train, X_test, y_train, y_test = train_test_spliter_1(dataSet)
    
    scores = []
    C_range = [10.0 ** i for i in range(-5, 6)]
    for i in C_range:
        log_reg = LogisticRegression(C = i)
        log_reg.fit(X_train, y_train['radiant_win'])
        pred = log_reg.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(y_test['radiant_win'], pred))
        
    print ('Processed data Logistic Regression best score: ', max(scores),'; '\
           ,'inverse of regularization strength = ',C_range[scores.index(max(scores))])
    return scores
