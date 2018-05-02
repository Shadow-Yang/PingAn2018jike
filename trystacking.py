# -*- coding: utf-8 -*-
'''
Created on 2018年4月24日
try stacking GBDT+logistic loss+XGB
@author: Mr.Yang
'''
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from SMOTElogisticloss import param_grid
import numpy as np
from statsmodels.tsa.kalmanf.kalmanfilter import penalty

def lr_model(x_train, x_test, y_train, y_test):
    ''' 返回训练好的逻辑分类模型及分数 '''
    lr = LogisticRegression(C=1.0)
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    return lr, metrics.accuracy_score(y_test, y_pred)

def gbdt_modle(x_train, x_test, y_train, y_test):
    ''' 返回训练好的GBDT模型及分数 '''
    param_grid = {
        'n_estimators':range(80, 120, 10),
        'max_features':np.arange(.6, .9,.1).tolist(),
        'max_depth':range(3, 9)+[None]
        }
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid,cv=10,
                        scoring='accuracy',n_jibs=-1)
    grid.fit(x_train, y_train)
    gbdt = GradientBoostingClassifier(**grid.best_params_)
    gbdt.fit(x_train, y_train)
    y_pred = gbdt.predict(x_test)
    return gbdt, metrics.accuracy_score(y_test, y_pred)

def stack_models(x_train, x_test, y_train, y_test):
    ''' 返回融合后的模型及分数 '''
    param_grid={
        'C':{.01, .1, 1, 10}
        }
    grid = GridSearchCV(LogisticRegression(),param_grid, cv=10,
                        scoring='accuracy', n_jobs=-1)
    grid.fit(x_train,y_train)
    #L1正则化，更稀疏
    stk = LogisticRegression(penalty='l1', tol=1e-6,
                             **grid.best_params_)
    stk.fit(x_train, y_train)
    y_pred = stk.predict(x_test)
    return stk, metrics.accuracy_score(y_test, y_pred)

