# -*- coding: utf-8 -*-
"""
#保存模型
from sklearn.externals import joblib
joblib.dump(lgb,'lgb.pkl')

train0 = train[train['acc_now_delinq'] == 0]
train1 = train[train['acc_now_delinq'] == 1]
#样本个数:709903;正样本占99.54%;负样本占0.46%
"""
### 导入模块
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation,metrics

#  忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore')  

# 01 读取数据
start_time=time.time()
df = pd.read_csv('data/train.csv')
train = df.drop(['grade','verification_status_joint','addr_state','desc','emp_title','earliest_cr_line','issue_d','member_id', \
                        'purpose','title','zip_code'], axis = 1)

# 02 缺失值处理（利用均值填补）
print("train 缺失值处理\n")
train = train.fillna(0)

# 03 字符串的替换--映射
print("map替换\n")
#删除grade因为sub_grade与之重复
train['term']=train['term'].map({' 36 months':1,' 60 months':0.8}).astype(float)
#train['grade'] = train['grade'].map( {'D':0.4, 'A':1, 'E':0.2, 'B':0.8, 'C':0.6, 'F':0.1, 'G':0.05} ).astype(float)
train['sub_grade']=train['sub_grade'].map({'A1':1,'A2':1.2,'A3':1.3,'A4':1.4,'A5':1.5,\
                                           'B1':2,'B2':2.2,'B3':2.3,'B4':2.4,'B5':2.5,\
                                           'C1':3,'C2':3.2,'C3':3.3,'C4':3.4,'C5':3.5,\
                                           'D1':4,'D2':4.2,'D3':4.3,'D4':4.4,'D5':4.5,\
                                           'E1':5,'E2':5.2,'E3':5.3,'E4':5.4,'E5':5.5,\
                                           'F1':6,'F2':6.2,'F3':6.3,'F4':6.4,'F5':6.5,\
                                           'G1':7,'G2':7.2,'G3':7.3,'G4':7.4,'G5':7.5,'0':10}).astype(float)
train['application_type']=train['application_type'].map({'JOINT':1.0,'INDIVIDUAL':0.5}).astype(float)
train['initial_list_status']=train['initial_list_status'].map({'f':0.5,'w':1}).astype(float)
train['pymnt_plan']=train['pymnt_plan'].map({'y':1,'n':0}).astype(int)
train['loan_status']=train['loan_status'].map({'Charged Off':1,'Fully Paid':1,'Current':0.6,'In Grace Period':0.8,\
                                                         'Late (31-120 days)':0.3,'Late (16-30 days)':0.5,'Default':0.1, \
                                                         'Does not meet the credit policy. Status:Fully Paid':0, \
                                                         'Does not meet the credit policy. Status:Charged Off':0,'Issued':0}).astype(float)
train['verification_status']=train['verification_status'].map({'Verified':1.0,'Source Verified':0.5,'Not Verified':0.0}).astype(float)
train['home_ownership']=train['home_ownership'].map({'NONE':0,'ANY':0.2,'OTHER':0.1,'OWN':1,'RENT':0.4,'MORTGAGE':0.8}).astype(float)
train['emp_length']=train['emp_length'].map({'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5, \
                                                                  '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'n/a':0}).astype(float)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

print("首次剔除一些特征\n")
x_val = train.drop(['acc_now_delinq'], axis = 1)
y_val = train['acc_now_delinq']

'''
# 建立逻辑回归分类器
model = LogisticRegression()
# 建立递归特征消除筛选器
rfe = RFE(model, 30) #通过递归选择特征，选择30个特征
rfe = rfe.fit(x_val, y_val)
# 打印筛选结果
print("rfe.support_:\n",rfe.support_)
print("rfe.ranking_:\n",rfe.ranking_) #ranking 为 1代表被选中，其他则未被代表未被选中
col_filter = x_val.columns[rfe.support_] #通过布尔值筛选首次降维后的变量
print(col_filter) # 查看通过递归特征消除法筛选的变量
'''

#新舍弃21个特征
x_val_new = x_val.drop(['funded_amnt','annual_inc','verification_status','pymnt_plan', \
                        'pub_rec','pub_rec','total_rec_prncp','total_rec_int','recoveries',\
                        'collection_recovery_fee', 'collections_12_mths_ex_med',\
                        'annual_inc_joint', 'dti_joint','tot_coll_amt', 'tot_cur_bal' ,\
                         'open_il_12m','total_bal_il','max_bal_bc', 'all_util', \
                         'total_rev_hi_lim', 'inq_fi', 'total_cu_tl'],axis = 1)

# 正负样本的不平衡  / 导入SMOTE算法模块   / 处理过采样的方法 增加反例的比例
print('通过SMOTE方法平衡正负样本后\n')
X = x_val_new
y = y_val
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)    
X, y = sm.fit_sample(X, y)
n_sample = y.shape[0]
n_pos_sample = y[y == 0].shape[0]#正样本
n_neg_sample = y[y == 1].shape[0]#负样本
print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))
print('特征维数：',X.shape[1])
#通过SMOTE方法平衡正负样本后
#样本个数：1413220; 正样本占50.00%; 负样本占50.00%
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression() # 构建逻辑回归分类器
clf.fit(X, y)
predicted1 = clf.predict(X) # 通过分类器产生预测结果

#基本精确度
from sklearn.metrics import accuracy_score
print("Test set accuracy score: {:.5f}".format(accuracy_score(predicted1, y,)))

from sklearn.metrics import confusion_matrix
m = confusion_matrix(y, predicted1)
print("混淆矩阵:\n",m)

from sklearn.metrics import roc_auc_score
roc_auc1 = roc_auc_score(y, predicted1)
print("Area under the ROC curve : %f" % roc_auc1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) # random_state = 0 每次切分的数据都一样
# 构建参数组合
param_grid = {'C': [0.01,0.1, 1, 10, 100, 1000,],
                            'penalty': [ 'l1', 'l2']}
grid_search = GridSearchCV(LogisticRegression(),  param_grid, cv=10) # 确定模型LogisticRegression，和参数组合param_grid ，cv指定5折
grid_search.fit(X_train, y_train) # 使用训练集学习算法
from sklearn.metrics import roc_auc_score
roc_auc1 = roc_auc_score(y, predicted1)
print("Area under the ROC curve : %f" % roc_auc1)

#模型性能评估
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))

end_time=time.time()
print("用时{%d}秒".format(end_time-start_time))