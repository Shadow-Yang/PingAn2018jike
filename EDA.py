import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

start_time=time.time()
#读取数据
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('train.head()\n',train.head(3))
print('train.info()\n',train.columns)
#print('train.describe()\n',train.describe())
#print('train.shape\n',train.shape) ( 709903, 64)

'''
#读取数据，查看每个基本特征列的取值个数
for col in train.columns:
    print('The different number of {} is {}'.format(col,train[col].nunique()))

dti_joint                      709495  **
annual_inc_joint               709493  **
verification_status_joint      709493  **
il_util                        694965  **
mths_since_rcnt_il             693234  **
inq_last_12m                   692782  **
open_acc_6m                    692782  **
open_il_6m                     692782  **
open_il_24m                    692782  **
open_il_12m                    692782  **
total_bal_il                   692782  **
open_rv_12m                    692782  **
open_rv_24m                    692782  **
max_bal_bc                     692782  **
all_util                       692782  **
inq_fi                         692782  **
total_cu_tl                    692782  **
desc                           608928  **
mths_since_last_record         600348  **
mths_since_last_major_derog    532479  **
tot_cur_bal                     56135  **
tot_coll_amt                    56135  **
total_rev_hi_lim                56135  **
emp_title                       41182  **
revol_util                        409  **
title                             127  **
collections_12_mths_ex_med        118  **
total_acc                          24
earliest_cr_line                   24
pub_rec                            24

train_drop=train.drop(['dti_joint','annual_inc_joint','verification_status_joint','il_util',\
                        'mths_since_rcnt_il','inq_last_12m','open_acc_6m','open_il_6m', \
                        'open_il_24m','open_il_12m','total_bal_il','open_rv_12m','open_rv_24m',\
                        'max_bal_bc','all_util','inq_fi','total_cu_tl','desc','mths_since_last_record', \
                        'mths_since_last_major_derog','tot_cur_bal','tot_coll_amt', \
                        'total_rev_hi_lim','emp_title','revol_util','title',\
                        'collections_12_mths_ex_med'], axis = 1)

'''
'''
#输出每个列丢失值也即值为NaN的数据和，#降序排列，缺失值超过50的特征列
missingValue = (train.isnull().sum().sort_values(ascending=False) > 50)
missingValue_index = missingValue[missingValue.values].index
train_drop = train.drop(missingValue_index, axis = 1)
#print('train_drop.shape\n',train_drop.shape) (709903, 37)

y_train_drop=train_drop['acc_now_delinq']
X_train_drop=train_drop.drop(['member_id','acc_now_delinq'], axis = 1)
# 缺失值处理（利用均值填补）
train_drop = X_train_drop.fillna(X_train_drop.mean())

#字符串的替换--映射
train_drop['term']=train_drop['term'].map({' 36 months':1,' 60 months':2}).astype(int)
train_drop['grade'] = train_drop['grade'].map( {'D':3, 'A':6, 'E':2, 'B':5, 'C':4, 'F':1, 'G':0} ).astype(int)
train_drop['sub_grade']=train_drop['sub_grade'].map({['A1','A2','A3','A4','A5']:7,\
                                                     ['B1','B2','B3','B4','B5']:6,\
                                                     ['C1','C2','C3','C4','C5']:5,\
                                                     ['D1','D2','D3','D4','D5']:4,\
                                                     ['E1','E2','E3','E4','E5']:3,\
                                                     ['F1','F2','F3','F4','F5']:2,\
                                                     ['G1','G2','G3','G4','G5']:1,}).astype(int)
train_drop['verification_status_joint']=train_drop['verification_status_joint'].map({'Verified':3,'Source Verified':1,'Not Verified':0}).astype(int)
train_drop['application_type']=train_drop['application_type'].map({'JOINT':1,'INDIVIDUAL':0}).astype(int)
train_drop['initial_list_status']=train_drop['initial_list_status'].map({'f':1,'w':0}).astype(int)
train_drop['pymnt_plan']=train_drop['pymnt_plan'].map({'y':1,'n':0}).astype(int)
train_drop['loan_status']=train_drop['loan_status'].map({'Charged Off':5,'Fully Paid':4,'Current':3,'In Grace Period':2,\
                                                         'Late (31-120 days)':2,'Late (16-30 days)':1,'Default':3, \
                                                         'Does not meet the credit policy. Status:Fully Paid':0, \
                                                         'Does not meet the credit policy. Status:Charged Off':0,'Issued':1}).astype(int)
train_drop['verification_status']=train_drop['verification_status'].map({'Verified':3,'Source Verified':1,'Not Verified':0}).astype(int)
train_drop['home_ownership']=train_drop['home_ownership'].map({'NONE':0,'ANY':1,'OTHER':1,'OWN':3,'RENT':2,'MORTGAGE':2}).astype(int)
train_drop['emp_length']=train_drop['emp_length'].map({'< 1 year':0,'1 year':1,'2 years':1,'3 years':1,'4 years':2,'5 years':2, \
                                                       '6 years':2,'7 years':3,'8 years':3,'9 years':3,'10+ years':4}).astype(int)

end_time = time.time()
print('程序总共耗时:%d 秒'%int(end_time-start_time))
'''

'''
train_x, test_x, train_y, test_y = train_test_split(x_train_df, y_train_df, test_size=0.2, random_state=40)
gdbt = GradientBoostingRegressor()
gdbt.fit(train_x, train_y)
res0 = gdbt.predict(test_x)
print ('本地 GBDT:')
print (mean_squared_error(res0, test_y))
'''