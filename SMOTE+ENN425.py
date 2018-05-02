'''
Created on 2018年4月25日
@author: 杨少华
SMOTE+ENN
http://contrib.scikit-learn.org/imbalanced-learn/stable/auto_examples/combine/plot_smote_enn.html
'''
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import pandas as pd
from imblearn.combine import SMOTEENN

#print(__doc__) 显示开头注释的文本部分

# Generate the dataset
# 01 读取数据
import time
start_time=time.time()
df = pd.read_csv('data/train.csv')
train = df.drop(['sub_grade','verification_status_joint','addr_state','desc','emp_title','earliest_cr_line','issue_d','member_id', \
                        'purpose','title','zip_code','funded_amnt','annual_inc','verification_status','pymnt_plan', \
                        'pub_rec','pub_rec','total_rec_prncp','total_rec_int','recoveries',\
                        'collection_recovery_fee', 'collections_12_mths_ex_med',\
                        'annual_inc_joint', 'dti_joint','tot_coll_amt', 'tot_cur_bal' ,\
                         'open_il_12m','total_bal_il','max_bal_bc', 'all_util', \
                         'total_rev_hi_lim', 'inq_fi', 'total_cu_tl'], axis = 1)
# 02 缺失值处理（利用均值填补）
print("train 缺失值处理\n")
train = train.fillna(0)

# 03 字符串的替换--映射
print("map替换\n")
#删除grade因为sub_grade与之重复
train['emp_length']=train['emp_length'].map({'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5, \
                                                                  '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'n/a':0}).astype(float)

train['grade'] = train['grade'].map( {'D':4, 'A':1, 'E':5, 'B':2, 'C':3, 'F':6, 'G':7} ).astype(float)

train['term']=train['term'].map({' 36 months':1,' 60 months':0.8}).astype(float)
train['application_type']=train['application_type'].map({'JOINT':1.0,'INDIVIDUAL':0.5}).astype(float)
train['initial_list_status']=train['initial_list_status'].map({'f':0.5,'w':1}).astype(float)
#train['pymnt_plan']=train['pymnt_plan'].map({'y':1,'n':0}).astype(int)
train['loan_status']=train['loan_status'].map({'Charged Off':1,'Fully Paid':0,'Current':0,'In Grace Period':1,\
                                                         'Late (31-120 days)':1,'Late (16-30 days)':1,'Default':1, \
                                                         'Does not meet the credit policy. Status:Fully Paid':0, \
                                                         'Does not meet the credit policy. Status:Charged Off':1,'Issued':1}).astype(int)
#train['verification_status']=train['verification_status'].map({'Verified':1.0,'Source Verified':0.5,'Not Verified':0.0}).astype(float)
train['home_ownership']=train['home_ownership'].map({'NONE':0,'ANY':0.2,'OTHER':0.1,'OWN':1,'RENT':0.4,'MORTGAGE':0.8}).astype(float)


X = train.drop(['acc_now_delinq'], axis = 1)
y = train['acc_now_delinq']

'''
# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)

# Apply SMOTE + ENN
sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_sample(X, y)
X_res_vis = pca.transform(X_resampled)
'''

#print("initial RandomUnderSampler.\n")
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
#rus = RandomUnderSampler(random_state=0)
#X_resampled, y_resampled = rus.fit_sample(X, y)
#print(sorted(Counter(y_resampled).items())) [(0, 3293), (1, 3293)]

#print("RandomUnderSampler bootstrap the data replacement to True.\n")
import numpy as np
#print(np.vstack({tuple(row) for row in X_resampled}).shape)  (6586, 31)
#rus = RandomUnderSampler(random_state=0, replacement=True)
#X_resampled, y_resampled = rus.fit_sample(X, y)
#print(np.vstack({tuple(row) for row in X_resampled}).shape) (6577, 31)


#print("NearMiss version=1.\n")
from imblearn.under_sampling import NearMiss
#nm1 = NearMiss(random_state=0, version=1)
#X_resampled_nm1, y_resampled = nm1.fit_sample(X, y)
# print(sorted(Counter(y_resampled).items()))  [(0, 3293), (1, 3293)]

#print("EditedNearestNeighbours.\n")
sorted(Counter(y).items())
from imblearn.under_sampling import EditedNearestNeighbours
#enn = EditedNearestNeighbours(random_state=0)
#X_resampled, y_resampled = enn.fit_sample(X, y)
#print(sorted(Counter(y_resampled).items()))  [(0, 696760), (1, 3293)]

#print("RepeatedEditedNearestNeighbours.\n")
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
#renn = RepeatedEditedNearestNeighbours(random_state=0)
#X_resampled, y_resampled = renn.fit_sample(X, y)
#print(sorted(Counter(y_resampled).items())) [(0, 692716), (1, 3293)]

#print("under_sampling  AllKNN.\n")
from imblearn.under_sampling import AllKNN
#allknn = AllKNN(random_state=0)
#X_resampled, y_resampled = allknn.fit_sample(X, y)
#print(sorted(Counter(y_resampled).items()))

print("CondensedNearestNeighbour.\n")
from imblearn.under_sampling import CondensedNearestNeighbour
cnn = CondensedNearestNeighbour(random_state=0)
X_resampled, y_resampled = cnn.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

print("OneSidedSelection.\n")
from imblearn.under_sampling import OneSidedSelection
oss = OneSidedSelection(random_state=0)
X_resampled, y_resampled = oss.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

print("NeighbourhoodCleaningRule.\n")
from imblearn.under_sampling import NeighbourhoodCleaningRule
ncr = NeighbourhoodCleaningRule(random_state=0)
X_resampled, y_resampled = ncr.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))



print("{}{}".format("一共消耗时间为",int(time.time()-start_time)))