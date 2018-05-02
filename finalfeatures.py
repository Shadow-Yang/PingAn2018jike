'''
Created on 2018年4月25日
@author: 杨少华
经过周密的特征筛选得到最后需要的特征以及，
多值有序变量的map以及多值无序变量的映射，得到最优的训练特征集合的确定
特征工程完备下的最优预测结果
'''
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

####################################    读取数据+填充缺失值        ########################################
start_time=time.time()
features_list = ['acc_now_delinq','loan_amnt','funded_amnt_inv','term','int_rate','installment','grade', \
'emp_length','annual_inc','verification_status','dti','total_rec_int', \
'revol_util','total_acc','revol_bal','out_prncp','out_prncp_inv','total_pymnt_inv', \
'total_rec_prncp', 'mths_since_last_record','mths_since_last_major_derog','tot_coll_amt', \
'tot_cur_bal','total_rev_hi_lim','home_ownership','loan_status','total_pymnt']
df = pd.read_csv('data/train.csv')

#train.shape  (709903, 26)
train = df[features_list]
#'mths_since_last_record','mths_since_last_major_derog' 两个特征确实严重，填充0作为常识
train = train.fillna(0)


###########################       多值无序变量，多值有序变量处理：       #################################
train['emp_length']=train['emp_length'].map({'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5, \
                                                                  '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'n/a':0}).astype(float)
train['grade'] = train['grade'].map( {'D':4, 'A':1, 'E':5, 'B':2, 'C':3, 'F':6, 'G':7} ).astype(float)
train['loan_status']=train['loan_status'].map({'Charged Off':1,'Fully Paid':0,'Current':0,'In Grace Period':1,\
                                                         'Late (31-120 days)':1,'Late (16-30 days)':1,'Default':1, \
                                                         'Does not meet the credit policy. Status:Fully Paid':0, \
                                                         'Does not meet the credit policy. Status:Charged Off':1,'Issued':1}).astype(int)

#print("缺失值处理多值无序变量处理：\n")
#check_null = train.isnull().sum(axis=0).sort_values(ascending=False)/float(len(train)) #查看缺失值比例
#print(check_null[check_null > 0.1]) # 查看缺失比例大于20%的属性。
#objectColumns = train.select_dtypes(include=["object"]).columns
#print(train[objectColumns].isnull().sum().sort_values(ascending=False))

#"多值无序变量处理
n_columns = ["home_ownership", "verification_status", "term"] 
dummy_df = pd.get_dummies(train[n_columns])# 用get_dummies进行one hot编码
loans = pd.concat([train, dummy_df], axis=1)#当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
#筛选包含home_ownership的所有变量
#print(loans.loc[:,loans.columns.str.contains("home_ownership")].head())
loans = loans.drop(n_columns, axis=1)  #清除原来的分类变量
#用pandas的info( )的方法作最后检查，发现已经将所有类型为object的变量作转化，所有数据类型均满足下一步算法的要求。
#print(loans.info())
#我们采用的是标准化的方法，与归一化相比，标准化的方法保持了异常值所包含的有用信息，并且使得算法受到异常值的影响较小。
col = loans.select_dtypes(include=['int64','float64']).columns
loans_ml_df = loans # 复制数据至变量loans_ml_df


##############################             标准化                                           ##########################
from sklearn.preprocessing import StandardScaler # 导入模块
sc =StandardScaler() # 初始化缩放器
loans_ml_df[col] =sc.fit_transform(loans_ml_df[col]) #对数据进行标准化
#print(loans_ml_df.head()) #查看经标准化后的数据
print(type(loans_ml_df))

############################       过采样处理数据并运用逻辑斯谛回归模型           ##############################################
# 构建自变量和因变量
X = loans_ml_df.drop(['acc_now_delinq'],axis=1)
y = loans_ml_df['acc_now_delinq']
n_sample = y.shape[0]
n_pos_sample = y[y == 0].shape[0] #正样本
n_neg_sample = y[y == 1].shape[0] #负样本
#样本个数：709903; 正样本占99.54%; 负样本占0.46%

from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块
# 处理不平衡数据
sm = SMOTE(random_state=42)    # 处理过采样的方法
X, y = sm.fit_sample(X, y)
n_sample = y.shape[0]
n_pos_sample = y[y == 0].shape[0]
n_neg_sample = y[y == 1].shape[0]
#样本个数：1413220; 正样本占50.00%; 负样本占50.00%
########################   XGB    #################
import xgboost as xgb
from sklearn.cross_validation import train_test_split
train_x, test_x, train_y, test_y=train_test_split(X,y,random_state=0)
 
dtrain=xgb.DMatrix(train_x,label=train_y)
dtest=xgb.DMatrix(test_x)
 
params={'booster':'gbtree',
    #'objective': 'reg:linear',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}
 
watchlist = [(dtrain,'train')]
 
bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
 
ypred=bst.predict(dtest)
 
# 设置阈值, 输出一些评价指标
list3 = list(ypred)
for i in range(len(list3)):
    if list3[i] > 0.5:
        list3[i] = 1
    else:
        list3[i] = 0
#y_pred = (ypred >= 0.5)*1
 
#模型校验
from sklearn import metrics
#print ('AUC: %.4f' % metrics.roc_auc_score(test_y,list3))
#print ('ACC: %.4f' % metrics.accuracy_score(test_y,list3))
print ('Recall: %.4f' % metrics.recall_score(test_y,list3))
#print ('F1-score: %.4f' %metrics.f1_score(test_y,list3))
print ('Precesion: %.4f' %metrics.precision_score(test_y,list3))
metrics.confusion_matrix(test_y,list3)
 
 
 
print("xgboost:") 
print("accuracy on the training subset:{:.3f}".format(bst.get_score(train_x,train_y)))
print("accuracy on the test subset:{:.3f}".format(bst.get_score(test_x,test_y)))
print('Feature importances:{}'.format(bst.get_fscore()))
 
 
'''
AUC: 0.8135
ACC: 0.7640
Recall: 0.9641
F1-score: 0.8451
Precesion: 0.7523
 
#特征重要性和随机森林差不多
Feature importances:{'Account Balance': 80, 'Duration of Credit (month)': 119,
 'Most valuable available asset': 54, 'Payment Status of Previous Credit': 84,
 'Value Savings/Stocks': 66, 'Age (years)': 94, 'Credit Amount': 149,
 'Type of apartment': 20, 'Instalment per cent': 37,
 'Length of current employment': 70, 'Sex & Marital Status': 29,
 'Purpose': 67, 'Occupation': 13, 'Duration in Current address': 25,
 'Telephone': 15, 'Concurrent Credits': 23, 'No of Credits at this Bank': 7,
 'Guarantors': 28, 'No of dependents': 6}
'''


'''
##################################    01构建逻辑回归分类器             #################################
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression() 
clf1.fit(X, y)
predicted1 = clf1.predict(X) # 通过分类器产生预测结果
from sklearn.metrics import accuracy_score
#print("Test set accuracy score: {:.5f}".format(accuracy_score(predicted1, y,)))
#采用classification_report快速查看混淆矩阵precision、recall、f1-score的计算值。
from sklearn.metrics import classification_report
#print(classification_report(y, predicted1))
from sklearn.metrics import roc_auc_score
roc_auc1 = roc_auc_score(y, predicted1)
#print("Area under the ROC curve : %f" % roc_auc1)

####  模型优化    cross-validation+grid search    ####
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) # random_state = 0 每次切分的数据都一样
# 构建参数组合
param_grid = {'C': [0.01,0.1, 1, 10, 100, 1000,],
                            'penalty': [ 'l1', 'l2']}
grid_search = GridSearchCV(LogisticRegression(),  param_grid, cv=10) # 确定模型LogisticRegression，和参数组合param_grid ，cv指定5折
grid_search.fit(X_train, y_train) # 使用训练集学习算法

results = pd.DataFrame(grid_search.cv_results_)
best = np.argmax(results.mean_test_score.values)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))
scores = np.array(results.mean_test_score).reshape(2, 6)
sns.heatmap(scores, ylabel='penalty', yticklabels=param_grid['penalty'],
                      xlabel='C', xticklabels=param_grid['C'], cmap="viridis")

print("Best estimator:\n{}".format(grid_search.best_estimator_))#grid_search.best_estimator_ 返回模型以及他的所有参数（包含最优参数）

y_pred = grid_search.predict(X_test)
print("Test set accuracy score: {:.5f}".format(accuracy_score(y_test, y_pred,)))

print(classification_report(y_test, y_pred))

roc_auc2 = roc_auc_score(y_test, y_pred)
print("Area under the ROC curve : %f" % roc_auc2)

from sklearn.externals import joblib
#lr是一个LogisticRegression模型
joblib.dump(grid_search, 'lr.model')
lr = joblib.load('lr.model')
'''
print("{}{}{}".format("一共消耗时间为",int(time.time()-start_time),"秒"))