import pandas as pd
import time

# 01 读取数据
start_time=time.time()
test = pd.read_csv('data/test.csv')
member = test['member_id']
test_drop=test.drop(['verification_status_joint','addr_state','desc','emp_title','earliest_cr_line','issue_d','member_id', \
                        'purpose','title','zip_code'], axis = 1)
df = pd.read_csv('data/train.csv')
train = df.drop(['verification_status_joint','addr_state','desc','emp_title','earliest_cr_line','issue_d','member_id', \
                        'purpose','title','zip_code'], axis = 1)

# 缺失值处理（利用均值填补）
print("train 缺失值处理\n")
train = train.fillna(0)
print("test 缺失值处理\n")
test_drop = test_drop.fillna(0)

#字符串的替换--映射
print("map替换\n")

train['term']=train['term'].map({' 36 months':1,' 60 months':0.8}).astype(float)
train['grade'] = train['grade'].map( {'D':0.4, 'A':1, 'E':0.2, 'B':0.8, 'C':0.6, 'F':0.1, 'G':0.05} ).astype(float)
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
                                                         'Late (31-120 days)':0.1,'Late (16-30 days)':0.3,'Default':0, \
                                                         'Does not meet the credit policy. Status:Fully Paid':0, \
                                                         'Does not meet the credit policy. Status:Charged Off':0,'Issued':0}).astype(float)
train['verification_status']=train['verification_status'].map({'Verified':1.0,'Source Verified':0.5,'Not Verified':0.0}).astype(float)
train['home_ownership']=train['home_ownership'].map({'NONE':0,'ANY':0.2,'OTHER':0.1,'OWN':1,'RENT':0.4,'MORTGAGE':0.8}).astype(float)
train['emp_length']=train['emp_length'].map({'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5, \
                                             '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'n/a':0}).astype(float)

#字符串的替换--映射
print("test map替换\n")
test_drop['term']=test_drop['term'].map({' 36 months':1,' 60 months':0.8}).astype(float)
test_drop['grade'] = test_drop['grade'].map( {'D':0.4, 'A':1, 'E':0.2, 'B':0.8, 'C':0.6, 'F':0.1, 'G':0.05} ).astype(float)
test_drop['sub_grade']=test_drop['sub_grade'].map({'A1':1,'A2':1.2,'A3':1.3,'A4':1.4,'A5':1.5,\
                                           'B1':2,'B2':2.2,'B3':2.3,'B4':2.4,'B5':2.5,\
                                           'C1':3,'C2':3.2,'C3':3.3,'C4':3.4,'C5':3.5,\
                                           'D1':4,'D2':4.2,'D3':4.3,'D4':4.4,'D5':4.5,\
                                           'E1':5,'E2':5.2,'E3':5.3,'E4':5.4,'E5':5.5,\
                                           'F1':6,'F2':6.2,'F3':6.3,'F4':6.4,'F5':6.5,\
                                           'G1':7,'G2':7.2,'G3':7.3,'G4':7.4,'G5':7.5,'0':10}).astype(float)
test_drop['application_type']=test_drop['application_type'].map({'JOINT':1.0,'INDIVIDUAL':0.5}).astype(float)
test_drop['initial_list_status']=test_drop['initial_list_status'].map({'f':0.5,'w':1}).astype(float)
test_drop['pymnt_plan']=test_drop['pymnt_plan'].map({'y':1,'n':0}).astype(int)
test_drop['loan_status']=test_drop['loan_status'].map({'Charged Off':1,'Fully Paid':1,'Current':0.6,'In Grace Period':0.8,\
                                                         'Late (31-120 days)':0.1,'Late (16-30 days)':0.3,'Default':0, \
                                                         'Does not meet the credit policy. Status:Fully Paid':0, \
                                                         'Does not meet the credit policy. Status:Charged Off':0,'Issued':0}).astype(float)
test_drop['verification_status']=test_drop['verification_status'].map({'Verified':1.0,'Source Verified':0.5,'Not Verified':0.0}).astype(float)
test_drop['home_ownership']=test_drop['home_ownership'].map({'NONE':0,'ANY':0.2,'OTHER':0.1,'OWN':1,'RENT':0.4,'MORTGAGE':0.8}).astype(float)
test_drop['emp_length']=test_drop['emp_length'].map({'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5, \
                                             '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'n/a':0}).astype(float)
train0 = train[train['acc_now_delinq'] == 0]
train1 = train[train['acc_now_delinq'] == 1]
y_train0=train0['acc_now_delinq']
y_train1=train1['acc_now_delinq']
train_drop0=train0.drop(['acc_now_delinq'], axis = 1)
train_drop1=train1.drop(['acc_now_delinq'], axis = 1)

'''"随机抽取正例32930/16465/6586行数据,10:1,5:1,2:1,9次提高分别训练精度\n'''
print("01.产生3类不同比例 6586 的9个数据集并放入data[]集合中\n")
data = []
for i in range(3):
    df2 = pd.DataFrame()
    df2 = train0.sample(n=16465,axis = 0)
    data.append(pd.concat([df2,train1],ignore_index=True))
    df5 = pd.DataFrame()
    df5 = train0.sample(n=32930,axis = 0)
    data.append(pd.concat([df5,train1],ignore_index=True))
    df10 = pd.DataFrame()
    df10 = train0.sample(n=49395,axis = 0)
    data.append(pd.concat([df10,train1],ignore_index=True))

''' 开始进行线下测试模型训练，循环3个模型在3种不同情况下的训练 '''
print("02.引入3个选定的模型\n")
#from sklearn.linear_model import LogisticRegression, Ridge
import lightgbm as lgb
from xgboost import XGBRegressor
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

model01 = lgb.LGBMRegressor(colsample_bytree=0.5,
                         learning_rate=0.03,
                         subsample=0.8,
                         num_leaves=3,
                         objective='binary',
                         metric='binary_logloss',
                         n_estimators=2000,
                         seed=0)
model02 = lgb.LGBMRegressor(colsample_bytree=0.5,
                         learning_rate=0.03,
                         subsample=0.8,
                         num_leaves=3,
                         objective='binary',
                         metric='binary_logloss',
                         n_estimators=2000,
                         seed=0)
model03 = lgb.LGBMRegressor(colsample_bytree=0.5,
                         learning_rate=0.03,
                         subsample=0.8,
                         num_leaves=3,
                         objective='binary',
                         metric='binary_logloss',
                         n_estimators=2000,
                         seed=0)


model04 = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=8, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1)
model05 = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=8, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1)
model06 = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=8, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1)

model07 = GradientBoostingRegressor(learning_rate=0.005,
                                    n_estimators=1500,
                                    max_depth=9, 
                                    min_samples_split=1200,
                                    min_samples_leaf=60,
                                    subsample=0.85,
                                    random_state=10, 
                                    max_features=7,
                                    warm_start=True)
model08 = GradientBoostingRegressor(learning_rate=0.005,
                                    n_estimators=1500,
                                    max_depth=9, 
                                    min_samples_split=1200,
                                    min_samples_leaf=60,
                                    subsample=0.85,
                                    random_state=10, 
                                    max_features=7,
                                    warm_start=True)
model09 = GradientBoostingRegressor(learning_rate=0.005,
                                    n_estimators=1500,
                                    max_depth=9, 
                                    min_samples_split=1200,
                                    min_samples_leaf=60,
                                    subsample=0.85,
                                    random_state=10, 
                                    max_features=7,
                                    warm_start=True)

print("03.9个基本模型的训练\n")

print(" ...model01...\n")
x_train_df01 = data[0].drop(['acc_now_delinq'], axis = 1)
y_train_df01 = data[0]['acc_now_delinq']
model01.fit(x_train_df01,y_train_df01)
print(" ...model02...\n")
x_train_df02 = data[1].drop(['acc_now_delinq'], axis = 1)
y_train_df02 = data[1]['acc_now_delinq']
model02.fit(x_train_df02,y_train_df02)
print(" ...model03...\n")
x_train_df03 = data[2].drop(['acc_now_delinq'], axis = 1)
y_train_df03 = data[2]['acc_now_delinq']
model03.fit(x_train_df03,y_train_df03)

print(" ...model04...\n")
x_train_df01 = data[3].drop(['acc_now_delinq'], axis = 1)
y_train_df01 = data[3]['acc_now_delinq']
model04.fit(x_train_df01,y_train_df01)
print(" ...model05...\n")
x_train_df02 = data[4].drop(['acc_now_delinq'], axis = 1)
y_train_df02 = data[4]['acc_now_delinq']
model05.fit(x_train_df02,y_train_df02)
print(" ...model06...\n")
x_train_df03 = data[5].drop(['acc_now_delinq'], axis = 1)
y_train_df03 = data[5]['acc_now_delinq']
model06.fit(x_train_df03,y_train_df03)

print(" ...model07...\n")
x_train_df01 = data[6].drop(['acc_now_delinq'], axis = 1)
y_train_df01 = data[6]['acc_now_delinq']
model07.fit(x_train_df01,y_train_df01)
print(" ...model08...\n")
x_train_df02 = data[7].drop(['acc_now_delinq'], axis = 1)
y_train_df02 = data[7]['acc_now_delinq']
model08.fit(x_train_df02,y_train_df02)
print(" ...model09...\n")
x_train_df03 = data[8].drop(['acc_now_delinq'], axis = 1)
y_train_df03 = data[8]['acc_now_delinq']
model09.fit(x_train_df03,y_train_df03)

print("04.随机选取2w行train集中的数据进行线下测试：\n")
df0 = pd.DataFrame()
df0 = train0.sample(n=20000,axis = 0)
df1 = pd.DataFrame()
df1 = train1.sample(n=94,axis = 0)
df = pd.DataFrame()
df = pd.concat([df0,df1],ignore_index=True)
y_df =df['acc_now_delinq']
x_df = df.drop(['acc_now_delinq'], axis = 1)

list1 = list(model01.predict(x_df))
for i in range(len(list1)):
    if list1[i] > 0.5:
        list1[i] = 1
    else:
        list1[i] = 0
list2 = list(model02.predict(x_df))
for i in range(len(list2)):
    if list2[i] > 0.5:
        list2[i] = 1
    else:
        list2[i] = 0
list3 = list(model03.predict(x_df))
for i in range(len(list3)):
    if list3[i] > 0.5:
        list3[i] = 1
    else:
        list3[i] = 0

list4 = list(model04.predict(x_df))
for i in range(len(list4)):
    if list4[i] > 0.5:
        list4[i] = 1
    else:
        list4[i] = 0
list5 = list(model05.predict(x_df))
for i in range(len(list5)):
    if list5[i] > 0.5:
        list5[i] = 1
    else:
        list5[i] = 0
list6 = list(model06.predict(x_df))
for i in range(len(list6)):
    if list6[i] > 0.5:
        list6[i] = 1
    else:
        list6[i] = 0

list7 = list(model07.predict(x_df))
for i in range(len(list7)):
    if list7[i] > 0.5:
        list7[i] = 1
    else:
        list7[i] = 0
list8 = list(model08.predict(x_df))
for i in range(len(list8)):
    if list8[i] > 0.5:
        list8[i] = 1
    else:
        list8[i] = 0
list9 = list(model09.predict(x_df))
for i in range(len(list9)):
    if list9[i] > 0.5:
        list9[i] = 1
    else:
        list9[i] = 0

result_down = []
for i in range(len(list7)):
    step1 = 0.1*list1[i] + 0.2*list2[i] + 0.7*list3[i]
    step2 = 0.1*list4[i] + 0.2*list5[i] + 0.7*list6[i]
    step3 = 0.1*list7[i] + 0.2*list8[i] + 0.7*list9[i]
    result_down.append(0.1*step1 + 0.8*step2 + 0.1*step3)

print("05.处理异常值:\n")
for i in range(len(result_down)):
    if result_down[i] > 0.5:
        result_down[i] = 1
    else:
        result_down[i] = 0
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(result_down)):
    if result_down[i] == list(y_df)[i]:
        if result_down[i] == 1:
            tp = tp + 1
        else:
            tn = tn + 1
    else:
        if result_down[i] == 1:
            fp = fp + 1
        else:
            fn = fn + 1     
p = tp / (tp + fp)
r = tp / (tp + fn)

print ('精确率为:%.5f\n'%float(p))
print ('召回率为:%.5f\n'%float(r))
F2 = (5*p*r)/(4*p+r)
print ('F2值为:%.5f\n'%float(F2))

end_time = time.time()
print('程序总共耗时:%d 秒'%int(end_time-start_time))

'''****************************test提交训练*************************************'''
'''
print("06 开始进行线上预测模型\n")
x_test = test_drop
list1 = list(model01.predict(x_test))
list2 = list(model02.predict(x_test))
list3 = list(model03.predict(x_test))
list4 = list(model04.predict(x_test))
list5 = list(model05.predict(x_test))
list6 = list(model06.predict(x_test))
list7 = list(model07.predict(x_test))
list8 = list(model08.predict(x_test))
list9 = list(model09.predict(x_test))
result_up = []
for i in range(len(list1)):
    step1 = 0.1*list1[i] + 0.2*list2[i] + 0.7*list3[i]
    step2 = 0.1*list4[i] + 0.2*list5[i] + 0.7*list6[i]
    step3 = 0.1*list7[i] + 0.2*list8[i] + 0.7*list9[i]
    result_up.append(0.1*step1 + 0.8*step2 + 0.1*step3)

print("07.线上数据异常值处理:\n")
for i in range(len(result_up)):
    if result_up[i] > 0.5:
        result_up[i] = 1
    else:
        result_up[i] = 0

pd_result1 = pd.DataFrame({'member_id': member, 'acc_now_delinq': result_up})
#print(pd_result1) 可用于显示文件对应的结构
pd_result1.to_csv("data/4241132p.csv",columns=['member_id','acc_now_delinq'],header=True, index=False, encoding='utf8')
'''
end_time = time.time()
print('程序总共耗时:%d 秒'%int(end_time-start_time))