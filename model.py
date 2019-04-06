# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_absolute_error
import winsound

dat1 = pd.read_csv('policy_0702.csv',encoding='UTF-8')
dat2 = pd.read_csv('claim_0702.csv',encoding='UTF-8')
dat3 = pd.read_csv('training-set.csv',encoding='UTF-8')
dat4 = pd.read_csv('testing-set.csv',encoding='UTF-8')


# 資料處理

dat1['Cancellation'] = dat1['Cancellation'].apply({'Y':1, ' ':0}.get)
dat1['Main_Insurance_Coverage_Group'] = dat1['Main_Insurance_Coverage_Group'].apply({'車責':0, '竊盜':1, '車損':2}.get)
dat1['ibirth'] = dat1['ibirth'].str[3:7].astype('float') + (dat1['ibirth'].str[0:2].astype('float')-1)/12
dat1['ibirth'] = dat1['ibirth'].fillna(dat1['ibirth'].mean())
dat1['dbirth'] = dat1['dbirth'].str[3:7].astype('float') + (dat1['dbirth'].str[0:2].astype('float')-1)/12
dat1['dbirth'] = dat1['dbirth'].fillna(dat1['dbirth'].mean())
dat1['nequipment9'] = dat1['nequipment9'].apply({    '                                                                                                    ':1,
    ' ':1,    '原裝車含配備':2,'5合1影音':3,'大包':4,
    '伸尾                                                                                                ':5}.get)
dat1.fsex = dat1.fsex.apply({'1':1,'2':2,' ':0}.get).fillna(0)
dat1.fmarriage = dat1.fmarriage.apply({'1':1,'2':2,' ':0}.get).fillna(0)

# 單一保單個數
dat1['Policy_counts'] = dat1['Policy_Number'].map(dat1['Policy_Number'].value_counts())



dat1['Main_Insurance_Coverage_Group'] = dat1['Main_Insurance_Coverage_Group'].astype('category')
dat1['Insurance_Coverage'] = dat1['Insurance_Coverage'].astype('category')
dat1['Cancellation'] = dat1['Cancellation'].astype('category')


# dat1[13:19] 要特殊處理

a1 = dat1.groupby(by ='Policy_Number',axis=0,sort=False).Main_Insurance_Coverage_Group.value_counts().\
                    reset_index(name='Main_Insurance_count')
a1 = a1.pivot_table(index='Policy_Number', columns='Main_Insurance_Coverage_Group',\
                     values='Main_Insurance_count',fill_value=0)
a2 = dat1.pivot_table(index=['Policy_Number'],columns='Main_Insurance_Coverage_Group',\
                      values=['Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3',\
                              'Coverage_Deductible_if_applied'],fill_value=0)
a3 = dat1.groupby(by ='Policy_Number',axis=0,sort=False).Insurance_Coverage.value_counts().\
                reset_index(name='Insurance_Coverage_count')
a3 = a3.pivot_table(index='Policy_Number', columns='Insurance_Coverage',\
                    values='Insurance_Coverage_count',fill_value=0)

# 補缺失值
dat1.Vehicle_identifier = dat1.Vehicle_identifier.fillna(dat1.Policy_Number)
dat1.Prior_Policy_Number = dat1.Prior_Policy_Number.fillna('0')

dat1.count()[dat1.count()<1747942]

# 將 Insured's_ID 轉數字
dat1["Insured's_ID"] = pd.Categorical(dat1["Insured's_ID"])
dat1["Insured's_ID"] = dat1["Insured's_ID"].cat.codes

dat1.Vehicle_identifier = pd.Categorical(dat1.Vehicle_identifier)
dat1.Vehicle_identifier = dat1.Vehicle_identifier.cat.codes
dat1.Vehicle_Make_and_Model1 = pd.Categorical(dat1.Vehicle_Make_and_Model1)
dat1.Vehicle_Make_and_Model1 = dat1.Vehicle_Make_and_Model1.cat.codes
dat1.Vehicle_Make_and_Model2 = pd.Categorical(dat1.Vehicle_Make_and_Model2)
dat1.Vehicle_Make_and_Model2 = dat1.Vehicle_Make_and_Model2.cat.codes
dat1.Distribution_Channel = pd.Categorical(dat1.Distribution_Channel)
dat1.Distribution_Channel = dat1.Distribution_Channel.cat.codes
dat1.aassured_zip = pd.Categorical(dat1.aassured_zip)
dat1.aassured_zip = dat1.aassured_zip.cat.codes
dat1.iply_area = pd.Categorical(dat1.iply_area)
dat1.iply_area = dat1.iply_area.cat.codes
dat1.Prior_Policy_Number = pd.Categorical(dat1.Prior_Policy_Number)
dat1.Prior_Policy_Number = dat1.Prior_Policy_Number.cat.codes
dat1["Coding_of_Vehicle_Branding_&_Type"] = pd.Categorical(dat1["Coding_of_Vehicle_Branding_&_Type"])
dat1["Coding_of_Vehicle_Branding_&_Type"] = dat1["Coding_of_Vehicle_Branding_&_Type"].cat.codes
dat1.ibirth = dat1.ibirth.astype('float')
dat1.dbirth = dat1.dbirth.astype('float')
del dat1['fpt']


dat1.dtypes[dat1.dtypes=='object']


dat = dat1[list(dat1.columns[0:12]) +list(dat1.columns[20:43]) ].drop_duplicates().reset_index(drop=True)
dat['Premium'] = dat1.groupby('Policy_Number',sort=False)['Premium'].sum().values
dat = pd.merge(dat, a1, on=['Policy_Number'], how='left')
dat = pd.merge(dat, a2, on=['Policy_Number'], how='left')
dat = pd.merge(dat, a3, on=['Policy_Number'], how='left')


dat.isnull().sum()[dat.isnull().sum()>0]

# 處理dat2資料

a4 = dat2[['Policy_Number','Claim_Number']].drop_duplicates().reset_index(drop=True)
a4['acc_count'] = 1
dat2['acc_count'] = dat2['Policy_Number'].map(a4.groupby('Policy_Number',sort=False)['acc_count'].sum())

dat2['DOB_of_Driver'] = dat2['DOB_of_Driver'].str[3:7].astype('float') + \
        (dat2['DOB_of_Driver'].str[0:2].astype('float')-1)/12
dat2['DOB_of_Driver'] = dat2['DOB_of_Driver'].fillna(dat2['DOB_of_Driver'].mean())

dat2['Accident_Time'] = dat2['Accident_Time'].str[0:2].astype('float')*60 + \
dat2['Accident_Time'].str[3:5].astype('float')
dat2["At_Fault?"] = dat2["At_Fault?"].fillna(dat2["At_Fault?"].mean()  )
dat2['Vehicle_identifier'] = dat2['Vehicle_identifier'].fillna(dat2['Policy_Number']  )
dat2.Accident_Date = (dat2.Accident_Date.str[0:4].astype('float'))+\
                        ( dat2.Accident_Date.str[5:7].astype('float')-1)/12

dat2["Cause_of_Loss"] = pd.Categorical(dat2["Cause_of_Loss"])
dat2["Cause_of_Loss"] = dat2["Cause_of_Loss"].cat.codes

dat2["Vehicle_identifier"] = pd.Categorical(dat2["Vehicle_identifier"])
dat2["Vehicle_identifier"] = dat2["Vehicle_identifier"].cat.codes

dat2["Accident_area"] = pd.Categorical(dat2["Accident_area"])
dat2["Accident_area"] = dat2["Accident_area"].cat.codes


dat2.isnull().sum()[dat2.isnull().sum()>0]


a5 = dat2.groupby(by ='Policy_Number',axis=0,sort=False).Coverage.value_counts().\
        reset_index(name='Coverage_count')
a5 = a5.pivot_table(index='Policy_Number', columns='Coverage', values='Coverage_count',fill_value=0)
dat = pd.merge(dat, a5, on=['Policy_Number'], how='left',sort=False).fillna(0)

#dat2.columns
dat2['Nature_of_the_claim'] = dat2['Nature_of_the_claim'].astype('category')
dat2["Driver's_Gender"] = dat2["Driver's_Gender"].astype('category')
dat2["Driver's_Relationship_with_Insured"] = dat2["Driver's_Relationship_with_Insured"].astype('category')
dat2['Marital_Status_of_Driver'] = dat2['Marital_Status_of_Driver'].astype('category')
dat2['Claim_Status_(close,_open,_reopen_etc)'] = dat2['Claim_Status_(close,_open,_reopen_etc)'].astype('category')
dat2['Accident_area'] = dat2['Accident_area'].astype('category')
del dat2['Vehicle_identifier'] 

df2 = dat2[['Policy_Number','Claim_Number',"Driver's_Gender",'DOB_of_Driver','Marital_Status_of_Driver', \
            'Accident_Date','Cause_of_Loss','Accident_area','number_of_claimants', \
            'Accident_Time',"At_Fault?"]].drop_duplicates().reset_index(drop=True)



# Nature_of_the_claim
a6 = dat2.groupby('Policy_Number',sort=False)['Nature_of_the_claim'].value_counts().\
        reset_index(name='Nature_of_the_claim_count')
a6 = a6.pivot_table(index='Policy_Number', columns='Nature_of_the_claim',\
                    values='Nature_of_the_claim_count',fill_value=0)

# Marital_Status_of_Driver
a7 = df2.groupby('Policy_Number',sort=False)['Marital_Status_of_Driver'].value_counts().\
            reset_index(name='Marital_Status_of_Driver_count')
a7 = a7.pivot_table(index='Policy_Number', columns='Marital_Status_of_Driver', \
                    values='Marital_Status_of_Driver_count',fill_value=0)

# Driver's_Gender
a8 = df2.groupby('Policy_Number',sort=False)["Driver's_Gender"].value_counts().reset_index(name=\
                "Driver's_Gender_count")
a8 = a8.pivot_table(index='Policy_Number', columns="Driver's_Gender",values=\
                    "Driver's_Gender_count",fill_value='0')

# Driver's_Relationship_with_Insured
a9 = dat2.groupby('Policy_Number',sort=False)["Driver's_Relationship_with_Insured"].value_counts().\
                   reset_index(name="Driver's_Relationship_with_Insured_count")
a9 = a9.pivot_table(index='Policy_Number', columns="Driver's_Relationship_with_Insured", \
                    values="Driver's_Relationship_with_Insured_count",fill_value='0')

# DOB_of_Driver
a10 = df2.groupby('Policy_Number',sort=False)["DOB_of_Driver"].mean().reset_index(name='DOB_of_Driver(mean)')

# Accident_Date
a11 = df2.groupby('Policy_Number',sort=False)["Accident_Date"].mean().reset_index(name='Accident_Date(mean)')

#Cause_of_Loss
a12 = df2.groupby('Policy_Number',sort=False)["Cause_of_Loss"].value_counts().\
                reset_index(name="Cause_of_Loss_count")
a12 = a12.pivot_table(index='Policy_Number', columns="Cause_of_Loss",\
                      values="Cause_of_Loss_count",fill_value='0')

# Accident_area
a13 = df2.groupby('Policy_Number',sort=False)["Accident_area"].value_counts().\
                reset_index(name="Accident_area_count")
a13 = a13.pivot_table(index='Policy_Number', columns="Accident_area",values="Accident_area_count",fill_value='0')

# number_of_claimants
a14 = df2.groupby('Policy_Number',sort=False)["number_of_claimants"].sum().\
                        reset_index(name="number_of_claimants(sum)")
a15 = df2.groupby('Policy_Number',sort=False)["number_of_claimants"].mean().\
                        reset_index(name="number_of_claimants(mean)")

# Accident_Time
a16 = df2.groupby('Policy_Number',sort=False)["Accident_Time"].sum().reset_index(name="Accident_Time(sum)")
a17 = df2.groupby('Policy_Number',sort=False)["Accident_Time"].mean().reset_index(name="Accident_Time(mean)")

# Paid_Loss_Amount
a18 = dat2.groupby('Policy_Number',sort=False)["Paid_Loss_Amount"].sum().reset_index(name="Paid_Loss_Amount(sum)")

# paid_Expenses_Amount
a19 = dat2.groupby('Policy_Number',sort=False)["paid_Expenses_Amount"].sum().reset_index(name="paid_Expenses_Amount(sum)")

# Salvage_or_Subrogation?
a20 = dat2.groupby('Policy_Number',sort=False)["Salvage_or_Subrogation?"].sum().reset_index(name="Salvage_or_Subrogation?(sum)")

# At_Fault
a21 = df2.groupby('Policy_Number',sort=False)["At_Fault?"].sum().reset_index(name="At_Fault?(sum)") 

# Claim_Status_(close,_open,_reopen_etc)
a22 = dat2.groupby('Policy_Number',sort=False)['Claim_Status_(close,_open,_reopen_etc)'].value_counts().\
        reset_index(name='Claim_Status_count')
a22 = a22.pivot_table(index='Policy_Number', columns='Claim_Status_(close,_open,_reopen_etc)', \
                      values='Claim_Status_count',fill_value = 0)

# Deductible
a23 = dat2.groupby('Policy_Number',sort=False)["Deductible"].sum().reset_index(name="Deductible(sum)")

a8 = a8.astype('float')
a9 = a9.astype('float')
a12 = a12.astype('float')
a13 = a13.astype('float')
a6.columns = ['Nature_of_the_claim_1','Nature_of_the_claim_2' ]
a7.columns = ['Marital_Status_of_Driver_1','Marital_Status_of_Driver_2' ]
a8.columns = ["Drivers_Gender_1","Drivers_Gender_2" ]
a9.columns = ["Drivers_Relationship_with_Insured_1","Drivers_Relationship_with_Insured_2","Drivers_Relationship_with_Insured_3",
             "Drivers_Relationship_with_Insured_4","Drivers_Relationship_with_Insured_5","Drivers_Relationship_with_Insured_6",
             "Drivers_Relationship_with_Insured_7"]
a12.columns = ['Cause_of_Loss_0','Cause_of_Loss_1','Cause_of_Loss_2','Cause_of_Loss_3','Cause_of_Loss_4','Cause_of_Loss_5',
              'Cause_of_Loss_6','Cause_of_Loss_7','Cause_of_Loss_8','Cause_of_Loss_9','Cause_of_Loss_10','Cause_of_Loss_11',
               'Cause_of_Loss_12','Cause_of_Loss_13','Cause_of_Loss_14','Cause_of_Loss_15','Cause_of_Loss_16']
a13.columns = ['Accident_area_0','Accident_area_1','Accident_area_2','Accident_area_3','Accident_area_4','Accident_area_5',
               'Accident_area_6','Accident_area_7','Accident_area_8','Accident_area_9','Accident_area_10','Accident_area_11',
               'Accident_area_12','Accident_area_13','Accident_area_14','Accident_area_15','Accident_area_16',
               'Accident_area_17','Accident_area_18','Accident_area_19','Accident_area_20','Accident_area_21']
a22.columns = ["Claim_Status__0","Claim_Status__1"]

# 合併資料
dat = pd.merge(dat, a6, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a7, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a8, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a9, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a10, on=['Policy_Number'], how='left',sort=False).fillna(0)

dat = pd.merge(dat, a11, on=['Policy_Number'], how='left',sort=False)
dat['ACC_Y_or_N'] = dat['Accident_Date(mean)'].isna()
dat['Accident_Date(mean)'] = dat['Accident_Date(mean)'].fillna(2017)

dat = pd.merge(dat, a12, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a13, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a14, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a15, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a16, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a17, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a18, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a19, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a20, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a21, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a22, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat, a23, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat,dat3, on=['Policy_Number'], how='left',sort=False)


dat['iage'] = dat['Accident_Date(mean)'] - dat['ibirth']
dat['dage'] = dat['Accident_Date(mean)'] - dat['dbirth']
dat.Cancellation = dat.Cancellation.astype('int')

dat = dat.rename(columns={ ('Coverage_Deductible_if_applied', 0):'Coverage_Deductible_if_applied_0',
                     ('Coverage_Deductible_if_applied', 1):'Coverage_Deductible_if_applied_1',
                     ('Coverage_Deductible_if_applied', 2):'Coverage_Deductible_if_applied_2',
                     ('Insured_Amount1', 0):'Insured_Amount1_0',('Insured_Amount1', 1):'Insured_Amount1_1',
                     ('Insured_Amount1', 2):'Insured_Amount1_2',('Insured_Amount2', 0):'Insured_Amount2_0',
                     ('Insured_Amount2', 1):'Insured_Amount2_1', ('Insured_Amount2', 2):'Insured_Amount2_2',
                     ('Insured_Amount3', 0):'Insured_Amount3_0',
                     ('Insured_Amount3', 1):'Insured_Amount3_1', ('Insured_Amount3', 2):'Insured_Amount3_2'  })

######## Modeling ##################

train = dat[dat['Next_Premium'].isna()==False]
test = dat[dat['Next_Premium'].isna()==True]
y = train['Next_Premium']
features = [f for f in train.columns if f not in ['Next_Premium','Policy_Number']]

data = train[features]
pred = test[features]

# Split train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, 
                                                      shuffle=True,random_state=42)

# 標準化

from sklearn import preprocessing 

sdata = preprocessing.scale( dat[features] )

strain = sdata[dat['Next_Premium'].isna()==False]
stest = sdata[dat['Next_Premium'].isna()==True]

sy = dat['Next_Premium'][dat['Next_Premium'].isna()==False] - \
        dat['Next_Premium'][dat['Next_Premium'].isna()==False].mean()








###################################################################################################
###################################################################################################
## 模型測試區(CV)
# 1.XGboost
from sklearn import cross_validation
from xgboost import XGBRegressor

dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(valid_x,label=valid_y)

dtrain1 = xgb.DMatrix(data,label=y)


gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in [6,7,8,9]
    for min_child_weight  in [0,1,2,3,4,5,6]
]

params = {
    # Parameters that we are going to tune.
    'max_depth':9,
    'min_child_weight': 1,
    'eta':0.015,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear',
}


min_mae = float("Inf")
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth,min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Run CV
    cv_results = xgb.cv(params,dtrain1, num_boost_round=999,
                        seed=42,nfold=5,metrics={'mae'},early_stopping_rounds=10 )

    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))





##########################

model_1 = XGBRegressor(n_estimators=500, learning_rate=0.015, gamma=0, subsample=1,
         colsample_bytree=1, max_depth=9,eval_metric='mae')

model_1.fit(data,y)




##########################################################################
##########################################################################

# 2.LightGBM

train_data2 = lgb.Dataset(train_x,label=train_y)
valid_data2 = lgb.Dataset(valid_x,label=valid_y)

param_={
'boosting_type': 'gbdt',
'class_weight': None,
'colsample_bytree': 1,
'learning_rate': 0.00764107,
'max_depth': -1,
'min_child_samples': 460,
'min_child_weight': 0.001,
'min_split_gain': 0.0,
'n_estimators': 2673,
'n_jobs': -1,
'num_leaves': 77,
'objective': None,
'random_state': 42,
'reg_alpha': 0.877551,
'reg_lambda': 0.204082,
'silent': True,
'subsample': 1,
'subsample_for_bin': 240000,
'subsample_freq': 1,
'metric': 'l1' # aliase for mae 
}


# Train model on selected parameters and number of iterations
lgbm = lgb.train(param_,train_data2,2500,valid_sets=valid_data2,
                 early_stopping_rounds= 40,verbose_eval= 500)

lgbm.predict(pred)













##########################################################################################
###########################################################################################
# 3.Lasso
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn import linear_model

alphas = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,
          3.6,3.8,4,4.5,5,5.5,6,7,8,9,10]

model_lasso = LassoCV(alphas = alphas,cv=5,max_iter=100000)

model_lasso.fit(strain,sy)
best_lambda = model_lasso.alpha_


lasso = linear_model.Lasso()
lasso.set_params(alpha=best_lambda)
lasso.fit(strain, sy)

pred_lasso = lasso.predict( stest ) + dat['Next_Premium'][dat['Next_Premium'].isna()==False].mean()
pred_lasso[pred_lasso<0] = 0
submit_lasso = output( pred_lasso )

submit_lasso
submit_lasso.to_csv("submit_lasso.csv",index=False)

##########################################################################################
###########################################################################################
# 3.elast net

from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression

alphas = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,
          3.6,3.8,4,4.5,5,5.5,6,7,8,9,10]
l1_ratio = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]


regr = ElasticNetCV(alphas=alphas, cv=5, eps=0.001, fit_intercept=True,
       l1_ratio = l1_ratio, max_iter=4000, n_jobs=1,
       normalize=False, positive=False, precompute='auto', random_state=0,
       selection='cyclic', tol=0.0001, verbose=0)

regr.fit(strain, sy)


print(regr.alpha_) 
print(regr.l1_ratio )
winsound.Beep(2500, 500)



##########################################################################################
###########################################################################################
# 4.SVM 












##########################################################################################
###########################################################################################
# 提交區

def output( pred_result ):
    result = pd.DataFrame()
    result['Policy_Number'] = test.Policy_Number.values
    result['Next_Premium'] = pred_result
    submit = pd.DataFrame()
    submit['Policy_Number'] = dat4['Policy_Number']
    submit = submit.merge(result, on=['Policy_Number'])
    return submit
    

# submit.to_csv("submit2.csv",index=False)



winsound.Beep(2500, 500)







