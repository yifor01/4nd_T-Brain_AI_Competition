# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 08:02:10 2018

@author: yifor
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_absolute_error
import winsound
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing 
import pickle


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


# 退保項目 保費歸0
# dat1.Premium[dat1[dat1.Coverage_Deductible_if_applied<0].index] = 0

# 與自負額無關 歸0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="09I"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="10A"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="14E"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="15F"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="15O"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="20B"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="20K"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="29K"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="32N"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="33F"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="33O"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="56K"].index]=0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="65K"].index]=0

# 特殊註記
dat1[(dat1.Insurance_Coverage=="00I") & (dat1.Coverage_Deductible_if_applied==-3)]['Coverage_Deductible_if_applied'] = -7500
dat1[(dat1.Insurance_Coverage=="00I") & (dat1.Coverage_Deductible_if_applied==-1)]['Coverage_Deductible_if_applied'] = -5000
dat1[(dat1.Insurance_Coverage=="00I") & (dat1.Coverage_Deductible_if_applied==1)]['Coverage_Deductible_if_applied'] = 5000
dat1[(dat1.Insurance_Coverage=="00I") & (dat1.Coverage_Deductible_if_applied==2)]['Coverage_Deductible_if_applied'] = 6500
dat1[(dat1.Insurance_Coverage=="00I") & (dat1.Coverage_Deductible_if_applied==3)]['Coverage_Deductible_if_applied'] = 7500

dat1[(dat1.Insurance_Coverage=="02K") & (dat1.Coverage_Deductible_if_applied==-2)]['Coverage_Deductible_if_applied'] = -6500
dat1[(dat1.Insurance_Coverage=="02K") & (dat1.Coverage_Deductible_if_applied==-1)]['Coverage_Deductible_if_applied'] = -5000
dat1[(dat1.Insurance_Coverage=="02K") & (dat1.Coverage_Deductible_if_applied==1)]['Coverage_Deductible_if_applied'] = 5000
dat1[(dat1.Insurance_Coverage=="02K") & (dat1.Coverage_Deductible_if_applied==2)]['Coverage_Deductible_if_applied'] = 6500
dat1[(dat1.Insurance_Coverage=="02K") & (dat1.Coverage_Deductible_if_applied==3)]['Coverage_Deductible_if_applied'] = 7500

## 自負額比例
q1 = dat1[['Policy_Number','Insurance_Coverage','Coverage_Deductible_if_applied']][dat1.Insurance_Coverage=="05N"]
qq1 = q1.groupby(by ='Policy_Number',axis=0,sort=False).Coverage_Deductible_if_applied.value_counts().\
                    reset_index(name='05N_counts')
qq1 = qq1.pivot_table(index='Policy_Number', columns='Coverage_Deductible_if_applied',\
                     values='05N_counts',fill_value=0)

qq1.columns = ['05N_counts__20','05N_counts__10','05N_counts__0','05N_counts_10','05N_counts_20']

q2 = dat1[['Policy_Number','Insurance_Coverage','Coverage_Deductible_if_applied']][dat1.Insurance_Coverage=="14N"]
qq2 = q2.groupby(by ='Policy_Number',axis=0,sort=False).Coverage_Deductible_if_applied.value_counts().\
                    reset_index(name='14N_counts')
qq2 = qq2.pivot_table(index='Policy_Number', columns='Coverage_Deductible_if_applied',\
                     values='14N_counts',fill_value=0)
qq2.columns = ['14N_counts__10','14N_counts__0','14N_counts_10','14N_counts_20']

q3 = dat1[['Policy_Number','Insurance_Coverage','Coverage_Deductible_if_applied']][dat1.Insurance_Coverage=="68E"]
qq3 = q3.groupby(by ='Policy_Number',axis=0,sort=False).Coverage_Deductible_if_applied.value_counts().\
                    reset_index(name='68E_counts')
qq3 = qq3.pivot_table(index='Policy_Number', columns='Coverage_Deductible_if_applied',\
                     values='68E_counts',fill_value=0)
qq3.columns = ['68E_counts_10']

## 自負額比例類 原變數歸0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="05N"].index] = 0
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="14N"].index] = 0 
dat1.Coverage_Deductible_if_applied[dat1[dat1.Insurance_Coverage=="68E"].index] = 0

dat1 = pd.merge(dat1, qq1, on=['Policy_Number'], how='outer',sort=False)
dat1 = pd.merge(dat1, qq2, on=['Policy_Number'], how='outer',sort=False)
dat1 = pd.merge(dat1, qq3, on=['Policy_Number'], how='outer',sort=False)


dat1[qq1.columns] = dat1[qq1.columns].fillna(0.0)
dat1[qq2.columns] = dat1[qq2.columns].fillna(0.0)
dat1[qq3.columns] = dat1[qq3.columns].fillna(0.0)

# 被退險種次數
q7 = dat1[['Policy_Number','Coverage_Deductible_if_applied']][dat1.Coverage_Deductible_if_applied<0].copy()

dat1['reject_count'] = dat1.Policy_Number.map(q7.Policy_Number.value_counts())
dat1['reject_count'] = dat1['reject_count'].fillna(0.0)

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

a1.columns = ['MI_0','MI_1','MI_2']
         
a2.columns = ['CD0_IA0','CD0_IA1','CD0_IA2','CD1_IA0','CD1_IA1','CD1_IA2',
              'CD2_IA0','CD2_IA1','CD2_IA2','CD3_IA0','CD3_IA1','CD3_IA2' ]

dat1 = pd.merge(dat1, a1, on=['Policy_Number'], how='outer',sort=False)
dat1 = pd.merge(dat1, a2, on=['Policy_Number'], how='outer',sort=False)

def unique_check(variable):
   return len(variable.unique() )

# 補缺失值
del dat1['Vehicle_identifier']
del dat1['Prior_Policy_Number']
del dat1["Insured's_ID"]
del dat1["Vehicle_Make_and_Model1"]
del dat1["Vehicle_Make_and_Model2"]
del dat1['Distribution_Channel']
del dat1["Coding_of_Vehicle_Branding_&_Type"]
del dat1['fpt']
del dat1['aassured_zip']

a01 = pd.get_dummies(dat1.iply_area)
a01_name =[]
for i in range(0,22): a01_name.append( 'iply_area_'+str(i) )
a01.columns = a01_name

dat1 = pd.concat([dat1,a01] ,axis=1, join_axes=[dat1.index])

del dat1['iply_area']

dat1.ibirth = dat1.ibirth.astype('float')
dat1.dbirth = dat1.dbirth.astype('float')
dat1.dbirth[dat1.dbirth>2018] = 2017

dat1['ibirth'] =  ( 2017-dat1['ibirth'] )
dat1['dbirth'] =  ( 2017-dat1['dbirth'] )


kk1 = dat1['ibirth'].copy()
kk1[kk1<20] = 1
kk1[(kk1>=20) & (kk1<25)] = 2
kk1[(kk1>=25) & (kk1<30)] = 3
kk1[(kk1>=30) & (kk1<60)] = 4
kk1[(kk1>=60) & (kk1<70)] = 5
kk1[kk1>=70] = 6
dat1['iage_leabel'] = kk1


dat1['carage'] =  ( 2017-dat1['Manafactured_Year_and_Month'] )
rr1 = dat1['carage'].copy()
rr1[rr1>4] = 5
dat1['carage_leabel'] = rr1

dat1['mix0'] = dat1['iage_leabel'] * dat1['carage_leabel'] 
dat1['mix1'] = dat1['iage_leabel'] * dat1['Premium']
dat1['mix2'] = dat1['carage_leabel'] * dat1['Premium']
dat1['mix3'] = dat1['carage_leabel'] * dat1['iage_leabel'] * dat1['Premium']


# 處理 dat1 不唯一資料
dat = dat1[list(dat1.columns[0:6]) +list(dat1.columns[13:84] ) ].drop_duplicates().reset_index(drop=True)
dat['Premium'] = dat1.groupby('Policy_Number',sort=False)['Premium'].sum().values



a24 = dat1.groupby('Policy_Number',sort=False)["mix1"].sum().reset_index(name="mix1(sum)")
a25 = dat1.groupby('Policy_Number',sort=False)["mix2"].sum().reset_index(name="mix2(sum)")
a26 = dat1.groupby('Policy_Number',sort=False)["mix3"].sum().reset_index(name="mix3(sum)")

dat = pd.merge(dat,a24, on=['Policy_Number'], how='left',sort=False)
dat = pd.merge(dat,a25, on=['Policy_Number'], how='left',sort=False)
dat = pd.merge(dat,a26, on=['Policy_Number'], how='left',sort=False)


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

del dat2['Vehicle_identifier'] 


dat2.Accident_Date = (dat2.Accident_Date.str[0:4].astype('float'))+\
                        ( dat2.Accident_Date.str[5:7].astype('float')-1)/12

# Coverage_count
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


df2 = dat2[['Policy_Number','Claim_Number',"Driver's_Gender",'DOB_of_Driver',
           'Marital_Status_of_Driver', 'Accident_Date', "Accident_area","Cause_of_Loss",
           'number_of_claimants', 'Accident_Time',"At_Fault?"]].drop_duplicates().reset_index(drop=True)



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

a99 = dat2[['Policy_Number','acc_count']].drop_duplicates().reset_index(drop=True)

dat = pd.merge(dat, a99, on=['Policy_Number'], how='left',sort=False).fillna(0)
dat = pd.merge(dat,dat3, on=['Policy_Number'], how='left',sort=False)



dat.Cancellation = dat.Cancellation.astype('int')

dat = dat.rename(columns={ ('Coverage_Deductible_if_applied', 0):'Coverage_Deductible_if_applied_0',
                     ('Coverage_Deductible_if_applied', 1):'Coverage_Deductible_if_applied_1',
                     ('Coverage_Deductible_if_applied', 2):'Coverage_Deductible_if_applied_2',
                     ('Insured_Amount1', 0):'Insured_Amount1_0',('Insured_Amount1', 1):'Insured_Amount1_1',
                     ('Insured_Amount1', 2):'Insured_Amount1_2',('Insured_Amount2', 0):'Insured_Amount2_0',
                     ('Insured_Amount2', 1):'Insured_Amount2_1', ('Insured_Amount2', 2):'Insured_Amount2_2',
                     ('Insured_Amount3', 0):'Insured_Amount3_0',
                     ('Insured_Amount3', 1):'Insured_Amount3_1', ('Insured_Amount3', 2):'Insured_Amount3_2'  })


a77 = pd.get_dummies(dat.Imported_or_Domestic_Car)

a77.columns = ["IoDC_10","IoDC_20","IoDC_21","IoDC_22","IoDC_23","IoDC_24","IoDC_30","IoDC_40","IoDC_50","IoDC_90"]

dat = pd.concat([dat,a77] ,axis=1, join_axes=[dat.index])




features = [f for f in dat.columns if f not in ['Next_Premium','Policy_Number']]

######## Modeling ##################
# 分類用模型 (全部資料) + 標準化
class_all = preprocessing.scale( dat[features] )


class_data = class_all[dat['Next_Premium'].isna()==False]
class_pred = class_all[dat['Next_Premium'].isna()==True]

class_y = dat.Next_Premium.copy()[dat['Next_Premium'].isna()==False]
class_y[class_y>0] = 1


# 保費模型 (要扣退保)
conti_all = dat[features][dat.Next_Premium!=0]

data = conti_all[dat['Next_Premium'].isna()==False]
pred = conti_all[dat['Next_Premium'].isna()==True]
y = dat['Next_Premium'][dat.Next_Premium>0]




# 保費標準化(SVM, Lasso)
conti_all_1 = dat[dat.Next_Premium!=0].reset_index(drop=True)
s_conti_all = preprocessing.scale( conti_all_1[features] )
sdata = s_conti_all[conti_all_1.Next_Premium.notna()]
spred = s_conti_all[conti_all_1.Next_Premium.isna()]

sy = conti_all_1.Next_Premium[conti_all_1.Next_Premium.notna()] - \
     conti_all_1.Next_Premium[conti_all_1.Next_Premium.notna()].mean()




# Split train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True,random_state=42)


##################################################################################
# 存變數
file1 = open('class_data.pickle', 'wb')
pickle.dump(class_data, file1)
file1.close()

file2 = open('class_y.pickle', 'wb')
pickle.dump(class_y, file2)
file2.close()

file3 = open('class_pred.pickle', 'wb')
pickle.dump(class_pred, file3)
file3.close()
##############################
file4 = open('data.pickle', 'wb')
pickle.dump(data, file4)
file4.close()

file5 = open('pred.pickle', 'wb')
pickle.dump(pred, file5)
file5.close()

file6 = open('y.pickle', 'wb')
pickle.dump(y, file6)
file6.close()
##############################
file7 = open('sdata.pickle', 'wb')
pickle.dump(sdata, file7)
file7.close()

file8 = open('spred.pickle', 'wb')
pickle.dump(spred, file8)
file8.close()

file9 = open('sy.pickle', 'wb')
pickle.dump(sy, file9)
file9.close()



import pickle
####################
# 提取變數
with open('class_data.pickle', 'rb') as file1:
    class_data = pickle.load(file1)

with open('class_y.pickle', 'rb') as file2:
    class_y = pickle.load(file2)

with open('class_pred.pickle', 'rb') as file3:
    class_pred = pickle.load(file3)
####################
with open('data.pickle', 'rb') as file4:
    data = pickle.load(file4)

with open('pred.pickle', 'rb') as file5:
    pred = pickle.load(file5)

with open('y.pickle', 'rb') as file6:
    y = pickle.load(file6)
####################
with open('sdata.pickle', 'rb') as file7:
    sdata = pickle.load(file7)

with open('spred.pickle', 'rb') as file8:
    spred = pickle.load(file8)

with open('sy.pickle', 'rb') as file9:
    sy = pickle.load(file9)




################################################################################



def output( pred_result ):
    result = pd.DataFrame()
    result['Policy_Number'] = dat.Policy_Number[dat['Next_Premium'].isna()==True].values
    result['Next_Premium'] = pred_result
    submit = pd.DataFrame()
    submit['Policy_Number'] = dat4['Policy_Number']
    submit = submit.merge(result, on=['Policy_Number'] ,how='outer' )
    submit = submit.fillna(0)
    return submit


#qq = pd.read_csv('submitLGMB_7_31.csv',encoding='UTF-8')

#qq.index()

#xxx_qq = list(set(qq.Policy_Number) & set(xxx))

#bbb = []
#for i in range(0, len(xxx_qq) ):
#    bbb.append( list(qq.Policy_Number).index(xxx_qq[i])   )

#qq.iloc[bbb,1] = 0
#
#qq.to_csv("qq.csv",index=False)

