# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 19:00:35 2018

@author: yifor
"""
from scipy import stats
 import statsmodels.api as sm


# train predict
train_lasso   
xg_train
LGBM_train

MX = pd.DataFrame( {"lasso":train_lasso, "xg":xg_train   ,"lgbm":LGBM_train })

MX = sm.add_constant( MX )
MX = np.array(MX)

mix_model = sm.OLS(y,MX).fit()
mix_model.summary()





mean_absolute_error(train_lasso,y)
mean_absolute_error(xg_train,y)
mean_absolute_error(LGBM_train,y)


#######################
# test predict
pred_lasso
xg_pred
LGBM_pred




MP = pd.DataFrame( {"lasso":pred_lasso, "xg":xg_pred   ,"lgbm":LGBM_pred })
MP = sm.add_constant( MP )
MP = np.array(MP)

FF = mix_model.predict(MP)
FF[FF<0] = 0
FF

submit_mixed = output(FF)
submit_mixed.to_csv("submit_mixed_8_6.csv",index=False)







LGBM_pred[]



xxx_test = list(set(submit_mixed.Policy_Number) & set(xxx))

b1 = []

for i in range(0, len(xxx_test) ):
    b1.append( list(submit_mixed.Policy_Number).index(xxx_test[ i ])   )


b1

submit_mixed[submit_mixed.Policy_Number=="5d379d819e07453da90823fc9e0b3d4c5af4c26e"]
submit_mixed[b1[0]]


submit_mixed.iloc[b1[0],1] = 0


