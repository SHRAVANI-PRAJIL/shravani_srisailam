# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:12:26 2020

@author: SHRAVANI SRISAILAM
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf # for regression model

startups = pd.read_csv("C://Users//HP//Desktop//EXCELR//ASSIGNMENTS//MLR//50_Startups.csv")
startups.columns
startups = startups.rename(columns={'R&D Spend':'RD', 'Administration' : 'ADMIN', 'Marketing Spend' : 'MARKT'})
startups.head(5) # to get top 5 rows
type(startups)
startups.dtypes
startups['State'].unique() 
# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
startups['State']= label_encoder.fit_transform(startups['State']) 
startups['State'].unique() 
# Correlation matrix 
startups.corr()

# we see there is no collinearity between input variables
 
# Scatter plot between the variables along with histograms
sns.pairplot(startups.iloc[:,:])

# preparing model considering all the variables 
         
# Preparing model                  
mlr1 = smf.ols('Profit~RD+ADMIN+MARKT+State', data=startups).fit()
 # regression model

# Getting coefficients of variables               
mlr1.params

# Summary
mlr1.summary() #R-squared:   0.951  ,  
# NO transformation required
# p-values for MARKT, State, and ADMIN are more than 0.05 


pred = mlr1.predict(startups) # Predicted values of profit using the model
pred

# preparing model based only on MARKT

mlr_m= smf.ols('Profit~MARKT',data = startups).fit()  
mlr_m.summary() # 0.0
# p-value <0.05 .. It is significant 

# preparing model based only on ADMIN
mlr_a= smf.ols('Profit~ADMIN',data = startups).fit()  
mlr_a.summary() # 0.162
# p-value >0.05 .. It is INSIGNIFICANT

# preparing model based only on ADMIN
mlr_rd= smf.ols('Profit~RD',data = startups).fit()  
mlr_rd.summary() # 0.162
# p-value >0.05 .. It is INSIGNIFICANT

# Preparing model based only on State
mlr_s= smf.ols('Profit~State',data = startups).fit()  
mlr_s.summary() # 0.482
# p-value >0.05 .. It is insignificant


#BASED ON P-values only market variable is significant and other variables are insignificant... 
# So there may be a chance of considering only one among State and ADMIN

# Checking whether data has any influential values 
# influence index plots


import statsmodels.api as sm
sm.graphics.influence_plot(mlr1)
# index 45,48 & is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

start_new = startups.drop(startups.index[[49,48,45]],axis=0) # ,inplace=False)
start_new



# Preparing model                  
mlr_new = smf.ols('Profit~RD+ADMIN+MARKT+State', data=start_new).fit()


# Getting coefficients of variables        
mlr_new.params

# Summary
mlr_new.summary() # 0.951

# Confidence values 99%
print(mlr_new.conf_int(0.05)) # 99% confidence level


# Predicted values of MPG 
profit_pred = mlr_new.predict(start_new)
profit_pred

# calculating VIF's values of independent variables
rsq_rd = smf.ols('RD~ADMIN+MARKT+State',data=start_new).fit().rsquared  
vif_rd = 1/(1-rsq_rd) # 2.1306

rsq_ad = smf.ols('ADMIN~MARKT+RD+State',data=start_new).fit().rsquared  
vif_ad = 1/(1-rsq_ad) # 1.2021

rsq_mrkt= smf.ols('MARKT~RD+ADMIN+State',data=start_new).fit().rsquared  
vif_mrkt= 1/(1-rsq_mrkt) #  2.1124

rsq_st = smf.ols('State~RD+ADMIN+MARKT',data=start_new).fit().rsquared  
vif_st = 1/(1-rsq_st) #  1.0369

           # Storing vif values in a data frame
d1 = {'Variables':['RD','ADMIN','MARKT', 'State'],'VIF':[vif_rd,vif_ad,vif_mrkt,vif_st]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# VIF VALUES ARE LOW FOR ALL SO WE DONT NEED TO EXCLUDE ANYTHING

# Added varible plot 
sm.graphics.plot_partregress_grid(mlr_new)


