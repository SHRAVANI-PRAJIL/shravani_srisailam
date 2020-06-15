
Created on THURSDAY May 14 2020

@author: SHRAVANI SRISAILAM
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf # for regression model
data = pd.read_csv('C://Users//HP//Desktop//EXCELR//ASSIGNMENTS//MLR//Computer_Data.csv', encoding= 'unicode_escape')
data.columns
data.head(5) # to get top 5 rows
type(data)
data.dtypes
data['cd'].unique() 
data['multi'].unique() 
data['premium'].unique() 

# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'cd', 'multi', 'premium'. 
data['cd']= label_encoder.fit_transform(data['cd']) 
data['cd'].unique() 
data['multi']= label_encoder.fit_transform(data['multi']) 
data['multi'].unique() 
data['premium']= label_encoder.fit_transform(data['premium']) 
data['premium'].unique() 


 # Correlation matrix 
data.corr()

# we see there is no collinearity between input variables
 
# Scatter plot between the variables along with histograms
#sns.pairplot(data.iloc[:,:])

# Preparing model  
import statsmodels.formula.api as smf # for regression model            
mlr = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend', data=data).fit()
 # regression model

# Getting coefficients of variables               
mlr.params

# Summary
mlr.summary() 
# R-squared:   0.776
# p-values < 0.05 for all variables  


pred = mlr.predict(data) # Predicted values of price using the model
pred

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm

#sm.graphics.influence_plot(mlr)

# index 1700, 1440 and 5960 are showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

#new_data = data.drop(data.index[[1700,1440,5960,1101, 900, 4477, 3783]],axis=0) # ,inplace=False)
#new_data



# Preparing model                  
mlr_new = smf.ols('price~speed+log(data.hd)+ram+screen+cd+multi+premium+ads+trend', data=data).fit()

#sm.graphics.influence_plot(mlr_new)
 
# Getting coefficients of variables        
mlr_new.params
mlr_new.summary() 

# Confidence values 99%
print(mlr_new.conf_int(0.05)) # 99% confidence level

# Predicted values of price
price_pred = mlr_new.predict(data)
price_pred

plt.boxplot(data.hd)
#data.hd.describe()
### too many outliers in hd variable
#q1 = data.hd.quantile(0.25)
#q3 = data.hd.quantile(0.75)
#iqr = q3-q1 #Interquartile range
#fence_low  = q1-1.5*iqr
#fence_high = q3+1.5*iqr
#data1 = data[(data.hd < fence_low)|(data.hd > fence_high)]
#data1 

#outliers > 400 ,  too many outliers so cant delete data.. hencee applying log transformation on hd variable
data['log_hd'] = np.log(data['hd'])
data.head(5) # to get top 5 rows
plt.boxplot(data.log_hd)


# calculating VIF's values of independent variables
rsq_sp = smf.ols('speed~hd+ram+screen+cd+multi+premium+ads+trend',data=new_data).fit().rsquared  
vif_sp = 1/(1-rsq_sp) # 1.264
rsq_hd = smf.ols('hd~speed+ram+screen+cd+multi+premium+ads+trend',data=new_data).fit().rsquared  
vif_hd = 1/(1-rsq_hd) # 4.459
rsq_ram = smf.ols('ram~speed+hd+screen+cd+multi+premium+ads+trend',data=new_data).fit().rsquared  
vif_ram = 1/(1-rsq_ram) # 3.0924
rsq_sc = smf.ols('screen~speed+hd+ram+cd+multi+premium+ads+trend',data=new_data).fit().rsquared  
vif_sc = 1/(1-rsq_sc) # 1.0818
rsq_cd = smf.ols('cd~speed+hd+ram+screen+multi+premium+ads+trend',data=new_data).fit().rsquared  
vif_cd = 1/(1-rsq_cd) # 1.8622
rsq_mt = smf.ols('multi~speed+hd+ram+screen+cd+premium+ads+trend',data=new_data).fit().rsquared  
vif_mt = 1/(1-rsq_mt) # 2.007
rsq_pr = smf.ols('premium~speed+hd+ram+screen+cd+multi+ads+trend',data=new_data).fit().rsquared  
vif_pr = 1/(1-rsq_pr) # 1.109
rsq_ads = smf.ols('ads~speed+hd+ram+screen+cd+multi+premium+trend',data=new_data).fit().rsquared  
vif_ads = 1/(1-rsq_ads) # 1.2213
rsq_td = smf.ols('trend~speed+hd+ram+screen+cd+multi+premium+ads',data=new_data).fit().rsquared  
vif_td = 1/(1-rsq_td) # 2.065

new_data.columns


           # Storing vif values in a data frame
d1 = {'Variables':['speed', 'hd', 'ram', 'screen', 'cd', 'multi','premium', 'ads', 'trend'],'VIF':[vif_sp,vif_hd,vif_ram,vif_sc,vif_cd, vif_mt,vif_pr, vif_ads,vif_td]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# VIF VALUES ARE LOW FOR ALL SO WE DONT NEED TO EXCLUDE ANYTHING

# Added varible plot 
sm.graphics.plot_partregress_grid(mlr_new1)

#final model
#outliers > 400 ,  too many outliers so cant delete data.. hencee applying log transformation on hd variable
data['log_hd'] = np.log(data['hd'])
data.head(5) # to get top 5 rows

mlr_new1 = smf.ols('price~speed+log_hd+ram+screen+cd+multi+premium+ads+trend', data=data).fit()
sm.graphics.influence_plot(mlr_new1)

# Summary
mlr_new1.summary() 
# R-squared:   0.796 improved
# Predicted values of price
price_pred = mlr_new.predict(data)
price_pred


