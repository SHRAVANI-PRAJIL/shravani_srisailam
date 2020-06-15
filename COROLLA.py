
Created on THURSDAY May 14 11:12:26 2020

@author: SHRAVANI SRISAILAM
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf # for regression model
Corolla_new = pd.read_csv('C://Users//HP//Desktop//EXCELR//ASSIGNMENTS//MLR//corolla.csv', encoding= 'unicode_escape')
Corolla_new.columns
corolla = Corolla_new[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
corolla.head(5) # to get top 5 rows
type(corolla)
corolla.dtypes
 # Correlation matrix 
corolla.corr()

# we see there is no collinearity between input variables
 
# Scatter plot between the variables along with histograms
sns.pairplot(corolla.iloc[:,:])

        
# Preparing model                  
mlr2 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data=corolla).fit()
 # regression model

# Getting coefficients of variables               
mlr2.params

# Summary
mlr2.summary() 
# R-squared:   0.864   
# p-values for Doors and cc are more than 0.05 


pred = mlr2.predict(corolla) # Predicted values of price using the model
pred


# Preparing model based only on cc
mlr_cc=smf.ols('Price~cc',data = corolla).fit()  
mlr_cc.summary()
# p-value <0.05 .. It is significant 

# Preparing model based only on DOORS
mlr_dr=smf.ols('Price~Doors',data = corolla).fit()  
mlr_dr.summary()
# p-value <0.05 .. It is significant 

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(mlr2)
# index 80, 221 and 960 are showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

new_corolla = corolla.drop(corolla.index[[80,221,960]],axis=0) # ,inplace=False)
new_corolla



# Preparing model                  
mlr2_new = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data=new_corolla).fit()


# Getting coefficients of variables        
mlr2_new.params

# Summary
mlr2_new.summary() 
# R-squared:   0.885   
# p-values also < 0.05 , hence no multicolinearrity exists after reemooving the three rows 


# Confidence values 99%
print(mlr2_new.conf_int(0.05)) # 99% confidence level


# Predicted values of MPG 
price_pred = mlr2_new.predict(new_corolla)
price_pred

# calculating VIF's values of independent variables
rsq_age = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=new_corolla).fit().rsquared  
vif_age = 1/(1-rsq_age) # 2.007

rsq_km = smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=new_corolla).fit().rsquared  
vif_km = 1/(1-rsq_km) # 1.914

rsq_hp = smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=new_corolla).fit().rsquared  
vif_hp = 1/(1-rsq_hp) # 1.599

rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=new_corolla).fit().rsquared  
vif_cc = 1/(1-rsq_cc) # 3.022

rsq_drs = smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=new_corolla).fit().rsquared  
vif_drs = 1/(1-rsq_drs) # 1.2039

rsq_gr= smf.ols('Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight',data=new_corolla).fit().rsquared  
vif_gr= 1/(1-rsq_gr) #  1.1015

rsq_qt = smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight',data=new_corolla).fit().rsquared  
vif_qt = 1/(1-rsq_qt) #  2.997

rsq_wt = smf.ols('Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=new_corolla).fit().rsquared  
vif_wt = 1/(1-rsq_wt) #  3.846

new_corolla.head()
new_corolla.columns


           # Storing vif values in a data frame
d1 = {'Variables':['Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears','Quarterly_Tax', 'Weight'],'VIF':[vif_age,vif_km,vif_hp,vif_cc,vif_drs, vif_gr,vif_qt, vif_wt]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# VIF VALUES ARE LOW FOR ALL SO WE DONT NEED TO EXCLUDE ANYTHING

# Added varible plot 
sm.graphics.plot_partregress_grid(mlr2_new)

