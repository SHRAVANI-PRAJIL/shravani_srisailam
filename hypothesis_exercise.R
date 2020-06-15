#example 1
#The diameter of the cutlet between two units A & B
setwd("C:\\Users\\HP\\Desktop\\EXCELR\\ASSIGNMENTS\\hypothesisassignment")
library(readxl)
library(openxlsx)
library(readr)

getwd()
cutlet <-read.xlsx("C:\\Users\\HP\\Desktop\\EXCELR\\ASSIGNMENTS\\hypothesisassignment\\Cutlets.xlsx")
cutlet <-read_csv(file.choose())
View(cutlet)
attach(cutlet)
colnames(cutlet)
#############Normality test###############
shapiro.test(`Unit A`)# Normality test
# p-value = 0.32 >0.05 so p high null fly => It follows normal distribution

shapiro.test(`Unit B`)
# p-value = 0.5225 >0.05 so p high null fly => It follows normal distribution

#############variance test###############
var.test(`Unit A`, `Unit B`)
# p-value = 0.3136 > 0.05 so p high null fly => Equal variances

############2 sample T Test ##################

t.test(`Unit A`, `Unit B`,alternative = "two.sided",conf.level = 0.95,correct = TRUE)#two sample T.Test
# alternative = "two.sided" means we are checking for equal and unequal
# null Hypothesis -> Equal means
# Alternate Hypothesis -> Unequal Hypothesis
# p-value = 0.4723 < 0.05 accept alternate Hypothesis 
# the diameter of the cutlet between two unitsis same accept Null hypothesis



#example 2
#the average Turn Around Time (TAT) of 4 laboratories
#########ANOVA TEST LAB TAT#################

#TAT<-read_excel("LabTAT.xlsx") # Bahaman.xlsx
TAT<-read_csv(file.choose())
View(TAT)
attach(TAT)
Stacked_TAT <- stack(TAT)
View(Stacked_TAT)
attach(Stacked_TAT)
###var.test(values,ind)
Anova_TAT <- aov(values~ind,data = Stacked_TAT)
summary(Anova_TAT)
# p-value = 2e-16 <  0.05 accept alternative hypothesis 
#there is significant difference in average TAT among the different laboratories


#example 3
#Sales of products in four different regions 
###########proportions test ###############
#f.m.buyers<-read_excel("BuyerRatio.xlsx")   
f.m.buyers<-read_csv(file.choose())
View(f.m.buyers) 
attach(f.m.buyers)
m.f.ratio <- data.frame(East, West, North, South)
View(m.f.ratio) 
chisq.test(m.f.ratio)
# p-value = 0.6603 > 0.05 accept NULL hypothesis 
# MALE - FEMALE RATIOS ARE SAME ACROSS ALL THE REGIONS


#example 4
# % defective of 4centres
#########proportion Test#################

per_defect<-read_csv(file.choose()) # customer order
View(per_defect)
attach(per_defect)
Stacked_defect <- stack(per_defect)
View(Stacked_defect)
attach(Stacked_defect)
table(ind,values)
chisq.test(table(ind,values))
# p-value = 0.2771 > 0.05  => Accept null hypothesis
# there is no significant differreencee for 4 centrees based on p-value
# manager need not take any action


####question5#####

library(readr)
faltoons <- read_csv(file.choose())
View(faltoons)
attach(faltoons)
table(Weekdays)
table(Weekend)
prop.test(x=c(113, 167),n=c(287, 233), conf.level = 0.95, alternative = "two.sided")
# p-value = 3.892e-13 < 0.05 reject NULL hypothesis 
# there is a siignificant difference in the % of males vs females walking into 
#the store based on day of the week 