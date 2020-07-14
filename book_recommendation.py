# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:49:26 2020

@author: SHRAVANI PRAJIL
"""
import pandas as pd
import numpy as np
import Recommender
#import Dataset 
book = pd.read_csv("C:\\Users\\SHRAVANI PRAJIL\\Desktop\\EXCELR\\ASSIGNMENTS\\Recommendation systems\\book.csv", encoding='latin-1')
book.shape #shape
book.columns
book['Book.Rating'].isnull().sum() 
##  no null values
book.describe()
#######                  average rating = 7.5                                    #######

#Unnamed: 0        User.ID  Book.Rating
#count  10000.00000   10000.000000  10000.00000
#mean    5000.50000   95321.249800      7.56630
#std     2886.89568  117645.703609      1.82152
#min        1.00000       8.000000      1.00000
#25%     2500.75000    2103.000000      7.00000
#50%     5000.50000    3757.000000      8.00000
#75%     7500.25000  162052.000000      9.00000
#max    10000.00000  278854.000000     10.00000

ratings = pd.DataFrame(book.groupby('Book.Title')['Book.Rating'].mean())
ratings.head()
ratings['number_of_ratings'] = book.groupby('Book.Title')['Book.Rating'].count()
ratings.head()
ratings.describe()
#       Book.Rating  number_of_ratings
#count  9659.000000        9659.000000
#mean      7.570961           1.035304
#std       1.806169           0.208278
#min       1.000000           1.000000
#25%       7.000000           1.000000
#50%       8.000000           1.000000
#75%       9.000000           1.000000
#max      10.000000           5.000000

import matplotlib.pyplot as plt
%matplotlib inline
# plot graph of 'ratings' column 
plt.figure(figsize =(10, 4)) 
ratings['Book.Rating'].hist(bins=50)
###         most of the book are rated between 7 and 8

# plot graph of 'number_of_ratings' column 
plt.figure(figsize =(10, 4)) 
ratings['number_of_ratings'].hist(bins=60)
#       and most of the books have one 1 and 2 number of ratings

import seaborn as sns
sns.jointplot(x='Book.Rating', y='number_of_ratings', data=ratings)
#       maximum number of ratings are 5 for the book Fahrenheit 451

book_matrix = book.pivot_table(index='User.ID', columns='Book.Title', values='Book.Rating')
book_matrix.head()
ratings.sort_values('number_of_ratings', ascending=False).head(10)
from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(ratings, test_size = 0.20, random_state=0)
   
Fahrenheit_user_rating = book_matrix['Fahrenheit 451']
Stardust_user_rating = book_matrix['Stardust']
Fahrenheit_user_rating.describe()
Fahrenheit_user_rating.head()
Stardust_user_rating.head()
Stardust_user_rating.describe()
similar_to_Fahrenheit=book_matrix.corrwith(Fahrenheit_user_rating)
similar_to_Fahrenheit
similar_to_Fahrenheit.head()
similar_to_Stardust=book_matrix.corrwith(Stardust_user_rating)
similar_to_Stardust
similar_to_Stardust.head()

corr_Stardust = pd.DataFrame(similar_to_Stardust, columns=['Correlation'])
corr_Stardust.dropna(inplace=True)
corr_Stardust.head()

corr_Fahrenheit = pd.DataFrame(similar_to_Fahrenheit, columns=['Correlation'])
corr_Fahrenheit.dropna(inplace=True)
corr_Fahrenheit.head()
corr_Fahrenheit = corr_Fahrenheit.join(ratings['number_of_ratings'])
corr_Stardust = corr_Stardust.join(ratings['number_of_ratings'])
corr_Stardust.head()
corr_Fahrenheit.head()

corr_Fahrenheit[corr_Fahrenheit['number_of_ratings'] > 3].sort_values(by='Correlation', ascending=False).head(10)
corr_Stardust[corr_Stardust['number_of_ratings'] > 3].sort_values(by='Correlation', ascending=False).head(10)


