# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 01:37:37 2020

@author: DELL
"""


################# IMDB reviews extraction ######################## Time Taking process as this program is going
# to operate the web page while extracting reviews 
############# time library in order to sleep and make it to extract for that specific page 
#### We need to install selenium for python
#### pip install selenium
#### time library to make the extraction process sleep for few seconds 

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

browser = webdriver.Chrome(ChromeDriverManager().install())

from bs4 import BeautifulSoup as bs
page="https://www.imdb.com/title/tt1979388/reviews?ref_=tt_urv"   
####             The Good Dinosaur (2015)  Movie         ####

# Importing few exceptions to surpass the error messages while extracting reviews 
from selenium.common.exceptions import NoSuchElementException 
from selenium.common.exceptions import ElementNotVisibleException

browser.get(page)
import time
reviews = []
i=1
# Below while loop is to load all the reviews into the browser till load more button dissapears
while (i>0):
    #i=i+25
    try:
        # Storing the load more button page xpath which we will be using it for click it through selenium 
        # for loading few more reviews
        button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]') # //*[@id="load-more-trigger"]
        button.click()
        time.sleep(3)
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break

# Getting the page source for the entire imdb after loading all the reviews
ps = browser.page_source 
#Converting page source into Beautiful soup object
soup=bs(ps,"html.parser")

#Extracting the reviews present in div html_tag having class containing "text" in its value
reviews = soup.findAll("div",attrs={"class","text"})
for i in range(len(reviews)):
    reviews[i] = reviews[i].text

##### If we want only few recent reviews you can either press ctrl+c to break the operation in middle but the it will store 
##### Whatever data it has extracted so far #######

# Creating a data frame 
import pandas as pd
dinosaur_reviews = pd.DataFrame(columns = ["reviews"])
dinosaur_reviews["reviews"] = reviews

dinosaur_reviews.to_csv("dinosaur_reviews.csv",encoding="utf-8")

#_____________________________________________________________________________________________________________________________
#               import files required for analysis
#_____________________________________________________________________________________________________________________________

import pandas as pd
import os
from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
#import plotly.plotly as py
import operator
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

############################## Loading Data #########################################
# importing dataset
#IMDB = pd.read_csv("C:\\Users\\DELL\\dinosaur_reviews.csv")

df_master = pd.read_csv("C:\\Users\\DELL\\dinosaur_reviews.csv", encoding='latin-1', index_col = 0)

# Start with one review:
text = df_master.reviews[0]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# Save the image in the img folder:
wordcloud.to_file("C:\\Users\\DELL\\first_review.png")
text = " ".join(review for review in df_master.reviews)
print ("There are {} words in the combination of all review.".format(len(text)))

# There are 675496 words in the combination of all review.
#stop words
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
stoplist = set(stopwords.words("english"))
for w in tokenized_sent:
    if w not in stoplist:
        pass # Do something.

# positive words # Choose the path for +ve words stored in system
with open("D:\\ExcelR\\ASSIGNMENTS\\pending\\text mining\\positive-words.txt","r") as pos:poswords = pos.read().split("\n")
  
poswords = poswords[36:]


# negative words  Choose path for -ve words stored in system
with open("D:\\ExcelR\\ASSIGNMENTS\\pending\\text mining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()