# -*- coding: utf-8 -*-
tf.keras.datasets.imdb.get_word_index(path='imdb_word_index.json'
)

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
import matplotlib.pyplot as plt
#pip install wordcloud
from wordcloud import WordCloud
trimmer_reviews=[]

for i in range(1,20):
  ip=[]  
  #url="https://www.amazon.in/Apple-MacBook-Air-13-3-inch-Integrated/product-reviews/B073Q5R6VR/ref=cm_cr_arp_d_paging_btm_2?showViewpoints=1&pageNumber="+str(i)
  url = "https://www.amazon.in/Philips-QT4011-cordless-Titanium-Trimmer/product-reviews/B00JJIDBIC/ref=cm_cr_arp_d_viewpnt_lft?ie=UTF8&reviewerType=all_reviews&filterByStar=positive&pageNumber=1"
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
  trimmer_review=trimmer_reviews+ip 
  
# writng reviews in a text file 
with open("trimmer.txt","w",encoding='utf8') as output:
    output.write(str(trimmer_review))
#to know the where the location is stored
import os   
os.getcwd()    

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(trimmer_review)


ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)

ip_reviews_words = ip_rev_string.split(" ")

#stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#import nltk
#nltk.download('stopwords')
stopwordss=[w for w in ip_reviews_words if not w in stop_words]
    
#from nltk.tokenize import word_tokenize
#wordtokeniser=word_tokenize
#wordtoken=[]
#for w in stopwordss:
#    wordtoken.append(wordtokeniser(w))    
 
    
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemm=[]
for w in stopwordss:
    stemm.append(ps.stem(w))
    
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(stemm)


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

#creating positive words
with open("D:\\ExcelR\\class_room_files\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

posword=poswords[0:]

with open("D:\\ExcelR\\class_room_files\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[1:]

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
#pip install textblob
from textblob import TextBlob
text = open("trimmer.txt",encoding='cp850')
text=text.read()



ip_pos_in_pos
blob=TextBlob(ip_neg_in_neg)
print(blob.sentiment)

#storing negative polarity and subjectivity
#python -m textblob.download_corpora
negblob=blob.sentiment
t=(blob.polarity)
for s in blob.sentences:
    if s.polarity < -0.2:
        print(s)
        
    
print(blob.sentences)
import requests

blob.polarity    #  -0.44545
blob.sentences

# writng reviews in a text file 
with open("pos.txt","w",encoding='utf8') as output:
    output.write(str(ip_pos_in_pos))

import os
os.getcwd()

text1 = open("pos.txt")
text1=text1.read()


ip_pos_in_pos
blob2=TextBlob(ip_pos_in_pos)
#here you will get polarity and subjectivity,how positive you get the information 
#and subjectivity means how factual (or) opinion the data.
print(blob2.sentiment)   #Sentiment(polarity=0.6214015151515152, subjectivity=0.6962121212121214)
blob2.polarity  #  0.6214015151515152







