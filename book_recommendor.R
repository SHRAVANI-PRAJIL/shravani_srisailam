#Installing and loading the libraries
#install.packages("recommenderlab", dependencies=TRUE)
#install.packages("Matrix")
library("recommenderlab")
library(caTools)
#book rating data
book_rate_data <- read.csv("C://Users//SHRAVANI PRAJIL//Desktop//EXCELR//ASSIGNMENTS//Recommendation systems//book.csv")
View(book_rate_data)
class(book_rate_data)
book_rate_data <- book_rate_data[-1]
View(book_rate_data)

#metadata about the variable
str(book_rate_data)
table(book_rate_data$Book.Title)
#rating distribution
hist(book_rate_data$Book.Rating)
#the datatype should be realRatingMatrix inorder to build recommendation engine
book_rate_data_matrix <- as(book_rate_data, 'realRatingMatrix')
#Popularity based 
book_recomm_model1 <- Recommender(book_rate_data_matrix, method="POPULAR")
#Predictions for two users 
recommended_items1 <- predict(book_recomm_model1, book_rate_data_matrix[413:414], n=5)
as(recommended_items1, "list")
####  $`1511`  top five recommendations from the two user
#[1] "In the Beauty of the Lilies" 
# [2]"Black House"  
#[3]"White Oleander : A Novel"   
#[4] "The Magician's Tale"        
#[5]"Nowle's Passing: A Novel"   

#$`1513`
#[1] "In the Beauty of the Lilies"
#[2]"Black House"       
#[3]"White Oleander : A Novel"   
#[4] "The Magician's Tale" 
#[5]"Nowle's Passing: A Novel"   


## Popularity model recommends the same movies for all users , we need to improve our model
#using # # Collaborative Filtering
#User Based Collaborative Filtering
book_recomm_model2 <- Recommender(book_rate_data_matrix, method="UBCF")
#Predictions for two users 
recommended_items2 <- predict(book_recomm_model2, book_rate_data_matrix[413:414], n=5)
as(recommended_items2, "list")
#$`1511`
#[1] "'48"                                                                  
#[2] "'O Au No Keia: Voices from Hawai'I's Mahu and Transgender Communities"
#[3] " Jason, Madison &amp"                                                 
#[4] " Other Stories;Merril;1985;McClelland &amp"                           
#[5] " Repairing PC Drives &amp"                                            

#$`1513`
#[1] "'48"                                                                  
#[2] "'O Au No Keia: Voices from Hawai'I's Mahu and Transgender Communities"
#[3] " Jason, Madison &amp"                                                 
#[4] " Other Stories;Merril;1985;McClelland &amp"                           
#[5] " Repairing PC Drives &amp"                                            
