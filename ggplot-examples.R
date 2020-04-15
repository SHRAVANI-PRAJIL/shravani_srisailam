data("mtcars")
attach(mtcars)
library(ggplot2)

#scatter plot between hp and cyl and best fitting line for the data
g1 <- ggplot(mtcars, aes(x=hp, y=cyl)) + geom_point(aes(col = cyl)) + geom_smooth(method="lm", col="brown") 
plot(g1)
g1 + ggtitle("hp Vs cyl", subtitle="From mtcars dataset") + xlab("hp of mtcars") + ylab("no. of cylnders of mtcars")

#histogram for mpg of mtcars
g2 = ggplot(mtcars, aes(mpg)) +  geom_histogram(color="darkblue", fill="lightblue") 
plot(g2)

# densty Plot
g3 <- ggplot(mtcars, aes(mpg))
plot(g3)
g3 + geom_density(aes(fill=factor(cyl)), alpha=0.8) + 
  

  labs(title="Density plot", 
       subtitle="mpg Grouped by no. of cylinders",
       caption="Source: mtcars",
       x="mpg",
       fill="# Cylinders")


data("iris")
attach(iris)
View(iris)
#histogram for categorical data
theme_set(theme_classic())
g4 = ggplot(iris, aes(Species))
g4 +  geom_bar(color="darkblue", fill="lightblue")+ theme(axis.text.x = element_text(angle=65, vjust=0.6)) + labs(title="Histogram on Categorical Variable", subtitle="species")
plot(g4)
   
#stacked data bar plot
theme_set(theme_classic())
g5 = ggplot(iris, aes(Petal.Width)) + scale_fill_brewer(palette = "Spectral")
plot(g5)
g5 +  geom_bar(aes(fill=Species)) + labs(title="Histogram with stacked data", subtitle="petal width")

#box plot
g6 <- ggplot(iris, aes(Species, Sepal.Length))
plot(g6)
g6 + geom_boxplot(varwidth=T, fill="orange") + 
  labs(title="Box plot", 
       subtitle="species vs sepal length",
       caption="Source: iris",
       x="Species",
       y="sepal length")

         

pnorm(740, 711, 29) - pnorm(697, 711, 29)                                                  
