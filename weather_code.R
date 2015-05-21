setwd("~/Walmart-storm")
getwd()
list.files()
# remove files
rm(c.mat,sub.train.data)
rm(c.matrix, cc,cCorpus,out,s.fit,sst,text)
# read the data in
key.data <- read.csv("key.csv",stringsAsFactors=FALSE )
weather.data <- read.csv("weather.csv", stringsAsFactors=FALSE)
train.data <- read.csv("train.csv", stringsAsFactors=FALSE)
# following function picks right weather code from weather data relevant to walmart indices
pick.weather <- function(store,d){
  station.number <- filter(key.data, store_nbr == store)$station_nbr
  weather.code <- filter(weather.data, station_nbr ==station.number & date==d)$codesum
  if (!weather.code==""&!weather.code== " " ){
    return(weather.code)
  }
  else{
    return("")
  }
}


names(train.data)
sub.train.data <- train.data[1:1000,]

sub.train.data$code <- mapply(FUN=pick.weather, sub.train.data$store_nbr,sub.train.data$date)

names(sub.train.data)
library(caret)
library(dplyr)
?createDataPartition
names(ptr.data)
set.seed(14)
# pick a sample of training data 
part.train.data <- createDataPartition(y= train.data$store_nbr, p =0.01, list= FALSE)[,1]
ptr.data <- train.data[part.train.data,]
# pick a cross validation set

set.seed(15)
nptr.data <- train.data[-part.train.data]
cv.index <- createDataPartition(y= nptr.data$store_nbr, p =0.01, list= FALSE)[,1]
cv.data <- nptr.data[cv.index,]
cv.data$code <- mapply(FUN=pick.weather, cv.data$store_nbr,cv.data$date)


########
rm(sub.train.data)
rm(train.data,c.matrix,cc,cCorpus,out,sst,text,train.weather)
rm(c.mat)
##########################
ptr.data$code <- mapply(FUN=pick.weather, ptr.data$store_nbr,ptr.data$date)
str(ptr.data)
sum(is.na(ptr.data$code))/length(ptr.data$code)


split.by.wspace <- function(string){
  if(!is.na(str)){  
    str.wspace.vector <- unlist(strsplit(string, split = "\\s+"))
    str.fxd.width <- unlist(lapply(str.wspace.vector, FUN=split.by.width))
    str.fxd.space <- paste0(str.fxd.width, collapse = " ")
    return(str.fxd.space)
  }
  else{
    return("")
  }
}

head(ptr.data)

split.by.width <- function(string){
  if(nchar(string)%%2==0){
    sst <- unlist(strsplit(string,split= character(0) ))
    out <- paste0(sst[c(TRUE, FALSE)], sst[c(FALSE, TRUE)])
    return(out)
  }
  else{
    return(string)
  }
}

# preparing training data
ptr.data$scode <- (sapply(ptr.data$code,FUN=split.by.wspace))


ptr.data$scode <-  gsub("+","P",ptr.data$scode, fixed=TRUE)
ptr.data$scode <-  gsub("-","M",ptr.data$scode, fixed=TRUE)
unique(ptr.data$scode)


library(tm)
#prparing cv data
cv.data$scode <- (sapply(cv.data$code,FUN=split.by.wspace))
cvCorpus <- Corpus(VectorSource(cv.data$scode))
cvMatrix <- DocumentTermMatrix(cvCorpus, control =  list(stopwords=FALSE, wordLengths=c(0, Inf)))
cv.matrix <- as.matrix(cvMatrix)
################
wCorpus <- Corpus(VectorSource(ptr.data$scode))
typeMatrix <- DocumentTermMatrix(wCorpus, control =  list(stopwords=FALSE, wordLengths=c(0, Inf)))
w.matrix <- as.matrix(typeMatrix)
dim(w.matrix)
###################

names(ptr.data)
library(dplyr)
w.tr.data <- select(ptr.data, store_nbr, item_nbr,units)
w.tr.data <- cbind(w.tr.data,w.matrix )
names(w.tr.data)
############
names(cv.data)
w.cv.data <- select(cv.data, store_nbr, item_nbr)
w.cv.data <- cbind(w.cv.data,cv.matrix )

str(w.cv.data)
library(e1071)
?svm

s.fit <- svm(x=subset(w.tr.data, select = -c(units)), y = w.tr.data$units, 
             type = "eps-regression", scale=FALSE, kernel = "linear" )
library(caret)


#################

library(randomForest)
rf <- randomForest(x=subset(w.tr.data, select = -c(units)), y = w.tr.data$units,ntree=300,mtry =5)

library(dplyr)
nrow(filter(train.data,units==0))/nrow(train.data)

n0.train.data <- filter(train.data,!units==0)
rm(n0.train.data)

str(n0.train.data)

?predict
?predict.svm
predict.units <- predict(rf,w.cv.data)
head(predict.units)
head(cv.data$units)
install.packages("hydroGOF")
library(hydroGOF)
ms <- rmse(predict.units,cv.data$units )
unique(cv.data$units)
pred.vs.test <- cbind(predict.units,cv.data$units)

se <- sum((predict.units-cv.data$units)^2)/sum(cv.data$units^2)

predict.units[1000:1100]
cv.data$units[1000:1100]
max(predict.units)
min(predict.units)
summary(s.fit)

####################################
library(magrittr)
library(dplyr)
library(caret)
library(tm)
## picking the data where units sold are non-zero
n0.train.data <- train.data %>% 
  filter(!units==0)

# adding the weather code
n0.train.data$code <-mapply(FUN=pick.weather, n0.train.data$store_nbr,n0.train.data$date)

str(n0.train.data)
# dividing between training and cross validation set
tr.index <- createDataPartition(y= n0.train.data$store_nbr, p =0.7, list= FALSE)[,1]


cv.n0.data <- n0.train.data[-tr.index,]
tr.n0.data <- n0.train.data[tr.index,]

# freeing the environment
rm (key.data,n0,train.data,train.data)
rm(weather.data)
rm(n0.train.data)

########################

# perparing the data for analysis
n0.train.data$scode <- (sapply(n0.train.data$code,FUN=split.by.wspace))

n0.matrix <- n0.train.data$scode %>%
  VectorSource %>%
  Corpus %>%
  DocumentTermMatrix(.,control =  list(stopwords=FALSE, wordLengths=c(0, Inf)))%>%
  as.matrix

dim(n0.matrix)
n0.data <- cbind(n0.train.data,n0.matrix)              

rm(part.train.data,ptr.data,rf, w.matrix,w.tr.data,train.weather)
rm(n0.matrix, n0.train.data,tr.matrix,tr.n0.data)
rm(tr.index)
rm(cv.n0.data)
###################

# dividing between training and cross validation set
tr.index <- createDataPartition(y= n0.data$store_nbr, p =0.7, list= FALSE)[,1]
# reducing the store and item numbers to factors
n0.data$store_nbr <- as.character((n0.data$store_nbr))
n0.data$item_nbr <- as.character(n0.data$item_nbr)

cv.n0.data <- n0.data[-tr.index,]
tr.n0.data <- n0.data[tr.index,]

names(tr.n0.data)
# train a svm on the training set
library(e1071)
svm.fit <- svm(x=subset(tr.n0.data, select = -c(units,date,code,scode)), 
               y = tr.n0.data$units, 
               type = "eps-regression", scale=FALSE, kernel = "linear" )

summary(svm.fit)
names(cv.n0.data)
predict.units <- predict(svm.fit,subset(cv.n0.data,select = -c(date,units,code,scode)))

se <- round(sum((predict.units-cv.n0.data$units)^2)/sum(cv.n0.data$units^2),3)

str(cv.n0.data)

library(randomForest)
rf.model <- randomForest(x=subset(tr.n0.data, select = -c(units,date,code,scode)), y = tr.n0.data$units,ntree=100,mtry =5)

predict.units <- predict(rf.model,subset(cv.n0.data,select = -c(date,units,code,scode)))

summary(rf.model)