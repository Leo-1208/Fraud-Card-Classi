credit_card <- read.csv('E:\\New folder\\creditcard.csv')
#Imported the dataset from the respected Folder

str(credit_card)

credit_card$Class <- factor(credit_card$Class, levels = c(0, 1))
#Convert Class from 'int' to a factor Variable 

summary(credit_card)

sum(is.na(credit_card))
#Count the missing dataset 0 here :)




#Disrtibution Table
table(credit_card$Class)
#Percentage Table
prop.table(table(credit_card$Class))

#Pie chart of credit card transcations

labels <- c("Legit","Fraud")
#round the percentage to 2 decimals only then pasting it with labels variable 
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)),2))
labels <- paste0(labels, "%")

pie(table(credit_card$Class),labels,col=c("green","black"), main= "Pie Chart of Credit card Transactions")




#No model Predictions
predictions <- rep.int(0,nrow(credit_card))
predictions <- factor(predictions,levels=c(0,1))

install.packages('caret')
library(caret)
confusionMatrix(data=predictions, reference = credit_card$Class)
#Accuracy comes 99.83% assumin each case as legit case



library(dplyr)

set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)
#Reducing the dataset to 10 percent to increase the speed for now. Set seed to generate same 10% datasets.

table(credit_card$Class)

library(ggplot2)

ggplot(data=credit_card,aes(x=V1,y=V2,col=Class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue','red'))
#Plotting the datasets
#We cannot train the model on this dataset




#Creating the training and test sets for Fraud Detection Model

install.packages('caTools') 
library(caTools)

set.seed(128)

#Spliting data into 80%-20% TRUE AND FALSE  SETSEED To create same data samples
data_sample =sample.split(credit_card$Class,SplitRatio=.80)

train_data = subset(credit_card,data_sample==TRUE)
test_data = subset(credit_card,data_sample==FALSE)




#RANDOM OVER SAMPLING (ROS)

table(train_data$Class)

no_legit <- 22750
no_fraud <- 22750
no_total <- no_legit/.50


install.packages('ROSE')
library(ROSE)
oversampling_result <- ovun.sample(Class ~ ., data = train_data, method ="over",N = no_total, seed=2003)
#Class ~ . in the function the '.' represents all other are independent variable...

oversampled_credit <- oversampling_result$data
table(oversampling_credit$Class)

str(oversampled_credit)

oversampling_credit$Class <- factor(oversampling_credit$Class)
#conerting class to factor again as again loaded the credit_card data

ggplot(data=oversampling_credit,aes(x=V1,y=V2,col=Class))+
  geom_point(position=position_jitter(width=.1))+
  theme_bw()+
  scale_color_manual(values = c('blue','red'))
#Plotting the oversampled datasets, adding jitter to see duplicates





#RANDOM UNDER SAMPLING (RUS)

table(train_data$Class)
no_u_fraud <- 35
no_u_legit <- 35
no_u_total <- 70

undersampling_result <- ovun.sample(Class ~ ., data = train_data, method ="under",N = no_u_total, seed=2003)
undersampled_credit <- undersampling_result$data

ggplot(data=undersampled_credit,aes(x=V1,y=V2,col=Class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue','red'))




#BOT RUN AND ROS

table(train_data$Class)

n_new <- nrow(train_data)
fraction <- .50
#we want 50% data is legit and 50 is fraud using both RUN and ROs

sampling_result <- ovun.sample(Class~., data= train_data, method="both",N=n_new,p=fraction,seed=2003)

sampled_credit <- sampling_result$data

table(sampled_credit$Class)

ggplot(data=sampled_credit,aes(x=V1,y=V2,col=Class))+
  geom_point(position=position_jitter(width=.1))+
  theme_bw()+
  scale_color_manual(values = c('blue','red'))





#SMOTE Synthetic Minority Oversampling Technique
install.packages("smotefamily")
library(smotefamily)

table(train_data$Class)

n0 <- 22750
n1 <- 35
r0 <- .55
#r0 is ratio we want after SMOTE

#Calculate the value for the dup_size paramete of SMOTE or No.of times we will perform SMOTE
ntimes <- ((1-r0)/r0)*(n0/n1)-1

smote_output = SMOTE(X = train_data[ , -c(1,31)],
                     target = train_data$Class,
                     K=5,
                     dup_size = ntimes)
#K is the number of near neighbours we want, X is the independent varibales and target to be predicted
#[,-c(1,31)] means every row and columns - 1st and 31st column which was time and Class(target)

#Class name is changed to class either rename the 31st row or use class only.
credit_smote <- smote_output$data
credit_smote$class  <- factor(credit_smote$class, levels = c(0, 1))

str(credit_smote)

prop.table(table(credit_smote$class))

ggplot(data=credit_smote,aes(x=V1,y=V2,col=class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue','red'))




##DECISION TREE

install.packages('rpart.plot')
library(rpart)
library(rpart.plot)

CART_model <-rpart(class ~ .,credit_smote)
rpart.plot(CART_model, extra =0 ,type =5 ,tweak =1.2)

predicted_val<-predict(CART_model, test_data, type= 'class')


#build Confusion Matrix
library(caret)
confusionMatrix(predicted_val, test_data$Class)
#Accuracy is 99.07. 7 out of 9 is predicted true in fardulent case and 5625 is corred predicted true out of 5687 cases 

predicted_val<-predict(CART_model, credit_card[,-1], type= 'class')
confusionMatrix(predicted_val, credit_card$Class)
#Accuracy is 97.56. 422 out of 492 is predicted true in fardulent case and 277441 is corred predicted true out of 284315 cases 



#DECISION TREE WITHOUT SMOTE

CART_model <-rpart(Class ~ .,train_data[,-1])
rpart.plot(CART_model, extra =0 ,type =5 ,tweak =1.2)

predicted_val<-predict(CART_model, test_data[,-1], type= 'class')


#build Confusion Matrix
library(caret)
confusionMatrix(predicted_val, test_data$Class)
#Accuracy is 98.89. only 6 out of 9 is predicted true in fardulent case and 5684 is corred predicted true out of 5686 cases 

predicted_val<-predict(CART_model, credit_card[,-1], type= 'class')
confusionMatrix(predicted_val, credit_card$Class)
#Accuracy is 99.93. 380 out of 492 is predicted true in fardulent case and 284232 is corred predicted true out of 284315 cases 

#Thus with SMOTE we are able to detect 42 more fradulent cases