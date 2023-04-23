library(caret) # for minmax scaling
library(ggcorrplot) # for a ggplot type correlation plot 
library(factoextra) # for fviz_eig function; it is used to plot the scree plot of PCA
library(corrplot) # for a different style correlation plot
library(rpart) # for creating decision tree
library(rpart.plot) # for plotting decision tree
library(e1071) # for SVM
library(randomForest) # for RandomForest
library(DMwR) # for SMOTE (handling data imbalance)

df=read.csv('College_admission.csv')
df$admit <- as.factor(df$admit)
df$ses <- as.factor(df$ses)
df$Gender_Male <- as.factor(df$Gender_Male)
df$Race <- as.factor(df$Race)
df$rank <- as.factor(df$rank)
head(df)

table(df$admit)

#for SMOTE, the target needs to be a factor
df$admit <- as.factor(df$admit)
#Using SMOTE
set.seed(3)
balanced_df <- SMOTE(admit ~ ., data=df, perc.over = 300, perc.under = 132)
table(balanced_df$admit)

sum(is.na(balanced_df))

any(is.na(balanced_df))

#To find all the unique values in categorical columns
c <- colnames(balanced_df)
c <- c[! c %in% c('gre','gpa')]
for (i in c){
    cat(paste(i,":",unique(balanced_df[i]),'\n'))
}

summary(balanced_df$gre)
Q1 <- quantile(balanced_df$gre,0.25)
Q3 <- quantile(balanced_df$gre,0.75)
iqr <- IQR(balanced_df$gre)
min_range1= Q1 - 1.5*iqr
max_range1= Q3 + 1.5*iqr
outlier=which(balanced_df$gre < min_range1 | balanced_df$gre > max_range1)
cat('GRE: Lower Whisker:',min_range1,'\n')
cat('GRE: Upper Whisker:',max_range1,'\n')
cat('GRE: Outlier indices are: [',outlier,']\n')
cat('GRE: Outlier value(s) are:',balanced_df[outlier,'gre'])

# verifying outliers we calculated with the boxplot's outlier values
boxplot(balanced_df$gre,plot=FALSE)$out

boxplot(balanced_df$gre, horizontal = T, notch=T, col='light blue',main='GRE with outliers')

summary(balanced_df$gpa)
Q1 <- quantile(balanced_df$gpa,0.25)
Q3 <- quantile(balanced_df$gpa,0.75)
iqr <- IQR(balanced_df$gpa)
min_range= Q1 - 1.5*iqr
max_range= Q3 + 1.5*iqr
outlier=which(balanced_df$gpa < min_range | balanced_df$gpa > max_range)
cat('GPA: Lower Whisker:',min_range,'\n')
cat('GPA: Upper Whisker:',max_range,'\n')
cat('GPA: Outlier indices are: [',outlier,']\n')
cat('GPA: Outlier value(s) are:',balanced_df[outlier,'gpa'])

# verifying outliers we calculated with the boxplot's outlier values
boxplot(balanced_df$gpa,plot=FALSE)$out

boxplot(balanced_df$gpa, horizontal = T, notch=T, col='light blue',main='GPA with outliers')

balanced_df <- subset(balanced_df, balanced_df$gre > min_range1 & balanced_df$gre < max_range1)

dim(balanced_df) ### checking the dimension after removing outliers

boxplot(balanced_df$gre, horizontal = T, notch=T, col='light blue',main='GRE without outliers')
### We can verify that there is no outlier in the box plot for GRE

#Removing outliers using subset function
balanced_df <- subset(balanced_df, balanced_df$gpa > min_range & balanced_df$gpa < max_range)

dim(balanced_df) ### checking the dimension after removing outliers

boxplot(balanced_df$gpa, horizontal = T, notch=T, col='light blue',main='GPA without outliers')
### We can verify that there is no outlier in the box plot for GPA

#Structure of the dataset
str(balanced_df)

head(balanced_df)

# Visualizing with KDE Plot & Q-Q Plot
par(mfcol = c(2,1),fig=c(0,1,0,1),mar=c(5, 6, 2, 4))
plot(density(balanced_df$gre),main='GRE KDE Plot')
qqnorm(balanced_df$gre,main='GRE Normal Q-Q Plot')
qqline(balanced_df$gre)

options(scipen=999) #disabling scientific notation
shapiro.test(balanced_df$gre)

# Visualizing with KDE Plot & Q-Q Plot
par(mfcol = c(2,1),fig=c(0,1,0,1),mar=c(5, 6, 2, 4))
plot(density(balanced_df$gpa),main='GPA KDE Plot')
qqnorm(balanced_df$gpa,main='GPA Normal Q-Q Plot')
qqline(balanced_df$gpa)

shapiro.test(balanced_df$gpa)

plot(density(scale(balanced_df$gre)),main = 'Normalized GRE with Standard Scaler')

minmax_object=preProcess(as.data.frame(balanced_df$gre),method = c('range')) ## Fit the data into minmax scaler object
minmax_values=predict(minmax_object,as.data.frame(balanced_df$gre))          ## Transform the data using minmax scaler object
plot(density(minmax_values$'balanced_df$gre'),main = 'Normalized GRE with MinMax Scaler') ## plot the minmax'ed values

plot(density(scale(balanced_df$gpa)),main = 'Normalized GPA with Standard Scaler')

minmax_object=preProcess(as.data.frame(balanced_df$gpa),method = c('range')) ## Fit the data into minmax scaler object
minmax_values=predict(minmax_object,as.data.frame(balanced_df$gpa))          ## Transform the data using minmax scaler object
plot(density(minmax_values$'balanced_df$gpa'),main = 'Normalized GPA with MinMax Scaler') ## plot the minmax'ed values

# we are creating this dataframe for modelling purpose so that we can scale train and test data separately
balanced_df1=balanced_df
balanced_df$gre=scale(balanced_df$gre)

balanced_df$gpa=scale(balanced_df$gpa)

head(balanced_df)

# Creating new dataframe
df1=balanced_df

#converting factor class features to numeric class
df1$admit <- as.numeric(df1$admit)
df1$ses <- as.numeric(df1$ses)
df1$Gender_Male <- as.numeric(df1$Gender_Male)
df1$Race <- as.numeric(df1$Race)
df1$rank <- as.numeric(df1$rank)

#checking head of df1
head(df1)

ggcorrplot(cor(df1),lab=T)

corrplot::corrplot(cor(df1))

pca=princomp(cor(df1))
summary(pca)

#install.packages('factoextra') --> to get the function: fviz_eig
# fviz_eig is used to plot the scree plot
fviz_eig(pca, addlabels = TRUE,choice='variance')

df2=balanced_df1

set.seed(3)
index=sample(x=1:nrow(df2),size=floor(0.75 * nrow(df2)))
train_data=df2[index,]
test_data=df2[-index,]

## Converting admit to factor class
#train_data$admit <- as.factor(train_data$admit)
#test_data$admit <- as.factor(test_data$admit)
cat("Train Data Dimension:",dim(train_data))
head(train_data,5)
cat("Test Data Dimension:",dim(test_data))
head(test_data,5)

train_data$gre=scale(train_data$gre)
train_data$gpa=scale(train_data$gpa)

test_data$gre=scale(test_data$gre)
test_data$gpa=scale(test_data$gpa)

base_logistic_model <- glm(admit ~ ., data = train_data, family = "binomial")
summary(base_logistic_model)

step_model=step(base_logistic_model)

step_model$coefficients

summary(step_model)

updated_step_model1=update(step_model, .~. -ses)
summary(updated_step_model1)

car::vif(updated_step_model1)

prob_train <- predict(updated_step_model1, newdata = train_data, type = "response")
class_prob_train=ifelse(prob_train < 0.5, 0, 1)
train_table <- table(train_data$admit,class_prob_train, dnn =list('Actual','Predicted'))
confusionMatrix(train_table,positive = '1')

prob_test <- predict(updated_step_model1, newdata = test_data, type = "response")
class_prob_test=ifelse(prob_test < 0.5, 0, 1)
test_table <- table(test_data$admit,class_prob_test, dnn =list('Actual','Predicted'))
confusionMatrix(test_table,positive = '1')

# K-Fold Cross Validation
set.seed(3)
train_control <- trainControl(method = "cv", number = 10)
K_fold_model <- train(admit ~ gre + rank + gpa, 
                      data = train_data, 
                      trControl = train_control, 
                      method = "glm", 
                      family = "binomial")
summary(K_fold_model)

print(K_fold_model)

tree_model=rpart(admit~gre + rank + gpa,data=train_data,method='class')
par(mfrow = c(1,1)) # resetting the subplotting to defaults
rpart.plot(tree_model,cex=0.7,extra=1)

tree_test_pred=predict(tree_model, newdata=test_data, type = "class")
tree_test_table <- table(test_data$admit, tree_test_pred, dnn = list("Actual", "Predicted"))
confusionMatrix(tree_test_table, positive = "1")

svm_model=svm(admit ~ gre + rank + gpa, data=train_data )
svm_test_pred=predict(svm_model, newdata=test_data)
svm_test_table <- table(test_data$admit, svm_test_pred, dnn = list("Actual", "Predicted"))
confusionMatrix(svm_test_table, positive = "1")

set.seed(3) #setting seed for reproducible outputs
randomforest_model <- randomForest(admit ~ gre + rank + gpa, data = train_data, ntree= 5000)
randomforest_model

randomforest_test_pred=predict(randomforest_model, newdata=test_data, type = "response")

randomforest_test_table <- table(test_data$admit, randomforest_test_pred, dnn = list("Actual", "Predicted"))
cat('__________________________________________________________________')
confusionMatrix(randomforest_test_table, positive = "1")

#type='response' in RandomdForest.Predict gives direct classes of Target(to get probabilites in RanfFor, use type='vote')
#type='response' in LogisticRegression.Predict gives probabilities of Target(to get the classes in LogReg, use ifelse)

df$cat_gre <- ifelse(df$gre < 441, 'Low', ifelse(df$gre <= 580, 'Medium','High'))
table(df$admit,df$cat_gre)
head(df,5)

# Create a dataframe to get the total length of GRE_Categorical for the entire dataset:
df_admitted=aggregate(df$admit~df$cat_gre,FUN='length')

# Create a column called Admitted Count that takes in only the count of value 1 from df2$admit 
# We use the table function for this purpose
df_admitted$'Admitted_Count'=table(df$admit,df$cat_gre)[2,]

#Rename the column names appropriately
colnames(df_admitted)[c(1,2)]=c('GRE_Categorical','Total_length_of_GRE_Categorical')

#Calculating the Admission Probability Percentage
df_admitted$'Admission_Probability_Percentage'=df_admitted$'Admitted_Count'/df_admitted$'Total_length_of_GRE_Categorical'
df_admitted

ggplot(df_admitted,aes(GRE_Categorical,Admission_Probability_Percentage))+geom_point()


