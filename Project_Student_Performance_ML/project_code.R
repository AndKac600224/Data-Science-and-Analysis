dane <- read.csv(file="Student_Performance.csv", sep=',', header=T)
head(dane)

#######PREPROCESSING

summary(dane)
sum(is.na(dane))
str(dane)

library(dplyr)
dane <- dane %>%
  mutate(Extracurricular.Activities = ifelse(Extracurricular.Activities == "Yes",
                                             1, 0)
  )

summary(dane)

#######BOXPLOTY

bxp_dane <- dane[, c(1, 2, 4, 5, 6)]

par(mfrow = c(2, 3)) 

for (i in 1:ncol(bxp_dane)) {
  boxplot(bxp_dane[, i], 
          main = names(bxp_dane)[i],  
          col = "lightgreen",         
          ylab = "Wartość",
          outpch = 19,                
          outcol = "red")             
}

#########PREPROCESSING

dane_std <- as.data.frame(scale(dane))
str(dane_std)
summary(dane_std)


#########Zbiór train-test

set.seed(419925)
sets <- sample(1:nrow(dane_std), 0.8 * nrow(dane_std))
train_data <- dane_std[sets, ]
test_data <- dane_std[-sets, ]

nrow(train_data)
nrow(test_data)

#########Dystanse i aglomeracja

###Euclidean

dist_e <- dist(train_data, method='euclidean')

head(dist_e)

m1_e <- hclust(dist_e, method='average')
m2_e <- hclust(dist_e, method='complete')
m3_e <- hclust(dist_e, method='ward.D')
m4_e <- hclust(dist_e, method='single')

par(mfrow=c(2,2))
plot(m1_e, main="Method: Average") #9
plot(m2_e, main="Method: Complete") #9
plot(m3_e, main="Method: Ward.D") #5
plot(m4_e, main="Method: Single") #slabo

cut_m1_e <- cutree(m1_e, k=9)
cut_m2_e <- cutree(m2_e, k=9)
cut_m3_e <- cutree(m3_e, k=5)

library(cluster)
library(clValid)


dunn(dist_e, cut_m1_e)
dunn(dist_e, cut_m2_e)
dunn(dist_e, cut_m3_e)

sil_ward_e <- silhouette(cut_m3_e, dist_e)
summary(sil_ward_e) 

sil_complete_e <- silhouette(cut_m2_e, dist_e)
summary(sil_complete_e)

results_clus_e <- mutate(train_data, m1_e=cut_m3_e)
table(results_clus_e$m1_e)

####EUCLIDEAN - WARD WYGRYWA SILHOUETTEM ALE COMPLETE WYGRYWA DUNNEM

###MANHATTAN

dist_m <- dist(train_data, method='manhattan')

head(dist_m)

m1_m <- hclust(dist_m, method='average')
m2_m <- hclust(dist_m, method='complete')
m3_m <- hclust(dist_m, method='ward.D')
m4_m <- hclust(dist_m, method='single')

par(mfrow=c(2,2))
plot(m1_m, main="Method: Average") #6
plot(m2_m, main="Method: Complete") #6
plot(m3_m, main="Method: Ward.D") #3
plot(m4_m, main="Method: Single") #slabo

cut_m1_m <- cutree(m1_m, k=7)
cut_m2_m <- cutree(m2_m, k=7)
cut_m3_m <- cutree(m3_m, k=4)

library(cluster)
library(clValid)


dunn(dist_m, cut_m1_m)
dunn(dist_m, cut_m2_m)
dunn(dist_m, cut_m3_m)

sil_ward_m <- silhouette(cut_m3_m, dist_m)
summary(sil_ward_m) 

sil_average_m <- silhouette(cut_m1_m, dist_m)
summary(sil_average_m)

results_clus_m <- mutate(train_data, m1_m=cut_m3_m)
table(results_clus_m$m1_m)


dunn_table <- data.frame(
  Average  = c(dunn(dist_e, cut_m1_e)[1], dunn(dist_m, cut_m1_m)[1]),
  Complete = c(dunn(dist_e, cut_m2_e)[1], dunn(dist_m, cut_m2_m)[1]),
  Ward.D   = c(dunn(dist_e, cut_m3_e)[1], dunn(dist_m, cut_m3_m)[1]),
  Single   = c("Not interpretable", "Not interpretable")
)

rownames(dunn_table) <- c("Euclidean", "Manhattan")

print(dunn_table)


####MANHATTAN WARD WYGRYWA i EUCLIDEAN COMPLETE###################

########Predykcja kNN

library(MLmetrics)
library(caret)

head(dane)
X_knn <- dane[, c(1,2,3,4,5)]
X_knn_scaled <- as.data.frame(scale(X_knn))
Y <- dane[, c(6)]
dane2 <- cbind(X_knn_scaled, Performance.Index=Y)
set.seed(419925)
sets2 <- sample(1:nrow(dane2), 0.8*nrow(dane2))
train_knn <- dane2[sets2,]
test_knn <- dane2[-sets2,]

#manhattan ward
model <- knnreg(Performance.Index ~ Hours.Studied + Previous.Scores + Extracurricular.Activities + Sleep.Hours + Sample.Question.Papers.Practiced , data=train_knn, k=4)
pred <- predict(model, test_knn)
mse  <- mean((test_knn$Performance.Index - pred)^2)
rmse <- RMSE(test_knn$Performance.Index, pred) #3.01
r2 <- cor(test_knn$Performance.Index, pred)^2 #.0.979
mape <-MAPE(pred, test_knn$Performance.Index) #0.051

#euclidean complete
model2 <- knnreg(Performance.Index ~ Hours.Studied + Previous.Scores + Extracurricular.Activities + Sleep.Hours + Sample.Question.Papers.Practiced , data=train_knn, k=9)
pred2 <- predict(model2, test_knn)
mse2  <- mean((test_knn$Performance.Index - pred2)^2)
rmse2<- RMSE(test_knn$Performance.Index, pred2) #2.90
r2_2 <- cor(test_knn$Performance.Index, pred2)^2 #.0.979
mape2 <-MAPE(pred2, test_knn$Performance.Index) #0.051

