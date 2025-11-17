
install.packages("tidyverse")
library(tidyverse)
library(corrplot)

FrameT <- read.csv("titanic_new.csv")

meanage <- FrameT %>% summarise(age_mean = mean(Age, na.rm = TRUE))
print(meanage)

male <- sum(FrameT$Sex == "male")
female <- sum(FrameT$Sex == "female")
survived <- sum(FrameT$Survived == "1")

#--------------------------PREPROCESSING------------------------------------

FrameT1 <- subset(FrameT, select = c(Survived, Pclass, Sex, Age))

FrameT1$Age[is.na(FrameT1$Age)] <- round(median(FrameT1$Age, na.rm = TRUE))

FrameT1$Survived[FrameT1$Survived != 0 & FrameT1$Survived != 1] <- 1

FrameT1$Sex <- ifelse(substr(FrameT1$Sex, 1, 1) == "f", "female",
                ifelse(substr(FrameT1$Sex, 1, 1) == "m", "male", FrameT1$Sex))

table(FrameT1$Survived)
table(FrameT1$Sex) 
table(FrameT1$Sex, FrameT1$Survived ) 
table(FrameT1$Pclass) 
passengers <- sum((FrameT$Cabin != "")) 
print(passengers)
summary(FrameT1)

hist(FrameT1$Pclass)

FrameT1$Survived <- as.numeric(FrameT1$Survived)
FrameT1$Pclass <- as.numeric(FrameT1$Pclass)

ggplot(FrameT1, aes(x = factor(Pclass), fill = factor(Survived))) +
    geom_bar(position = "fill") +
    labs(x = "Pclass", y = "Percentage", fill = "Survived") +
    ggtitle("Relation between SURVIVED and PCLASS") +
    theme_minimal()

hist(FrameT1$Age)

cor(FrameT1$Age, FrameT1$Survived, method = 'spearman')
cor(FrameT1$Age, FrameT1$Pclass, method = 'spearman')
