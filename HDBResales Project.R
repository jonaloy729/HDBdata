## Loy Jong Sheng
## HDBResales Project.R 
## HarvardX PH125.9x Data Science: Capstone - Choose Your Own Project
## title: Prediction of Singapore HDB Resale Prices
## date: 01/05/2021
##############################################################
# Prediction of Singapore HDB Resale Prices Project R Script 
##############################################################

## ----setup, include=FALSE-------------------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo=TRUE)


## ----Load libraries, echo=FALSE, message=FALSE, warning=FALSE-------------------------------------------------------------------------------------------------------------
if (!require(tidyverse)) install.packages('tidyverse')
if (!require(caret)) install.packages('caret')
if (!require(kernlab)) install.packages('kernlab')
if (!require(dplyr)) install.packages('dplyr')
if (!require(data.table)) install.packages('data.table')
if (!require(ggplot2)) install.packages('ggplot2')
if (!require(lubridate)) install.packages('lubridate')
if (!require(recosystem)) install.packages('recosystem')
if (!require(Matrix)) install.packages('Matrix')
if (!require(recommenderlab)) install.packages('recommenderlab')
if (!require(readxl)) install.packages('readxl')
if (!require(corrplot)) install.packages('corrplot')
if (!require(coefplot)) install.packages('coefplot')
if (!require(boot)) install.packages('boot')
if (!require(tibble)) install.packages('tibble')
if (!require(car)) install.packages('car')
if (!require(mctest)) install.packages('mctest')
if (!require(rpart)) install.packages('rpart')
if (!require(readr)) install.packages('readr')
if(!require(tinytex)) install.packages("tinytex")
library(tidyverse)
library(caret)
library(kernlab)
library(dplyr)
library(data.table)
library(ggplot2)
library(lubridate)
library(recosystem)
library(Matrix)
library(recommenderlab)
library(readxl)
library(corrplot)
library(coefplot)
library(boot)
library(tibble)
library(car)
library(mctest)
library(rpart)
library(readr)
library(knitr)


## ----Download Singapore HDB Resales dataset and data preparation, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------
#Download the data file from github
urlfile <- 
  "https://raw.githubusercontent.com/jonaloy729/HDBdata/main
/resale-flat-prices%20from-jan-2015-onwards.csv"

HDBResales<-read_csv(url(urlfile))


## ----70% training dataset and 30% validation dataset from Singapore HDB Resales dataset, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------
set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(123)`

#Splitting the dataset randomly for training and validating the models.
test_index <- createDataPartition(y = HDBResales$resale_price, 
                                  times = 1, 
                                  p = 0.3, 
                                  list = FALSE)

#70% of the dataset use for training the models. 
HDBResalestrain <- HDBResales[-test_index,]
#30% of the dataset use for validating the models.
HDBResalesvalidation <- HDBResales[test_index,]


## ----Explore training dataset, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------------------------
head(HDBResalestrain)


## ----summarize the number of distinct parameters, echo=TRUE, message=FALSE, warning=FALSE---------------------------------------------------------------------------------
HDBResalestrain %>% summarize(Num_Distinct_Town = n_distinct(town), 
                              Num_Distinct_Flat_Type = n_distinct(flat_type), 
                              Num_Distinct_Storey_Range = n_distinct(storey_range), 
                              Num_Distinct_Flat_Model = n_distinct(flat_model), 
                              Tot_size = nrow(HDBResalestrain))

## ----summary of HDB Resales training dataset, echo=TRUE, message=FALSE, warning=FALSE-------------------------------------------------------------------------------------
summary(HDBResalestrain)


## ----Exploring HDB Resale Flat transactions from Jan 2015 and Sept 2020, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------
HDBresaledemandbymth <- HDBResalestrain %>% group_by(month) %>% 
  summarize(Num_transactions = n())
#Plot Number of HDB Resale flat transaction by month
HDBresaledemandbymth %>% ggplot(mapping = aes(x = month, y = Num_transactions)) + 
  geom_col(fill = "blue", color = "grey") + 
  labs(y = "Number of transactions", x = "Month") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of HDB Resale flat transaction by month")


## ----Exploring HDB Resale Flat transactions by Towns, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------------------
NumResalesBytown <- HDBResalestrain %>% group_by(town) %>% 
  summarize(Num_transaction = n())
#Plot Number of HDB Resale Flat Transactions by Towns
NumResalesBytown %>% ggplot(mapping = aes(x = town, y = Num_transaction)) + 
  geom_col(fill = "blue", color = "grey") + 
  labs(y = "Number of transactions", x = "Town") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of HDB Resale Flat Transactions by Towns")


## ----Number of HDB Flats by Town, echo=TRUE, message=FALSE, warning=FALSE-------------------------------------------------------------------------------------------------
HDBTowns <- c("ANG MO KIO", "BEDOK", "BISHAN", 
              "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG", 
              "BUKIT TIMAH", "CENTRAL AREA", "CHAO CHU KANG", 
              "CLEMENTI", "GEYLANG", "HOUGANG", 
              "JURONG EAST", "JURONG WEST", "KALLANG/WHAMPOA", 
              "MARINE PARADE", "PASIR RIS", "PUNGGOL", 
              "QUEENTOWN", "SEMBAWANG", "SENGKANG", 
              "SERANGOON", "TAMPINES", "TOA PAYOH", 
              "WOODLANDS", "YISHUN")
#HDB data as of 31 March 2020
NumHDBFlatsInTowns <- c(50726, 62816, 20072, 
                        44285, 54227, 35325, 
                        2554, 12003, 48900, 
                        26730, 30829, 57272, 
                        24122, 75208, 39931, 
                        7860, 29654, 50663, 
                        33164, 30020, 69196, 
                        21632, 72683, 39737, 
                        69900, 65158)
#Plot the Number of HDB Flats by Towns
ggplot(mapping = aes(x = HDBTowns, y = NumHDBFlatsInTowns)) + 
  geom_col(fill = "blue", color = "grey") + 
  labs(y = "Number of HDB Flats", x = "Town") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Number of HDB Flats by Towns")


## ----Exploring HDB Resale Flat transaction by Flat Type, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------
NumResalesbyflattype <- HDBResalestrain %>% group_by(flat_type) %>% 
  summarize(Num_transaction = n())
#Plot Resale Flat transactions by Flat Types
NumResalesbyflattype %>% 
  ggplot(mapping = aes(x = flat_type, y = Num_transaction)) + 
  geom_col(fill = "blue", color = "grey") + 
  labs(y = "Number of transactions", x = "Flat Type") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of HDB Resale Flat Transactions by Flat Type")

## ----Number of HDB Flat Type built, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------
HDBFlatType <- c("1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE AND MULTI-GEN")
#HDB data as of 31 March 2020
NumHDBFlatTypes <- c(30906, 57660, 243519, 424769, 243707, 65107)
#Plot Number of HDB Flats by Types
ggplot(mapping = aes(x = HDBFlatType, y = NumHDBFlatTypes)) + 
  geom_col(fill = "blue", color = "grey") + 
  scale_y_continuous(breaks= seq(0, 450000, by=50000)) + 
  labs(y = "Number of HDB Flats", x = "Flat Type") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Number of HDB Flats by Types")


## ----Exploring HDB Resale Flat transactions by lease commence year, echo=TRUE, message=FALSE, warning=FALSE---------------------------------------------------------------
NumResalesbyleasecommence <- HDBResalestrain %>% group_by(lease_commence_year) %>% 
  summarize(Num_transaction = n())
#Plot Number of HDB Sale Flat Transactions by Lease Commence Year
NumResalesbyleasecommence %>% 
  ggplot(mapping = aes(x = lease_commence_year, y = Num_transaction)) + 
  scale_x_continuous(breaks= seq(1960, 2020, by=5)) + 
  geom_col(fill = "blue", color = "grey") + 
  labs(y = "Number of Transactions", x = "Lease Commence Year") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of HDB Sale Flat Transactions by Lease Commence Year")


## ----Exploring HDB Resale Flat transaction by remaining_lease_year, echo=TRUE, message=FALSE, warning=FALSE---------------------------------------------------------------
NumResalesbyremainlease <- HDBResalestrain %>% 
  group_by(remaining_lease_year) %>% 
  summarize(Num_transaction = n())
#Plot Number of HDB Sale Flat Transactions by Remaining Lease Year
NumResalesbyremainlease %>% 
  ggplot(mapping = aes(x = remaining_lease_year, y = Num_transaction)) + 
  scale_x_continuous(breaks= seq(40, 99, by=5)) + 
  geom_col(fill = "blue", color = "grey") + 
  labs(y = "Number of Transactions", x = "Remaining Lease Year") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of HDB Sale Flat Transactions by Remaining Lease Year")


## ----HDB Resale Flat Prices, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------
#Plot HDB Resale Flat Prices transacted
ggplot(HDBResalestrain, aes(x=resale_price)) + 
  geom_histogram(binwidth = 10000, fill = "blue", color = "grey") + 
  scale_x_continuous(breaks= seq(0, 1300000, by=100000)) + 
  geom_vline(aes(xintercept=mean(resale_price)),color="red",
             linetype="dashed", size=1, alpha = 1) + 
  geom_text(aes(x = mean(resale_price)+100000, y = 3000, 
                label = "Overall Mean"), color="red") + 
  geom_vline(aes(xintercept=median(resale_price)),color="chocolate3", 
             linetype="dashed", size=1, alpha = 1) + 
  geom_text(aes(x = median(resale_price)-100000, y = 3000, 
                label = "Overall Median"), color="chocolate3") + 
  labs(y = "Number of Transactions", x = "HDB Resale Flat Price (S$)") + 
  ggtitle("Distribution of HDB Resale Flat Prices")

## ----Boxplot HDB Resale Flat Prices, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------------------
ggplot(HDBResalestrain, aes(y=resale_price)) + 
  geom_boxplot() +
  labs(y = "HDB Resale Flat Price (S$)") + 
  ggtitle("Boxplot of HDB Resale Flat Prices")


## ----Monthly HDB Resale Flat Price, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------
AvgHDBresalepricebymth <- HDBResalestrain %>% group_by(month) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
#Plot Monthly Average HDB Resale Flat Prices
AvgHDBresalepricebymth %>% 
  ggplot(mapping = aes(x = month, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") + 
  scale_y_continuous(breaks= seq(0, 500000, by=50000)) + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=30), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=30), colour="chocolate3", angle=0) + 
  labs(y = "Average Resale Flat Price (S$)", x = "Month") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Monthly Average HDB Resale Flat Prices")


## ----Exploring HDB Resales Prices by Flat Types, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------
AvgHDBresalepricebyflattype <- HDBResalestrain %>% 
  group_by(flat_type) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
#Plot Average HDB Resale Flat Prices by Flat Types
AvgHDBresalepricebyflattype %>% 
  ggplot(mapping = aes(x = flat_type, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") + 
  scale_y_continuous(breaks= seq(0, 1000000, by=50000)) + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=3), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=3), colour="chocolate3", angle=0) + 
  labs(y = "Average Resale Flat Price (S$)", x = "Flat Type") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Average HDB Resale Flat Prices by Flat Types")

## ----Boxplot HDB Resale Flat Prices by flat types, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------------
ggplot(HDBResalestrain, aes(x = flat_type, y=resale_price)) + 
  geom_boxplot() + geom_hline(yintercept = mean(HDBResalestrain$resale_price), 
                              linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=6), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=6), colour="chocolate3", angle=0) + 
  labs(x = "Flat Type", y = "HDB Resale Flat Price (S$)") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Boxplot of HDB Resale Flat Prices by Flat Types")


## ----Exploring HDB Resales Price by towns, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------------
AvgHDBresalepricebytown <- HDBResalestrain %>% group_by(town) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
#Plot Average HDB Resale Flat Prices by Towns
AvgHDBresalepricebytown %>% 
  ggplot(mapping = aes(x = town, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") + 
  scale_y_continuous(breaks= seq(0, 1000000, by=50000)) + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=15), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=15), colour="chocolate3", angle=0) + 
  labs(y = "Average Resale Flat Price (S$)", x = "Town") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Average HDB Resale Flat Prices by Towns")

## ----Boxplot HDB Resale Flat Prices by Town, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------------------
ggplot(HDBResalestrain, aes(x = town, y=resale_price)) + 
  geom_boxplot() + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=6), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=6), colour="chocolate3", angle=0) + 
  labs(x = "Towns", y = "HDB Resale Flat Price (S$)") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Boxplot of HDB Resale Flat Prices by Towns")


## ----Exploring HDB Resales Price by streets within MARINE PARADE, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------
AvgHDBresalepricebyaddress <- HDBResalestrain %>% filter(town == "MARINE PARADE") %>% 
  group_by(street_name) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
AvgHDBresalepricebyaddress %>% 
  ggplot(mapping = aes(x = street_name, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") + 
  scale_y_continuous(breaks= seq(0, 700000, by=50000)) + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=2), colour="red", angle=0) +
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=2), colour="chocolate3", angle=0) +
  labs(y = "Average Resale Flat Price (S$)", x = "Marine Parade Area") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Avg HDB Resale Flat Price within Marine Parade area")


## ----Exploring HDB Resales Price by streets within ANG MO KIO, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------
AvgHDBresalepricebyaddress <- HDBResalestrain %>% filter(town == "ANG MO KIO") %>% 
  group_by(street_name) %>% summarize(Avg_Resales_Price = mean(resale_price))

AvgHDBresalepricebyaddress %>% 
  ggplot(mapping = aes(x = street_name, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") +
  scale_y_continuous(breaks= seq(0, 900000, by=50000)) +
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=7), colour="red", angle=0) +
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=7), colour="chocolate3", angle=0) +
  labs(y = "Average Resale Flat Price (S$)", x = "Ang Mo Kio Area") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Avg HDB Resale Flat Price within Ang Mo Kio area")


## ----Exploring HDB Resales Price by streets within YISHUN, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------------------
AvgHDBresalepricebyaddress <- HDBResalestrain %>% 
  filter(town == "YISHUN") %>% group_by(street_name) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
#Plot Average HDB Resale Flat Price within Yishun area
AvgHDBresalepricebyaddress %>% 
  ggplot(mapping = aes(x = street_name, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") +
  scale_y_continuous(breaks= seq(0, 600000, by=50000)) +
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=10), colour="red", angle=0) +
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=10), colour="chocolate3", angle=0) +
  labs(y = "Average Resale Flat Price (S$)", x = "Yishun Area") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Avg HDB Resale Flat Price within Yishun area")


## ----Exploring HDB Resales Price by storey range, echo=TRUE, message=FALSE, warning=FALSE---------------------------------------------------------------------------------
AvgHDBresalepricebystorey <- HDBResalestrain %>% 
  group_by(storey_range) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
#Plot Average HDB Resale Flat Prices by Storey Ranges
AvgHDBresalepricebystorey %>% 
  ggplot(mapping = aes(x = storey_range, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=10), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=10), colour="chocolate3", angle=0) + 
  labs(y = "Average Resale Flat Price (S$)", x = "Storey Range") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Average HDB Resale Flat Prices by Storey Ranges")

## ----Boxplot HDB Resale Flat Prices by Storey Ranges, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------------------
ggplot(HDBResalestrain, aes(x = storey_range, y=resale_price)) + 
  geom_boxplot() + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=6), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=6), colour="chocolate3", angle=0) + 
  labs(x = "Storey Range", y = "HDB Resale Flat Price (S$)") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Boxplot of HDB Resale Flat Prices by Storey Ranges")


## ----Exploring HDB Resale Prices by flat models, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------
AvgHDBresalepricebyflatmodel <- HDBResalestrain %>% 
  group_by(flat_model) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
#Plot Average HDB Resale Flat Prices by Flat Models
AvgHDBresalepricebyflatmodel %>% 
  ggplot(mapping = aes(x = flat_model, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+200, 
                label="Overall Mean", x=10), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=10), colour="chocolate3", angle=0) + 
  labs(y = "Average Resale Flat Price (S$)", x = "Flat Model") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Average HDB Resale Flat Prices by Flat Models")


## ----Boxplot HDB Resale Flat Prices by Flat Models, echo=TRUE, message=FALSE, warning=FALSE-------------------------------------------------------------------------------
ggplot(HDBResalestrain, aes(x = flat_model, y=resale_price)) + 
  geom_boxplot() + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=6), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=6), colour="chocolate3", angle=0) + 
  labs(x = "Flat Models", y = "HDB Resale Flat Price (S$)") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Boxplot of HDB Resale Flat Prices by Flat Models")


## ----Exploring HDB Resales Price by lease_commence_year, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------
AvgHDBresalepricebyleasecommence <- HDBResalestrain %>% 
  group_by(lease_commence_year) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
AvgHDBresalepricebyleasecommence %>% 
  ggplot(mapping = aes(x = lease_commence_year, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") + 
  scale_x_continuous(breaks= seq(1965, 2020, by=5)) + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=2000), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=2000), colour="chocolate3", angle=0) + 
  labs(y = "Average Resale Flat Price (S$)", x = "Lease Commence Year") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Average HDB Resale Flat Price by Lease Commence Year")

## ----Boxplot HDB Resale Flat Prices by Lease Commence Year, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------------
ggplot(HDBResalestrain, aes(x = lease_commence_year, y=resale_price, group=lease_commence_year)) + 
  geom_boxplot() + 
  scale_x_continuous(breaks= seq(1965, 2020, by=5)) + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=1990), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=1990), colour="chocolate3", angle=0) + 
  labs(x = "Lease Commence Year", y = "HDB Resale Flat Price (S$)") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Boxplot of HDB Resale Flat Prices by Lease Commence Year")

## ----Display 2017,2018 Lease Commence Year, echo=TRUE, message=FALSE, warning=FALSE---------------------------------------------------------------------------------------
TempRecord <- HDBResalestrain %>% 
  filter(lease_commence_year == 2018 | lease_commence_year == 2019) %>% 
  select(month, lease_commence_year, resale_price)
TempRecord


## ----Exploring HDB Resales Price by Remaining Lease Year, echo=TRUE, message=FALSE, warning=FALSE-------------------------------------------------------------------------
AvgHDBresalepricebyremainlease <- HDBResalestrain %>% 
  group_by(remaining_lease_year) %>% 
  summarize(Avg_Resales_Price = mean(resale_price))
#Plot Average HDB Resale Flat Price by Remaining Lease Year
AvgHDBresalepricebyremainlease %>% 
  ggplot(mapping = aes(x = remaining_lease_year, y = Avg_Resales_Price)) + 
  geom_col(fill = "blue", color = "grey") + 
  scale_x_continuous(breaks= seq(40, 99, by=5)) + 
  scale_y_continuous(breaks= seq(0, 800000, by=50000)) + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=50), colour="red", angle=0) +
  geom_hline(yintercept=median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=50), colour="chocolate3", angle=0) + 
  labs(y = "Average Resale Flat Price", x = "Remaining Lease Year") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Distribution of Average HDB Resale Flat Price by Remaining Lease Year")

## ----Boxplot HDB Resale Flat Prices by Remaining Lease Year, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------
ggplot(HDBResalestrain, aes(x = remaining_lease_year, y=resale_price, 
                            group=remaining_lease_year)) + 
  geom_boxplot() + 
  scale_x_continuous(breaks= seq(40, 99, by=5)) + 
  geom_hline(yintercept= mean(HDBResalestrain$resale_price), 
             linetype="dashed", color = "red") + 
  geom_text(aes(y=mean(HDBResalestrain$resale_price)+25000, 
                label="Overall Mean", x=50), colour="red", angle=0) + 
  geom_hline(yintercept= median(HDBResalestrain$resale_price), 
             linetype="dashed", color = "chocolate3") + 
  geom_text(aes(y=median(HDBResalestrain$resale_price)-25000, 
                label="Overall Median", x=50), colour="chocolate3", angle=0) + 
  labs(x = "Remaining Lease Year", y = "HDB Resale Flat Price (S$)") + 
  theme(axis.text.x= element_text(angle=90,hjust=1)) + 
  ggtitle("Boxplot of HDB Resale Flat Prices by Remaining Lease Year")


## ----Factorizing the character variables, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------
tempHDBResalestrain <- HDBResalestrain %>% 
  select(month, town, block, street_name, storey_range, 
         flat_type, flat_model, floor_area_sqm, 
         lease_commence_year, remaining_lease_year, 
         pop_pri_school_nearby, mrt_nearby, resale_price)

tempHDBResalestrain$month <- 
  as.numeric(factor(HDBResalestrain$month))
tempHDBResalestrain$town <- 
  as.numeric(factor(HDBResalestrain$town))
tempHDBResalestrain$block <- 
  as.numeric(factor(HDBResalestrain$block))
tempHDBResalestrain$street_name <- 
  as.numeric(factor(HDBResalestrain$street_name))
tempHDBResalestrain$storey_range <- 
  as.numeric(factor(HDBResalestrain$storey_range))
tempHDBResalestrain$flat_type <- 
  as.numeric(factor(HDBResalestrain$flat_type))
tempHDBResalestrain$flat_model <- 
  as.numeric(factor(HDBResalestrain$flat_model))
tempHDBResalestrain$lease_commence_year <- 
  as.numeric(factor(HDBResalestrain$lease_commence_year))
tempHDBResalestrain$remaining_lease_year <- 
  as.numeric(factor(HDBResalestrain$remaining_lease_year))

tempHDBResalesvalidation <- HDBResalesvalidation %>% 
  select(month, town, block, street_name, storey_range, 
         flat_type, flat_model, floor_area_sqm, 
         lease_commence_year, remaining_lease_year, 
         pop_pri_school_nearby, mrt_nearby, resale_price)

tempHDBResalesvalidation$month <- 
  as.numeric(factor(HDBResalesvalidation$month))
tempHDBResalesvalidation$town <- 
  as.numeric(factor(HDBResalesvalidation$town))
tempHDBResalesvalidation$block <- 
  as.numeric(factor(HDBResalesvalidation$block))
tempHDBResalesvalidation$street_name <- 
  as.numeric(factor(HDBResalesvalidation$street_name))
tempHDBResalesvalidation$storey_range <- 
  as.numeric(factor(HDBResalesvalidation$storey_range))
tempHDBResalesvalidation$flat_type <- 
  as.numeric(factor(HDBResalesvalidation$flat_type))
tempHDBResalesvalidation$flat_model <- 
  as.numeric(factor(HDBResalesvalidation$flat_model))
tempHDBResalesvalidation$lease_commence_year <- 
  as.numeric(factor(HDBResalesvalidation$lease_commence_year))
tempHDBResalesvalidation$remaining_lease_year <- 
  as.numeric(factor(HDBResalesvalidation$remaining_lease_year))


## ----Exploring the Correlation among the variables, echo=TRUE, message=FALSE, warning=FALSE-------------------------------------------------------------------------------
corresult<- cor(tempHDBResalestrain)
corrplot(corresult, method = "number", number.cex = .7)


## ----Detecting Multicollinearity with Variance Inflation Factors (VIF), echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------
#Train the model
modellm = lm(resale_price ~ ., 
             data = tempHDBResalestrain)
#Detecting Multicollinearity with Variance Inflation Factors (VIF) 
vif(modellm)


## ----Removing lease_commence_year, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------
#Remove lease_commence_year variable
modellm <- update(modellm, .~. - lease_commence_year, 
                  data = tempHDBResalestrain)
vif(modellm)


## ----Removing flat_type, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------------------------------
#Remove flat_type variable
modellm <- update(modellm, .~. - flat_type, 
                  data = tempHDBResalestrain)
vif(modellm)


## ----Derive the Multiple Regression formula, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------------------
#Print the model summary
summary(modellm)


## ----Removing month variable because it is not statistically significant, echo=TRUE, message=FALSE, warning=FALSE---------------------------------------------------------
#Remove month variable because it is not significant.
modellm <- update(modellm, .~. - month, data = tempHDBResalestrain)
summary(modellm)

## ----Coefficient Plot Visualization, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------------------
#Using coefficient plot to visualize the regression results
coefplot(modellm)


## ----Enlarge the check those variables close to zero line in Coefficient Plot, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------
#Enlarge to check those variables that seems to have little effect on resale flat prices.
coefplot(modellm, sort='mag') + scale_x_continuous(limits=c(-400, 400))


## ----Scaling the variables, echo=TRUE, message=FALSE, warning=FALSE-------------------------------------------------------------------------------------------------------
#Train the model
modellm = lm(resale_price ~ town + 
               scale(block) + 
               scale(street_name) + 
               storey_range + 
               scale(flat_model) + 
               floor_area_sqm + 
               remaining_lease_year + 
               pop_pri_school_nearby + 
               mrt_nearby, data = tempHDBResalestrain)
summary(modellm)

## ----Review Coefficient Plot after scaling, echo=TRUE---------------------------------------------------------------------------------------------------------------------
#Review Coefficient Plot after scaling
coefplot(modellm)


## ----Deriving the final formula, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------------------------------
#Derive the final formula
HDBResalePriceFormula <- resale_price ~ town + 
  scale(block) + 
  scale(street_name) + 
  storey_range + 
  scale(flat_model) + 
  floor_area_sqm + 
  remaining_lease_year + 
  pop_pri_school_nearby + 
  mrt_nearby


## ----RMSE Evaluation, echo=TRUE, message=FALSE, warning=FALSE-------------------------------------------------------------------------------------------------------------
#Define the Loss function
RMSE <- function(actual, predicted){
    sqrt(mean((actual - predicted)^2))
}
#Create a RMSE results dataframe to store and compare the results.
RMSE_results <- tibble()


## ----train control, echo=TRUE, message=FALSE, warning=FALSE---------------------------------------------------------------------------------------------------------------
#10-fold cross-validation
TrainCtrl <- trainControl(method = "cv", number = 10, verboseIter = FALSE)


## ----Exploring the lm model, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------
set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(123)`
#Train the model
HDBResalesPriceModellm <- train(HDBResalePriceFormula, 
                                data=tempHDBResalestrain, 
                                method = "lm", 
                                trControl = TrainCtrl)


## ----Multiple Regression Model Residual Plots, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------------------------------
par(mfrow = c(2, 2))  # Split the plotting panel into a 2 x 2 grid
plot(HDBResalesPriceModellm$finalModel)  # Plot the model information


## ----Predict and Calculate RMSE for lm model, echo=TRUE-------------------------------------------------------------------------------------------------------------------
#Make prediction with the validation dataset
HDBResalePricePredictlm <- predict(HDBResalesPriceModellm, newdata = tempHDBResalesvalidation)
#Calculate RMSE
rmse <- RMSE(tempHDBResalesvalidation$resale_price, HDBResalePricePredictlm)
RMSE_results <- tibble(Model = "lm", RMSE_value = rmse)
RMSE_results


## ----Plot Linear Regression Predicted Resale Flat Price with Actual Resale Flat Price, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------
Plotdata = as.data.frame(cbind(predicted = HDBResalePricePredictlm, 
                               actual = tempHDBResalesvalidation$resale_price))
ggplot(Plotdata,aes(predicted, actual)) + 
  geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm) + 
  ggtitle("Linear Regression: Prediction vs Actual Resale Flat Price") + 
  xlab("Predicted Resale Flat Price (S$)") + 
  ylab("Actual Resale Flat Price (S$)") + 
  theme(plot.title = element_text(color="darkgreen",size=12,hjust = 0.5), 
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12,hjust=.5), 
        axis.title.x = element_text(size=12), 
        axis.title.y = element_text(size=1))


## ----Exploring Bagged Decision Tree, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------------------
set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(123)`
#Train the model
HDBResalePriceModelTreeBag <- train(HDBResalePriceFormula, 
                                    data = tempHDBResalestrain, 
                                    method = "treebag", 
                                    trControl = TrainCtrl, 
                                    nbagged = 200, 
                                    control = rpart.control(minsplit = 2, cp = 0))


## ----Bagged Decision Tree, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------------------------------------
#Print the model results
HDBResalePriceModelTreeBag


## ----Predict Resale Prices using Bagged Decision Tree, echo=TRUE----------------------------------------------------------------------------------------------------------
#Make prediction with the validation dataset
HDBResalePricePredictTreeBag <- predict(HDBResalePriceModelTreeBag, 
                                        newdata = tempHDBResalesvalidation)
#Calculate RMSE
rmse <- RMSE(tempHDBResalesvalidation$resale_price, 
             HDBResalePricePredictTreeBag)
RMSE_results <- RMSE_results %>% add_row(Model = "Bagged Decision Tree", 
                                         RMSE_value = rmse)
RMSE_results


## ----Plot Bagged Decision Tree Model Predicted Resale Flat Price with Actual Resale Flat Price, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------
PlotdataBaggedTree = as.data.frame(cbind(predicted = HDBResalePricePredictTreeBag, 
                                         actual = tempHDBResalesvalidation$resale_price))
ggplot(PlotdataBaggedTree,aes(predicted, actual)) + 
  geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm) + 
  ggtitle("Bagged Decision Tree: Prediction vs Acutal Resale Flat Price") + 
  xlab("Predicted Resale Flat Price (S$)") + 
  ylab("Actual Resale Flat Price (S$)") + 
  theme(plot.title = element_text(color="darkgreen",size=12,hjust = 0.5), 
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12,hjust=.5), 
        axis.title.x = element_text(size=12), 
        axis.title.y = element_text(size=12))


## ----Exploring Random Forest, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------
set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(123)`
#Train the model
HDBResalePriceModelRF <- train(HDBResalePriceFormula, 
                               data=tempHDBResalestrain, 
                               method = "ranger", 
                               importance = 'impurity', 
                               tuneLength = 10, 
                               trControl = TrainCtrl)

## ----Display Random Forest info, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------------------------------
#Print model results
HDBResalePriceModelRF


## ----Plot Random Forest, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------------------------------
#Plot the RMSE
plot(HDBResalePriceModelRF)


## ----Predict Resale Price using Random Forest, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------------------------------
#Make prediction with the validation dataset
HDBResalePricePredictRF <- predict(HDBResalePriceModelRF, 
                                   newdata = tempHDBResalesvalidation)
#Calculate RMSE
rmse <- RMSE(tempHDBResalesvalidation$resale_price, 
             HDBResalePricePredictRF)
RMSE_results <- RMSE_results %>% 
  add_row(Model = "Random Forest", RMSE_value = rmse)
RMSE_results


## ----Plot Random Forest Model Predicted Resale Flat Price with Actual Resale Flat Price, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------
PlotdataRF = as.data.frame(cbind(predicted = HDBResalePricePredictRF, 
                                 actual = tempHDBResalesvalidation$resale_price))
ggplot(PlotdataRF,aes(predicted, actual)) + 
  geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm) + 
  ggtitle("Random Forest: Prediction vs Actual Resale Flat Price") + 
  xlab("Predicted Resale Flat Price (S$)") + 
  ylab("Actual Resale Flat Price (S$)") + 
  theme(plot.title = element_text(color="darkgreen",size=12,hjust = 0.5), 
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12,hjust=.5), 
        axis.title.x = element_text(size=12), 
        axis.title.y = element_text(size=12))


## ----Exploring stochastic gradient boosting machine, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------------------------
set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(123)`
#Train the model
HDBResalePriceModelGBM <- train(HDBResalePriceFormula, 
                                data = tempHDBResalestrain,
                                method = "gbm", 
                                tuneLength = 10, 
                                trControl = TrainCtrl, 
                                verbose = FALSE)


## ----Stochastic Gradient Boosting Machine, echo=TRUE, message=FALSE, warning=FALSE----------------------------------------------------------------------------------------
#Plot the RMSE
plot(HDBResalePriceModelGBM)


## ----Predict HDB Rsale Flat Prices using stochastic gradient boosting machine, echo=TRUE----------------------------------------------------------------------------------
#Make prediction with the validation dataset
HDBResalePricePredictGBM <- predict(HDBResalePriceModelGBM, 
                                    newdata = tempHDBResalesvalidation)
#Calculate RMSE
rmse <- RMSE(tempHDBResalesvalidation$resale_price, 
             HDBResalePricePredictGBM)
RMSE_results <- RMSE_results %>% 
  add_row(Model = "Stochastic Gradient Boosting", RMSE_value = rmse)
RMSE_results


## ----Plot stochastic gradient boosting machine Model Predicted Resale Flat Price with Actual Resale Flat Price, echo=TRUE, message=FALSE, warning=FALSE-------------------
PlotdataGBM = as.data.frame(cbind(predicted = HDBResalePricePredictGBM, 
                                  actual = tempHDBResalesvalidation$resale_price))
ggplot(PlotdataGBM,aes(predicted, actual)) + 
  geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm) + 
  ggtitle("Stochastic GBM: Prediction vs Actual Resale Flat Price") + 
  xlab("Predicted Resale Flat Price (S$)") + 
  ylab("Actual Resale Flat Price (S$)") + 
  theme(plot.title = element_text(color="darkgreen",size=12,hjust = 0.5), 
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12,hjust=.5), 
        axis.title.x = element_text(size=12), 
        axis.title.y = element_text(size=12))


## ----Exploring Extreme gradient boosting, echo=TRUE, message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------
set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(123)`
#Train the model
HDBResalePriceModelXGB <- train(HDBResalePriceFormula, 
                                data = tempHDBResalestrain, 
                                method = "xgbTree", 
                                tuneLength = 10, 
                                trControl = TrainCtrl, 
                                objective ="reg:squarederror", 
                                verbose = FALSE)


## ----Predict HDB Resale Price using Extreme Gradient Boosting machine, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------
#Make prediction with the validation dataset
HDBResalePricePredictXGB <- predict(HDBResalePriceModelXGB, 
                                    newdata = tempHDBResalesvalidation)
#Calculate RMSE
rmse <- RMSE(tempHDBResalesvalidation$resale_price, HDBResalePricePredictXGB)
RMSE_results <- RMSE_results %>% 
  add_row(Model = "Extreme Gradient Boosting", RMSE_value = rmse)
RMSE_results


## ----Plot Extreme gradient boosting Model Predicted Resale Flat Price with Actual Resale Flat Price, echo=TRUE, message=FALSE, warning=FALSE------------------------------
PlotdataXGB = as.data.frame(cbind(predicted = HDBResalePricePredictXGB, 
                                  actual = tempHDBResalesvalidation$resale_price))
ggplot(PlotdataXGB,aes(predicted, actual)) + 
  geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm) + 
  ggtitle("Extreme GB: Prediction vs Actual Resale Flat Price") + 
  xlab("Predicted Resale Flat Price (S$)") + 
  ylab("Actual Resale Flat Price (S$)") + 
  theme(plot.title = element_text(color="darkgreen",size=12,hjust = 0.5), 
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12,hjust=.5), 
        axis.title.x = element_text(size=12), 
        axis.title.y = element_text(size=12))


## ----Summary of Resampling metrics for Regression Trees, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------
Resamples <- resamples(list(
                             "Bagged" = HDBResalePriceModelTreeBag,
                             "Random Forest" = HDBResalePriceModelRF,
                             "GBM" = HDBResalePriceModelGBM,
                             "XGBoost" = HDBResalePriceModelXGB
                            ))
summary(Resamples)

## ----Boxplot of Resampling metrics for Regression Trees, echo=TRUE, message=FALSE, warning=FALSE--------------------------------------------------------------------------
#Boxplot of Resampling metrics for Regression Trees
bwplot(Resamples)


## ----RMSE Table, echo=TRUE, message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------------------
#Show the summary of RMSE value obtained from validation dataset
RMSE_results

