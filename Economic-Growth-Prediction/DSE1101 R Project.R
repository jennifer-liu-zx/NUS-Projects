library(ROCR)
library(kknn)
library(e1071)
library(rpart)
library(MASS)
library(rpart)
library(readxl)
library(pls)

rm(list = ls())
set.seed(0283094)

df = read_xls("BACE_data.xls")

clean <- function(dataframe, col_lst) {
  df_clean <- dataframe
  
  for (c in col_lst) {
    df_clean <- df_clean[!is.na(df_clean[, c]), ]
  }
  
  return(df_clean)
}

# structuring and cleaning datasets.

df_num = cbind(df[,1:3], lapply(df[,4:ncol(df)], as.numeric)) # as numeric
df_num_clean = clean(df_num, colnames(df_num))

factor_variables = c('BRIT', 'COLONY', 'EAST', 'ECORG', 'EUROPE', 'LAAM', 
                     'SCOUT', 'SOCIALIST', 'SPAIN', 'WARTORN', 'LANDLOCK', 'NEWSTATE', 'OIL', 'SAFRICA')
df_fact = df_num
df_fact[factor_variables] <- lapply(df_fact[factor_variables], as.factor)
df_fact_clean = clean(df_fact, colnames(df_fact))

# PCA

set.seed(0283094)
ntrain_num = nrow(df_num_clean)*0.9
tr_num = sample(1:nrow(df_num_clean), ntrain_num) 
train_num = df_num_clean[tr_num,]
test_num = df_num_clean[-tr_num,]

prall = prcomp(df_num_clean[,4:ncol(df_num_clean)], scale=TRUE)
biplot(prall, cex=0.8, arrow=FALSE)
biplot(prall, choices=c(3, 4), cex=0.8, arrow=FALSE)
prall.s = summary(prall)
scree = prall.s$importance[2,]
plot(scree, main = "Scree Plot", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", ylim = c(0,0.4), type = 'b', cex = .8)

train_num_slice = as.data.frame(apply(train_num[,4:ncol(train_num)], MARGIN=2, FUN=as.numeric))
test_num_slice = as.data.frame(apply(test_num[,4:ncol(test_num)], MARGIN=2, FUN=as.numeric))
pcr.fit=pcr(GR6096~.,data=train_num_slice, scale=TRUE, validation="CV")
pcr.pred=predict(pcr.fit, newdata=test_num_slice, ncomp=3)
mean((test_num_slice$GR6096-pcr.pred)^2) # 0.0003185723

# decision tree

dev.new()
big.tree = rpart(GR6096~., method="anova", data=train_num_slice, minsplit=30, cp=.0005)
best <- big.tree$cptable[which.min(big.tree$cptable[,"xerror"]), "CP"]
pruned_tree <- prune(big.tree, cp=best)

plot(big.tree, uniform=TRUE)
text(big.tree, digits=4, use.n=TRUE, fancy=FALSE, bg='lightblue', cex=0.5)

treefit = predict(pruned_tree, newdata=test_num_slice, type="vector")
mean((test_num_slice$GR6096-treefit)^2) # 0.0005300571

# linear model

set.seed(0283094)
fact = df_fact_clean[,4:ncol(df_fact_clean)]
ntrain_fact = nrow(df_fact_clean)*0.9
tr_fact = sample(1:nrow(fact[,4:ncol(fact)]), ntrain_fact) 
train_fact = fact[tr_fact,]
test_fact = fact[-tr_fact,]

lm_infos <- function(lm){
  par(mfrow=c(2,2))
  plot(lm)
  predlm=predict.lm(lm, newdata=test_fact)
  mean((test_fact$GR6096-predlm)^2)
}

lm_all = lm(GR6096~., data=train_fact)
lm_infos(lm_all) # 0.0181667

lm_war = lm(GR6096~WARTIME+WARTORN, data=df_fact_clean)
summary(lm_war)
lm_infos(lm_war)

lm_natres = lm(GR6096~OIL+MINING+LHCPC, data=df_fact_clean)
summary(lm_natres)
lm_infos(lm_natres)

lm_combine = lm(GR6096~GOVSH61+YRSOPEN+RERD+P60+I(MALFAL66^120)+EAST+SPAIN+DENS65C+GDPCH60L+AVELF+ENGFRAC+MINING, 
                data=train_fact)
summary(lm_combine)
lm_infos(lm_combine) # 0.00025091

# k means clustering

numeric_columns <- train_num[,4:ncol(train_num)]
scaled_data <- scale(numeric_columns)
kmeans_result <- kmeans(scaled_data, centers = 4, nstart = 30)

par(mfrow=c(1,1))
plot(train_num$FERTLDC1, train_num$GR6096, main = "K = 4", xlab="FERTLDC1", ylab="GR6096", type="n")
text(train_num$FERTLDC1, train_num$GR6096, labels=train_num$COUNTRY, col = rainbow(5, v = 0.8, start = 0, end = 1, alpha = 1)[kmeans_result$cluster])

n=nrow(train_num)
d=68
kt = 1:20
bic = rep(0,20)
aicc = rep(0,20)
for(ii in 1:20) {
  fit = kmeans(scale(numeric_columns[,colnames(numeric_columns)]), centers = ii, nstart = 20) 
  df = d*ii 
  bic[ii] = fit$tot.withinss + log(n)*df
  aicc[ii] = fit$tot.withinss + 2*df*n/(n-df-1)
}

bicsel=which.min(bic) #K=4
aiccsel=which.min(aicc) #K=20


cluster_assignments <- kmeans_result$cluster

train_clusters <- cbind(train_num[,4:ncol(train_num)], clusters=cluster_assignments)
test_clusters <- cbind(test_num[,4:ncol(test_num)], clusters=rep(0, nrow(test_num)))

df1 <- train_clusters[cluster_assignments == 1,] # the west (EUROPE)
df2 <- train_clusters[cluster_assignments == 2,] # asia (EAST)
df3 <- train_clusters[cluster_assignments == 3,] # africa (SAFRICA)
df4 <- train_clusters[cluster_assignments == 4,] # latin america (LAAM)
train_num[train_clusters['clusters']==1,'COUNTRY']
train_num[train_clusters['clusters']==2,'COUNTRY']
train_num[train_clusters['clusters']==3,'COUNTRY']
train_num[train_clusters['clusters']==4,'COUNTRY']

lm1 = lm(GR6096~YRSOPEN+EAST+GDPCH60L+P60+NEWSTATE+FERTLDC1+AVELF, data=df1)
summary(lm1)
lm2 = lm(GR6096~I(YRSOPEN^2)+EAST+P60+FERTLDC1+ECORG+I(PRIGHTS^2)+I(GDPCH60L^2), data=df2)
summary(lm2)
lm3 = lm(GR6096~I(P60^4)+I(GEEREC1^2)+I(MINING^3)+GOVSH61+I(SQPI6090^12)+PRIEXP70+DPOP6090+I(GDPCH60L^2), data=df3)
summary(lm3)
lm4 = lm(GR6096~I(P60^2)+I(H60^2)+I(SQPI6090^2)+OIL+I(DENS65C^20)+I(FERTLDC1^2), data=df4)
summary(lm4)

# GOVSH61+YRSOPEN+RERD+P60+I(MALFAL66^120)+EAST+SPAIN+DENS65C+GDPCH60L+AVELF+ENGFRAC+MINING

model_list <- list(lm1, lm2, lm3, lm4)

# knn and prediction

cluster_pred = kknn(clusters~.,train_clusters,test_clusters,k=15)
tested_clusters <- cbind(test_clusters[,1:ncol(test_clusters)-1], clusters=round(cluster_pred$fitted.values))

test_lm <- function(test_set){
  tot_serror = 0
  for (i in 1:nrow(test_set)) {
    model = model_list[[test_set[i, 'clusters']]]
    pred = predict.lm(model, newdata=test_set)
    tot_serror = tot_serror + (test_set$GR6096-pred)^2
  }
  return (tot_serror/nrow(test_set))
}

mean(test_lm(tested_clusters)) # 0.000763958


