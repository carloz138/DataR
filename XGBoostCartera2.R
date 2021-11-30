#Modelo de XGBOOST de cartera 

#Importar dataset 
dataset = read.csv("Cartera2.csv")
dataset = dataset[,3:13]

library(DMwR2)
library(themis)

#Codificar los factores para la RNA
dataset$Genero = as.numeric(factor(dataset$Genero,
                                   levels = c("H", "M"),
                                   labels = c(1,2)))

dataset$vOpPeriod = as.numeric(factor(dataset$vOpPeriod,
                                      levels = c("Mensual",
                                                 "Catorcenal",
                                                 "Quincenal",
                                                 "Semanal"),
                                      labels = c(1,2,3,4)))


dataset$TipoCartera = as.numeric(factor(dataset$TipoCartera,
                                        levels = c("Adm", "PROP"),
                                        labels = c(1,2)))
str(dataset)
dataset$Clasificacion = factor(dataset$Clasificacion)

table(dataset$Clasificacion)



#Creación de set de datos balanceados 

dataset2 = smote(dataset, var = "Clasificacion", k= 5 , over_ratio =  1)

str(dataset2)
table(dataset2$Clasificacion)
#Guardado de dataset 
write.csv(dataset2, "Dataset2")

dataset3 = read.csv("Dataset2")
dataset3$X = NULL



#Dividir conjuntos de datos en entrenamiento y testing
library(caTools)
set.seed(123)
split = sample.split(dataset3$Clasificacion, SplitRatio = 0.8)
training_set = subset(dataset3, split == TRUE)
testing_set = subset(dataset3, split == FALSE)



#Escalado de variables
training_set[,1:10] = scale(training_set[,1:10])
testing_set[,1:10] = scale(testing_set[,1:10])


#Codificar los factores para la RNA
dataset$Genero = as.numeric(factor(dataset$Genero,
                                   levels = c("H", "M"),
                                   labels = c(1,2)))

dataset$vOpPeriod = as.numeric(factor(dataset$vOpPeriod,
                                      levels = c("Mensual",
                                                 "Catorcenal",
                                                 "Quincenal",
                                                 "Semanal"),
                                      labels = c(1,2,3,4)))


dataset$TipoCartera = as.numeric(factor(dataset$TipoCartera,
                                        levels = c("Adm", "PROP"),
                                        labels = c(1,2)))
table(dataset$TipoCartera)
str(dataset)

#Dividir conjuntos de datos en entrenamiento y testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Clasificacion, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


#Ajustar XGBOOST al conjunto de entrenamiento

library(xgboost)

classifier = xgboost(data = as.matrix(training_set[,-11]),
                     label = training_set$Clasificacion,
                     nrounds = 10)

#Comparar modelo contra conjunto de test
y_pred = predict(classifier, newdata = as.matrix(testing_set[,-11]))
y_pred = (y_pred >= 0.7)
cm = table(testing_set[,11], y_pred)
cm
y_pred
#Evaluacion de rendimiento 
library(caret)
folds = createFolds(training_set$Clasificacion, k=10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x,]
  test_fold = training_set[x,]
  classifier = xgboost(data = as.matrix(training_set[,-11]),
                       label = training_set$Clasificacion,
                       nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[,-11]))
  y_pred = (y_pred >= 0.7)
  cm = table(test_fold[,11], y_pred)
  accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy_sd = sd(as.numeric(cv))
accuracy
accuracy_sd




