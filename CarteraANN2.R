#Modelo de red Neuronal de cartera 

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



#Estructura del conjunto de training
table(training_set2$Clasificacion)
str(training_set2$TipoCartera)

#Crear red neuronal
library(h2o)
h2o.init(nthreads = -1)

classifier = h2o.deeplearning(x =1:10 ,y="Clasificacion",
                              training_frame = as.h2o(training_set),     
                              activation = "Rectifier",
                              nfolds = 5,
                              standardize = TRUE,
                              hidden = c(10,10),
                              epochs = 100,
                              train_samples_per_iteration = -2,
                              set.seed(122))

#Prediccion de resultados con el conjunto de testing

prob_pred = h2o.predict(classifier, newdata = as.h2o(testing_set[,-11]))
prob = as.vector(prob_pred)
y_pred = ifelse(prob_pred > 0.7,1,0)
y_pred = as.vector(y_pred)
cm = table(testing_set[,11], y_pred)
cm
table(testing_set[,11])
#Cerrar conexion con el servidor
h2o.shutdown()
y

dataset$Clasificacion = factor(dataset$Clasificacion)
dataframe = as.data.frame(dataset)




