import sys
import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from datetime import datetime

# semilla
np.random.seed(3611)

# Generación del nombre del archivo usando la fecha actual
now = datetime.now()
filename = "Resul_model_{:04d}{:02d}{:02d}{:02d}{:02d}.csv".format(
        now.year, now.month, now.day, now.hour, now.minute
    )

# Cargar datos
dataset = pd.read_csv("Gama_secretM .csv")

# Separamos variables predictoras (x) y etiqueta (y)
x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Normalizamos los datos
sc = StandardScaler()
x = sc.fit_transform(x)

# Aplicamos PCA (reducción a 30 componentes)
pca = PCA(n_components=30)
x_t = pca.fit_transform(x)
# Definir métricas a calcular
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
# iniciamos el archivo
with open(filename,'w', encoding='utf-8',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["suerte "])
#funcion de evaluacion
def evalu(modelo,va_x,va_y):
    # Definir métricas a calcular
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    # Ejecutamos validación cruzada de 10 folds
    scores = cross_validate(modelo, va_x, va_y, cv=10, scoring=scoring)
    #Guardamos los resultados en el archivo
    with open(filename,'a', encoding='utf-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        # escribimos nombre del modelo
        writer.writerow(["Modelo ",modelo.__class__.__name__])
        # escribimos el encabezado
        writer.writerow(['Division',
                         'fit_time',
                         'score_time',
                         'test_f1_macro',
                         'test_accuracy',
                         'test_precision_macro',
                         'test_recall_macro'])
        # escribimos las metricas obtenidas
        num_folds=len(scores['fit_time'])
        for i in range(num_folds):
            writer.writerow([i + 1,
                             scores['fit_time'][i],
                             scores['score_time'][i],
                             scores['test_f1_macro'][i],
                             scores['test_accuracy'][i],
                             scores['test_precision_macro'][i],
                             scores['test_recall_macro'][i]
                             ])
    print("listo modelo: ", modelo)

# Creamos los clasificadores con los hiperparámetros indicados
gpc = GaussianProcessClassifier(n_restarts_optimizer=10,
                                max_iter_predict=111,
                                multi_class='one_vs_one')

rf=RandomForestClassifier(n_estimators=270,
                          criterion='gini',
                          max_depth=29,
                          min_samples_split=2,
                          min_samples_leaf=1
                          )
lgre=LogisticRegression(C=2408665484.98762,
                        solver='sag',
                        max_iter=27700000,
                        class_weight='balanced')

clsnn=MLPClassifier(activation='logistic',
                    solver='adam',
                    learning_rate= 'adaptive',
                    random_state=3611,
                    hidden_layer_sizes=(243,222),
                    max_iter=10000,
                    alpha=1e-6
                    ).fit(x_t, y)
mdsvm=svm.SVC( class_weight='balanced',
               kernel='rbf',
               gamma=0.085477695,
               tol=1e-6,
               decision_function_shape='ovo',
               C=976.4451786,
               max_iter= 510885,
               degree=1
               )
nca = NeighborhoodComponentsAnalysis(init='auto',
                                     random_state=3611,
                                     warm_start=True,
                                     n_components=10)
knn = KNeighborsClassifier(n_neighbors=6,
                           algorithm='kd_tree',
                           p=1.07501539)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])

#ejecutamos el cv
evalu(gpc,x_t,y)
evalu(rf,x_t,y)
evalu(lgre,x_t,y)
evalu(clsnn,x_t,y)
evalu(mdsvm,x_t,y)
evalu(nca_pipe,x_t,np.around(y))
# Imprimir el archivo en pantalla
with open(filename, 'r', encoding='UTF8') as f:
    contenido = f.read()
    print(contenido)

