import sys
import csv
import optuna
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sklearn.ensemble
import sklearn.model_selection
from sklearn.pipeline import Pipeline
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from datetime import datetime
import plotly
#semilla
np.random.seed(3611)
sampler = optuna.samplers.TPESampler(seed=3611)  
#inicio variables

# archivos
now= datetime.now()
filename= "rslt_Hknn_{}_op1_{}{:02d}{:02d}{:02d}{:02d}.csv".format(sys.argv[1],now.year,now.month,now.day,now.hour,now.minute)
#Datos
dataset = pd.read_csv('BACE25.csv')

#Separamos conjunto de datos
x = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

#normalizar datos
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#pca
pca = PCA(n_components=30)
x_t = pca.fit_transform(x)


a= []
count=0

def objective(trial):
    
    ncapar = {"init":trial.suggest_categorical("init",['auto', 'pca',  'identity', 'random']),
              "tol":1e-6,
              "random_state":3611,
              "warm_start":True,
              "n_components":trial.suggest_int("n_components", 2, 10)
              
              }
    knnpar ={"n_neighbors":trial.suggest_int("n_neighbors", 2,10),
             "algorithm":trial.suggest_categorical("algorithm",[sys.argv[1]]),
             "p":trial.suggest_float("p",1,5)
             
             }
        #knn
    nca = NeighborhoodComponentsAnalysis(**ncapar)
    knn = KNeighborsClassifier(**knnpar )
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    return sklearn.model_selection.cross_val_score(
        nca_pipe, x, np.around(y), n_jobs=-1, cv=10,scoring='f1_macro',error_score=-1).mean()
with open(filename, 'a+', encoding='UTF8') as rsult:
  writer = csv.writer(rsult)
  writer.writerow(['KNN\n f1_score: ',"Best hyperparameters"])
  for i in range(200):
    study = optuna.create_study(direction="maximize",sampler=sampler)
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    writer.writerow([ trial.value, trial.params])
    if i%5 == 0:
      rsult.flush()
    print('f1_score: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params),i)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(file="./history_{}{}.png".format(filename,i),format="png")
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(file="./slice_{}{}.png".format(filename,i))
    fig = optuna.visualization.plot_contour(study, params=['init', 'n_components','n_neighbors','p'])
    fig.write_image(file="./contour_{}{}.png".format(filename,i)) 
  rsult.flush()
    
