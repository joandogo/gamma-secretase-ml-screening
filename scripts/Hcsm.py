# -*- coding: utf-8 -*-
"""
2025

@author: jose7
"""


import optuna
import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn import svm
import sklearn.ensemble
import sklearn.model_selection
import os
import plotly
from datetime import datetime
# archivos
now= datetime.now()
filename= "rslt_HannGSM_{}_op1_{}{:02d}{:02d}{:02d}{:02d}.csv".format("CSM",now.year,now.month,now.day,now.hour,now.minute)
#semilla
np.random.seed(3611)
sampler = optuna.samplers.TPESampler(seed=1111)  
#inicio variables

# importar datos
dataset = pd.read_csv('BACE25.csv')

#Separamos conjunto de datos
x = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

#normalizar datos
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x = sc.fit_transform(x)

#pca
pca = PCA(n_components=30)
x_t = pca.fit_transform(x)


a= []
count=0

def objective(trial):
    
    C = trial.suggest_uniform ("C", 1, 1000)
    degree=trial.suggest_int("degree",1,7)
    kernel=trial.suggest_categorical("kernel", ['poly', 'rbf', 'sigmoid'])
    gamma=trial.suggest_uniform("gamma", 0.01,100)
    max_iter=trial.suggest_int("max_iter",1000,100000)
    #SVM
    clsvm=svm.SVC( class_weight='balanced',
                   kernel=kernel,
                  gamma=gamma,
                  tol=1e-6,
                  decision_function_shape='ovo',
                  C=C,
                  max_iter= max_iter,
                  degree=degree
                  )
    
    return sklearn.model_selection.cross_val_score(
        clsvm, x_t, y, n_jobs=-1, cv=10,scoring='f1_macro',error_score=0.5).mean()
    

with open(filename, 'a+', encoding='UTF8') as rsult:
  writer = csv.writer(rsult)
  writer.writerow(['CSM\n f1_score: ',"Best hyperparameters"])
  for i in range(300):
    print("Interaccion",i)
    study = optuna.create_study(direction="maximize",sampler=sampler,study_name="study{:03d}".format(i))
    study.optimize(objective, n_trials=200)
    trial = study.best_trial
    writer.writerow([ trial.value, trial.params])
    if i%3 == 0:
      rsult.flush()
    print('f1_score: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    """fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(file="./history_{}{}.png".format(filename,i),format="png")
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(file="./slice_{}{}.png".format(filename,i))
    fig = optuna.visualization.plot_contour(study, params=['activation', 'solver','learning_rate','n_layers'])
    fig.write_image(file="./contour_{}{}.png".format(filename,i)) """
  rsult.flush()
        

