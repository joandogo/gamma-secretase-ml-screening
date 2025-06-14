# -*- coding: utf-8 -*-
"""
2025
@author: jose7
"""
import sys
import csv
import optuna
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import sklearn.ensemble
import sklearn.model_selection
from sklearn.pipeline import Pipeline
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from datetime import datetime

#import plotly
#plotly.io.orca.config.executable = '/home/ulc/co/jdg/bin/orca'
#plotly.io.orca.config.save()

now= datetime.now()
filename= "rslt_total3_{}_op1_{}{:02d}{:02d}{:02d}{:02d}.csv".format(sys.argv[1],now.year,now.month,now.day,now.hour,now.minute)
#op1 elimino ativation ('identity', 'logistic'), learning_rate('invscaling',)
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
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#pca
pca = PCA(n_components=50)
x_t = pca.fit_transform(x)


a= []
count=0

def objective(trial):
    
    n_layers = trial.suggest_int('n_layers', 1, 6)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 200+200/(i+1)))
        
    annpar = {
              'activation':trial.suggest_categorical("activation",[ 'relu', 'tanh','logistic']),
              'solver':trial.suggest_categorical('solver',[sys.argv[1]]),
              #'solver':trial.suggest_categorical('solver',['lbfgs', 'sgd', 'adam']),              
              'learning_rate':trial.suggest_categorical('learning_rate',['invscaling',  'adaptive']),
              'random_state':1111,
              'hidden_layer_sizes':tuple(layers),
              'max_iter':10000,
              'alpha':5e-6}
    clsnn=MLPClassifier(**annpar).fit(x_t,y)
    
    
    return sklearn.model_selection.cross_val_score(
        clsnn, x_t, y, n_jobs=-2, cv=10,scoring='f1_macro',error_score=-1).mean()

with open(filename, 'a+', encoding='UTF8') as rsult:
    writer = csv.writer(rsult)
    writer.writerow(['ANN\n f1_score: ',"Best hyperparameters"])
    
    for i in range(100):
        print("Interaccion",i)
        
        study = optuna.create_study(direction="maximize",sampler=sampler, study_name="study{:03d}".format(i))
        study.optimize(objective, n_trials=100)
        trial = study.best_trial
        writer.writerow([ trial.value, trial.params])
        if i%5 == 0:
          rsult.flush()
        print('f1_score: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))
        """    
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(file="./history_{}{}.png".format(filename,i),format="png")

        fig = optuna.visualization.plot_slice(study)
        fig.write_image("./slice_{}{}.png".format(filename,i))
        fig = optuna.visualization.plot_contour(study, params=['activation', 'solver','learning_rate','n_layers'])
        fig.write_image("./contour_{}{}.png".format(filename,i))
        """
    rsult.flush()
