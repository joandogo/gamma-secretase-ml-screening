# -*- coding: utf-8 -*-
"""
Created on 05/01/2025

@author: j.dgortaire
"""
import sys
import csv
import optuna
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sklearn.ensemble
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# semilla
np.random.seed(3611)
sampler = optuna.samplers.TPESampler(seed=3611)

# archivos

now= datetime.now()
filename= "rslt_RF_{}_{}{:02d}{:02d}{:02d}{:02d}.csv".format(sys.argv[1],now.year,now.month,now.day,now.hour,now.minute)
# Datos
dataset = pd.read_csv("BACE25.csv")

# Separamos conjunto de datos
x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# normalizar datos

sc = StandardScaler()
x = sc.fit_transform(x)

# pca
pca = PCA(n_components=30)
x_t = pca.fit_transform(x)

a = []
count = 0


def objective(trial):
    rfcapar = {"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    "criterion": trial.suggest_categorical("criterion", ['gini','entropy','log_loss']),
    "max_depth": trial.suggest_int("max_depth", 5, 30),
    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }

    # RF
    rfmodelo = RandomForestClassifier(**rfcapar)
    return sklearn.model_selection.cross_val_score(
        rfmodelo, x_t,y, n_jobs=-1,  cv=10, scoring='f1_macro', error_score=-1).mean()


with open(filename, 'a+', encoding='UTF8') as rsult:
    writer = csv.writer(rsult)
    writer.writerow(['RF \n f1_score: ', "Best hyperparameters"])
    for i in range(300):
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=100)
        trial = study.best_trial
        writer.writerow([trial.value, trial.params])
        if i % 5 == 0:
            rsult.flush()
        print('f1_score: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params), i)
        #fig = optuna.visualization.plot_optimization_history(study)
        #fig.write_image(file="./history_{}{}.png".format(filename, i), format="png")
        #fig = optuna.visualization.plot_slice(study)
        #fig.write_image(file="./slice_{}{}.png".format(filename, i))
        #fig = optuna.visualization.plot_contour(study, params=['init', 'n_components', 'n_neighbors', 'p'])
        #fig.write_image(file="./contour_{}{}.png".format(filename, i))
    rsult.flush()
