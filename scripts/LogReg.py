import sys
import csv
import optuna
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sklearn.ensemble
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# semilla
np.random.seed(3611)
sampler = optuna.samplers.TPESampler(seed=3611)

# archivos

now= datetime.now()
filename= "rslt_logR_{}_{}{:02d}{:02d}{:02d}{:02d}.csv".format(sys.argv[1],now.year,now.month,now.day,now.hour,now.minute)
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
    logrcapar = {
    "C":trial.suggest_float("C",1e-10,1e10),
    "solver":trial.suggest_categorical("solver",['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']),
    "max_iter":trial.suggest_int("max_iter", 100,1000),
    "class_weight":trial.suggest_categorical("class_weight",['balanced',None]),}
    # LogR
    logrmodelo = LogisticRegression(**logrcapar)
    return sklearn.model_selection.cross_val_score(
        logrmodelo, x_t,y, n_jobs=-1,  cv=10, scoring='f1_macro', error_score=-1).mean()


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
