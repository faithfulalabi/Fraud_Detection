# packages for data loading, data analysis, and data preparation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# packages for modek evaluation and classification models
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
from tqdm import tqdm

def show_model_accuracy_scores(models,x,y):
   # stores the name of model and results
    names =[]
    results =[]
    scoring = 'accuracy' # scoring metric
    seed = 7
   # loops through each model, grabs the name, performs kflod and uses cross val score to score model
    for name,model in tqdm(models):
        kflod = KFold(n_splits=5,random_state=seed,shuffle=True)
        cv_results = cross_val_score(model,x,y,cv=kflod,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print(f'Model {name} has an average accuracy score of: {cv_results.mean()}')
        
    # plot the model results 
    sns.set_style("darkgrid")
    plt.figure(figsize=(10,8),dpi=150)
    axes = sns.boxplot(data=results) # left, bottom, width, height (range 0 to 1)
    ind = np.arange(len(names))
    axes.set_xticks(ind)
    axes.set_xticklabels(names)
    plt.title('Training Accuracy Errors')
    plt.show()

def show_model_recall_scores(models,x,y):
   # stores the name of model and results
    names =[]
    results =[]
    scoring = 'recall' # scoring metric
    seed = 7
   # loops through each model, grabs the name, performs kflod and uses cross val score to score model
    for name,model in tqdm(models):
        kflod = KFold(n_splits=5,random_state=seed,shuffle=True)
        cv_results = cross_val_score(model,x,y,cv=kflod,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print(f'Model {name} has an average accuracy score of: {cv_results.mean()}')
        
    # plot the model results 
    sns.set_style("darkgrid")
    plt.figure(figsize=(10,8),dpi=150)
    axes = sns.boxplot(data=results) # left, bottom, width, height (range 0 to 1)
    ind = np.arange(len(names))
    axes.set_xticks(ind)
    axes.set_xticklabels(names)
    plt.title('Training Recall Errors')
    plt.show()