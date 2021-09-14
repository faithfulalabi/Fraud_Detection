# packages for data loading, data analysis, and data preparation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# packages for model evaluation and classification models
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix

# package to check the progress of our models
from tqdm import tqdm


def show_model_scores(models,x,y,scoring):
    """
    Inputs
    1. classification models eg: LogisticRegression(),KNeighborsClassifier() etc.
    2. training x data
    3. training y data
    4. scoring metric as a string input eg: accuracy, percision, Recall 
    """
   # stores the name of model and results
    names =[]
    results =[]
    seed = 7
   # loops through each model, grabs the name, performs kflod and uses cross val score to score model
    for name,model in tqdm(models):
        kflod = KFold(n_splits=5,random_state=seed,shuffle=True)
        cv_results = cross_val_score(model,x,y,cv=kflod,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print(f'Model {name} has an average {scoring} score of: {cv_results.mean()}')
        
    # plot the model results 
    sns.set_style("darkgrid")
    plt.figure(figsize=(10,8),dpi=150)
    axes = sns.barplot(data=results) # left, bottom, width, height (range 0 to 1)
    ind = np.arange(len(names))
    axes.set_xticks(ind)
    axes.set_xticklabels(names)
    plt.title(f'Training {scoring} Scores')
    plt.savefig(f'Training_{scoring}_Scores.png')
    plt.show()

def evaluate_model(model,x_train,y_train,x_test,y_test):
    '''
    Inputs
    1. classification model eg: LogisticRegression(),KNeighborsClassifier() etc.
    2. training x data
    3. training y data
    4. testing x data
    5. testing y data
    '''
    # fits the model to the training data,and stores the model name
    model.fit(x_train,y_train)
    model_name= type(model).__name__
    
    # scores the train and test datasets
    train_score=model.score(x_train,y_train)
    test_score=model.score(x_test,y_test)
    print (f"Training Mean Accuracy score: {train_score}\nTesting Mean Accuracy score: {test_score}")
    
    # makes prediction based off the test set
    y_pred=model.predict(x_test)
    print("Prediction completed.")
    
    #plot confusion Matrix
    plt.figure(figsize=(10,8),dpi=150)
    plot_confusion_matrix(model,x_test,y_test,cmap=plt.cm.Blues,
                                   display_labels=['No_Fraud=0','Fraud=1'])
    plt.title(f'Confusion Matrix for {model_name} Model')
    plt.savefig(f'{model_name}_Confusion Matrix.png')
    plt.show()
    print(classification_report(y_test,y_pred))