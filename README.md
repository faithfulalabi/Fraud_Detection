# Fraud_Detection: Project Project Overview

* Created a model that classifies if a series of credit card transactions are fraudulent or not fraudulent. For a fraudulent transaction it predicts a 1, and a 0 for a non fraudulent transaction. (Recall ~ 82%, Precision ~ 95%, Accuracy ~ 99%).
* Dataset is from Kaggle and can be found [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).
* Performed Unsampling technique to balance out the target variable data.
* Used a Random Forest Classifier, performed some parameter adjustments and optimized it using RandomizedSearchCV for the best parameter combinations.

## __Motivation__: 

I chose this particular problem to solve because I wanted to improve my classification technique. For every new project I work on, I try to improve my approach,learn a new library, and better my modeling technique.


## Dataset Feature Explaination

There are no feature explanations due to confidentiality issues. Features contains only numerical input labeled V1-V28.  

## Data Prepping

The only data preparation I had to do for this dataset was fixing the imbalance of the amount of Fraudulent transaction to none fraudulent transactions. (See the before and after plots below).

![alt text](https://github.com/faithfulalabi/Fraud_Detection/blob/main/Imbalanced_plot_of_data.png?raw=true)

![alt text](https://github.com/faithfulalabi/Fraud_Detection/blob/main/Balanced_plot_of_data.png?raw=true)



## Model Building

I first split the data into training and testing sets with a test size of 20%. I then scaled the data and tested a Logistic, KNeighborsClassifier, SVC, XGBClassifier, DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier, and ExtraTreesClassifier model on the training dataset using both accuracy and recall as my scoring metric (See the results of model performance on training set down below). I chose recall as my main scoring metric for this problem because the business will care more reducing the amount of false negatives, meaning transactions the model predicts as none fraudulent but were actually fraudulent. 

![alt text](https://github.com/faithfulalabi/Fraud_Detection/blob/main/Training_accuracy_Scores.png?raw=true)
![alt text](https://github.com/faithfulalabi/Fraud_Detection/blob/main/Training_recall_Scores.png?raw=true)

Ultimately I chose the Random Forest model because of it's robustness to dealing with outliers, ability to run efficiently on a large dataset, and lower risk of overfitting. I then used a RandomizedSearchCV to tune these parameters: max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, n_estimators for the best combination using recall as my scoring metric.



## Model Performance

* RandomForestClassifier:
    Training Mean Accuracy score: 1.0
    Testing Mean Accuracy score: 0.9996137776061234
    
     precision    recall  f1-score   support

           0       1.00      1.00      1.00     56862
           1       0.95      0.82      0.88       100
 
![alt text](https://github.com/faithfulalabi/Fraud_Detection/blob/main/RandomForestClassifier_Confusion&#32;Matrix.png?raw=true)

