##################################################
###
###   BNP Paribas Cardif Claims Management
###
###    Single model: xgboost (with CV)
###
##################################################

# imports
import csv
import numpy as np
import pandas as pd
import xgboost as xgb
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
#import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def cleanData(train, test):
    target = train['target']
    toDrop = ['v22', 'v112', 'v125', 'v74', 'v1', 'v110', 'v47']
    print 'Drop features:', toDrop
    trainDrop = ['ID', 'target']
    trainDrop.extend(toDrop)
    testDrop = ['ID']
    testDrop.extend(toDrop)
    train = train.drop(trainDrop, axis=1)
    test = test.drop(testDrop, axis=1) # test = test.drop(['ID','v22'], axis=1)
    
    for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()): # Iterator over (column name, Series) pairs
        if train_series.dtype == 'O':
            #for objects: factorize: to convert Object/String/Category to 0-based int value (index is -1 if None!!)
            #The pandas factorize function assigns each unique value in a series to a sequential, 0-based index, and calculates which index each series entry belongs to.
            train[train_name], tmp_indexer = pd.factorize(train[train_name])
            test[test_name] = tmp_indexer.get_indexer(test[test_name])
        else:
            #for int or float: fill NaN
            tmp_len = len(train[train_series.isnull()])
            if tmp_len>0:
                train.loc[train_series.isnull(), train_name] = train_series.median() #train_series.mean() #
            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                test.loc[test_series.isnull(), test_name] = train_series.median() #train_series.mean() #
    return train, target, test
    

def modelfit(alg, train, target, test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgboost_params = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train.values, label=target.values)
        xgtest = xgb.DMatrix(test.values)
        watchlist = [(xgtrain, 'train')] # Specify validations set to watch performance
        cvresult = xgb.cv(xgboost_params, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, early_stopping_rounds=early_stopping_rounds) #metrics='auc',show_progress=False
        alg.set_params(n_estimators=cvresult.shape[0])
    
    # Fit the algorithm on the data
    alg.fit(train, target, eval_metric='auc')

    # Predict training set:
    train_preds = alg.predict(train)
    train_predprob = alg.predict_proba(train)[:,1]
    
    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(target.values, train_preds)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(target, train_predprob)

    # Make a prediction:
    print('Predicting......')
    test_predprob = alg.predict_proba(test)[:,1]

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    #plt.show()
    return test_predprob


if __name__ == "__main__":
    print('Start:')
    print
    print('Step 1: Load Data')
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    ids = test['ID'].values # generate for saving resullts
    print train.shape, test.shape
    print train['target'].value_counts()
    print
    print('Step 2: Clean Data')
    train, target, test = cleanData(train, test)
    print
    print('Step 3: Train the model')
    xgb1 = XGBClassifier(
                     learning_rate =0.1,
                     n_estimators=5, #2000, #5, #1000,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0.1,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='binary:logistic',
                     nthread=4,
                     scale_pos_weight=1,
                     seed=27,
                     silent=1 )
    test_predprob = modelfit(xgb1, train, target, test)
    print
    print('Step 4: Save results')
    # Save results
    predictions_file = open("xgboost_result.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ID", "PredictedProb"])
    open_file_object.writerows(zip(ids, test_predprob))
    predictions_file.close()
    print
    plt.show()
    print('Done!')
    

