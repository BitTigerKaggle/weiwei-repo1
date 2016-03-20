##################################################
###
###   BNP Paribas Cardif Claims Management
###
###    Single model: xgboost (without CV)
###
##################################################

# imports
import csv
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

def cleanData(train, test):
    target = train['target']
    #toDrop = ['v22', 'v112', 'v125', 'v74', 'v1', 'v110', 'v47']
    #print 'Drop features:', toDrop
    trainDrop = ['ID', 'target']
    #trainDrop.extend(toDrop)
    testDrop = ['ID']
    #testDrop.extend(toDrop)
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


def fitModel(train, target, test):
    xgboost_params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",#"auc",
        "eta": 0.01, #0.005,# 0.06, #step size shrinkage used in update to prevents overfitting.
        "min_child_weight": 3,# minimum sum of instance weight(hessian) needed in a child.
        "subsample": 0.8, #0.75, #[default=1] subsample ratio of the training instance.
        "colsample_bytree": 0.7, #0.68, #[default=1] subsample ratio of columns when constructing each tree.
        "max_depth": 10,# Maximum delta step we allow each tree's weight estimation to be.
        "seed": 0,
        "silent": 0 # [default=0] 0 means printing running messages, 1 means silent mode.
    }
    
    xgtrain = xgb.DMatrix(train.values, label=target.values)
    xgtest = xgb.DMatrix(test.values)

    watchlist = [(xgtrain, 'train')] # testing dataset only for prediction, not training
    boost_round = 5 #5000 #2000 #1800
    bst = xgb.train(xgboost_params, xgtrain, num_boost_round=boost_round, evals=watchlist, verbose_eval=True, early_stopping_rounds=10) #eval_metric='error'?

    # Make prediction
    print('Predicting...')
    train_preds = bst.predict(xgtrain, ntree_limit=bst.best_iteration)
    trainLogloss = log_loss(target.values, train_preds)
    print 'Primary Score: ', trainLogloss
    test_predprob = bst.predict(xgtest, ntree_limit=bst.best_iteration)
    
    # Saving model After training, you can save model and dump it out.
    bst.save_model('0001.model')
    # dump model
    bst.dump_model('dump.raw.txt')

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
    test_predprob = fitModel(train, target, test)
    print
    print('Step 4: Save results')
    # Save results
    predictions_file = open("xgboost_result.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ID", "PredictedProb"])
    open_file_object.writerows(zip(ids, test_predprob))
    predictions_file.close()
    print
    print('Done!')
    

