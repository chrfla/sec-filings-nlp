#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Import packages #########

import re
import numpy as np
import pandas as pd
import datetime 
from tpot import TPOTClassifier
from assets.TimeBasedCV import TimeBasedCV
from assets.CfMatrix import make_confusion_matrix
from assets.utils import calculate_previous_quantile, down_sample_majority_class
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from _settings import main_dir
import warnings
warnings.filterwarnings('ignore')



####### Define functions #########

def prepare_data_sim(df, downsample = False):

    # Clean and convert data
    df_sim_clean = df.dropna(subset=['Sim_Cosine', 'Sim_Jaccard'])
    sec_filing_date = pd.to_datetime(df_sim_clean['sec_filing_date'], format = '%Y%m%d')
    df_sim_clean['sec_filing_date'] = sec_filing_date
    df_sim_clean = df_sim_clean[df_sim_clean['sec_filing_date'] < datetime.datetime(2019,1,1)]
    df_sim_clean = df_sim_clean[df_sim_clean['sec_filing_date'] > datetime.datetime(2006,12,31)]
    
    # Create train & Validation set
    df_sim_clean = df_sim_clean[['rating_downgrade', 'sec_year', 'Sim_Cosine', 'Sim_Jaccard', 'default', 'sec_filing_date']]
    df_sim_clean = df_sim_clean.dropna()
    df_sim_clean = calculate_previous_quantile(df_sim_clean, 5, ['Sim_Jaccard', 'Sim_Cosine'])
    df_sim_clean = df_sim_clean.dropna()
    
    # Create holdoutset
    df_holdout = df_sim_clean[df_sim_clean['sec_filing_date'] > datetime.datetime(2017,12,31)]
    if downsample:
        df_holdout = down_sample_majority_class(df_holdout, 'rating_downgrade')
    
    # Create train & Validation set
    df_train = df_sim_clean[df_sim_clean['sec_filing_date'] <= datetime.datetime(2017,12,31)]
    if downsample:
        df_train = down_sample_majority_class(df_train, 'rating_downgrade')
    
    return df_train, df_holdout
 
    

def model_training_out_of_time(df, holdout, target, features, algo, standard = True, show_holdout = False, downsample = False):
      
    X = df[features]
    y = df[target].astype('bool')
    X_holdout = holdout[features].drop('sec_filing_date', axis=1)
    y_holdout = holdout[target].astype('bool')

    if standard:
        clf = make_pipeline(StandardScaler(), algo)
    else:
        clf = algo
        
    alog_name = str(clf.steps[1][1]).split('(')[0]
    alog_name = " ".join(re.findall('[A-Z][^A-Z]*',alog_name))  
    if alog_name == 'S V C':
        alog_name = 'SVM Classifier'
    print('### ' + alog_name + ' ###')
    
    scores = {'acc': [], 'f1': []}
    cf_matrix_val = np.zeros((2,2), dtype=np.int)
    
    tbcv = TimeBasedCV(train_period=3, test_period=1, freq='years')
    tbcv_folds = tbcv.split(df,  validation_split_date = datetime.date(2008,12,31), date_column = 'sec_filing_date')
    for train_index, test_index in tbcv_folds:
        
        if downsample: 
            df_kfold = down_sample_majority_class(df, 'rating_downgrade')
            df_kfold_index = df_kfold.index.tolist()
            train_index = [idx for idx in list(train_index) if idx in df_kfold_index]
        
        data_train   = X.loc[train_index].drop('sec_filing_date', axis=1)
        target_train = y.loc[train_index]
    
        data_test    = X.loc[test_index].drop('sec_filing_date', axis=1)
        target_test  = y.loc[test_index]
    
        clf.fit(data_train,target_train)
        preds = clf.predict(data_test)
    
        # accuracy for the current fold only    
        score = clf.score(data_test,target_test)
        f1 = f1_score(target_test, preds)
        
        cf_matrix_val += confusion_matrix(target_test, preds)
        scores['acc'].append(score)
        scores['f1'].append(f1)
    

    print("Cross Validation Score: " + str(sum(scores['acc']) / len(scores['acc'])))     

    
    
    if show_holdout:
        # Test model trained on last three years on holdout data
        
        frames = [test_index for train_index, test_index in tbcv_folds[-3:]]
        frames = [item for sublist in frames for item in sublist]
        data_train   = X.loc[frames].drop('sec_filing_date', axis=1)
        target_train = y.loc[frames]
        clf.fit(data_train, target_train)    
        holdout_preds = clf.predict(X_holdout)
        cf_matrix = confusion_matrix(y_holdout, holdout_preds)
        
        print("Holdout Score: " + str(clf.score(X_holdout, y_holdout)))     
        print('\n')
        # Visualize confusion matrix for holdout data
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['No Downgrade', 'Downgrade']
        make_confusion_matrix(cf_matrix, 
                          group_names=labels,
                          categories=categories,
                          cbar = False,
                          title='Confusion Matrix: ' + alog_name,
                          figsize= (10,10))
    else:
        #Visualize confusion matrix for cross-val data
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['No Downgrade', 'Downgrade']
        make_confusion_matrix(cf_matrix_val, 
                          group_names=labels,
                          categories=categories,
                          cbar = False,
                          title='Confusion Matrix: ' + alog_name,
                          figsize= (10,10))
    
    return scores, clf, cf_matrix_val



def model_training_out_of_time_tpot(df, holdout, target, features, show_holdout = False, downsample = False):
    
    X = df[features]
    y = df[target].astype('bool')
    X_holdout = holdout[features].drop('sec_filing_date', axis=1)
    y_holdout = holdout[target].astype('bool')



    scores = {'acc': [], 'f1': []}
    cf_matrix_val = np.zeros((2,2), dtype=np.int)
    tbcv = TimeBasedCV(train_period=3, test_period=1, freq='years')
    tbcv_folds = tbcv.split(df,  validation_split_date = datetime.date(2008,12,31), date_column = 'sec_filing_date')
    for train_index, test_index in tbcv_folds:
        
        if downsample: 
            df_kfold = down_sample_majority_class(df, 'rating_downgrade')
            df_kfold_index = df_kfold.index.tolist()
            train_index = [idx for idx in list(train_index) if idx in df_kfold_index]
    
        data_train   = X.loc[train_index].drop('sec_filing_date', axis=1)
        target_train = y.loc[train_index]
    
        data_test    = X.loc[test_index].drop('sec_filing_date', axis=1)
        target_test  = y.loc[test_index]
    

        clf = TPOTClassifier(generations = 5, population_size = 50, verbosity = 2, max_time_mins = 5)
        clf.fit(data_train,target_train)
        preds = clf.predict(data_test)
    
        # accuracy for the current fold only    
        score = clf.score(data_test,target_test)
        f1 = f1_score(target_test, preds)
        
        cf_matrix_val += confusion_matrix(target_test, preds)
        scores['acc'].append(score)
        scores['f1'].append(f1)
    

    print("Cross Validation Score: " + str(sum(scores['acc']) / len(scores['acc'])))     

    if show_holdout:
    
    # Test model trained on last three years on holdout data
    
        frames = [test_index for train_index, test_index in tbcv_folds[-3:]]
        frames = [item for sublist in frames for item in sublist]
        data_train   = X.loc[frames].drop('sec_filing_date', axis=1)
        target_train = y.loc[frames]
        clf.fit(data_train, target_train)    
        holdout_preds = clf.predict(X_holdout)
        cf_matrix = confusion_matrix(y_holdout, holdout_preds)
        
        print("Holdout Score: " + str(clf.score(X_holdout, y_holdout)))     
        print('\n')
        # Visualize confusion matrix for holdout data
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['No Downgrade', 'Downgrade']
        make_confusion_matrix(cf_matrix, 
                          group_names=labels,
                          categories=categories,
                          cbar = False,
                          title='Confusion Matrix: TPOT',
                          figsize= (10,10))
    else:
        #Visualize confusion matrix for cross-val data
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['No Downgrade', 'Downgrade']
        make_confusion_matrix(cf_matrix_val, 
                          group_names=labels,
                          categories=categories,
                          cbar = False,
                          title='Confusion Matrix: TPOT',
                          figsize= (10,10))
        
        
    return scores, clf, cf_matrix_val




####### Execute code below #########

# Import dataset
df_sim = pd.read_csv(main_dir + 'data/df_sim_cosine.csv', sep=',')
df_sim_train, df_sim_holdout = prepare_data_sim(df_sim)
features_sim = ['Sim_Cosine', 'sec_filing_date', 'quant_Sim_Cosine']
#features_sim = ['Sim_Jaccard', 'sec_filing_date',  'quant_Sim_Jaccard']
target_sim = ['rating_downgrade']

### No resampling - Similarity Score ###
scores_lr_sim, model_lr_sim, cm_lr_sim = model_training_out_of_time(df_sim_train, df_sim_holdout, target_sim, features_sim, LogisticRegression(), downsample = False)
scores_rfc_sim, model_rfc_sim, cm_rfc_sim = model_training_out_of_time(df_sim_train, df_sim_holdout, target_sim, features_sim, RandomForestClassifier(max_depth=10), downsample = False)
scores_ada_sim, model_ada_sim, cm_ada_sim = model_training_out_of_time(df_sim_train, df_sim_holdout, target_sim, features_sim, AdaBoostClassifier(n_estimators=5), downsample = False)
scores_svc_sim, model_svc_sim, cm_svc_sim = model_training_out_of_time(df_sim_train, df_sim_holdout, target_sim, features_sim, SVC(), downsample = False)
scores_tpot_sim, model_tpot_sim, cm_tpot_sim = model_training_out_of_time_tpot(df_sim_train, df_sim_holdout, target_sim, features_sim, downsample = False)

### Under-sampling - Similarity Score ###
scores_lr_sim, model_lr_sim, cm_lr_sim = model_training_out_of_time(df_sim_train, df_sim_holdout, target_sim, features_sim, LogisticRegression(), downsample = True)
scores_rfc_sim, model_rfc_sim, cm_rfc_sim = model_training_out_of_time(df_sim_train, df_sim_holdout, target_sim, features_sim, RandomForestClassifier(max_depth=10), downsample = True)
scores_ada_sim, model_ada_sim, cm_ada_sim = model_training_out_of_time(df_sim_train, df_sim_holdout, target_sim, features_sim, AdaBoostClassifier(n_estimators=5), downsample = True)
scores_svc_sim, model_svc_sim, cm_svc_sim = model_training_out_of_time(df_sim_train, df_sim_holdout, target_sim, features_sim, SVC(), downsample = True)
scores_tpot_sim, model_tpot_sim, cm_tpot_sim = model_training_out_of_time_tpot(df_sim_train, df_sim_holdout, target_sim, features_sim, downsample = True)
