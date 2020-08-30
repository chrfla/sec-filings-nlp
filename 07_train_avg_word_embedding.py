#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Import packages #########

from _settings import main_dir
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import datetime 
import torch
from tpot import TPOTClassifier
from assets.TimeBasedCV import TimeBasedCV
from assets.CfMatrix import make_confusion_matrix
from assets.utils import down_sample_majority_class
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score



####### Define functions #########

def prepare_data_avg_embedding(df, downsample = False):
    
    """
    Title: Learning Word Embeddings from 10-K Filings for Financial NLP Tasks
    Author: Saurabh Sehrawat
    Date: 2019
    Code version: 1.0
    Availability: https://github.com/ssehrawat/10K-word-embeddings
    """
    embed = torch.load(main_dir + 'data/10K-word-embeddings/10k_word_embeddings.tar')
    vocab_to_int = torch.load(main_dir + 'data/10K-word-embeddings/vocab_to_int.tar')
    
   
    df = df.dropna(subset=['diff_text'])

    avg_embd = np.zeros([1, 300], dtype = float)
    embd_matrix = np.zeros([1, 300], dtype = float)
    for index, row in tqdm(df.iterrows()):
        doc = row['diff_text']
        doc = doc.lower()
        doc = doc.split()
        embd_array = np.zeros([1, 300], dtype = float)
        for word in doc:
            if word in vocab_to_int.keys():
                word_emd = embed[vocab_to_int[word]]
                embd_array = np.append(embd_array, word_emd.reshape(1,300), axis=0)
        avg_embd = 	np.average(embd_array[1:], axis=0).reshape(1,300)
        embd_matrix = np.concatenate((embd_matrix, avg_embd), axis=0)
        
    data = pd.DataFrame(embd_matrix[1:])

    sec_filing_date = pd.to_datetime(df['sec_filing_date'], format = '%Y%m%d')
    data['sec_filing_date'] = sec_filing_date
    data['rating_downgrade'] = df['rating_downgrade']

    # Create train & Validation set
    data = data.dropna()
    data = data[data['sec_filing_date'] < datetime.datetime(2019,1,1)]
    
    
    # Create holdoutset
    df_holdout = data[data['sec_filing_date'] > datetime.datetime(2017,12,31)]

    if downsample:
        df_holdout = down_sample_majority_class(df_holdout, 'rating_downgrade')
    
    # Create train & Validation set
    df_train = data[data['sec_filing_date'] <= datetime.datetime(2017,12,31)]

    if downsample:
        df_train = down_sample_majority_class(df_train, 'rating_downgrade')
    
    return df_train, df_holdout, data





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
    

        clf.fit(data_train,target_train.values.ravel())
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
        clf.fit(data_train, target_train.values.ravel())    
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
        clf.fit(data_train,target_train.values.ravel())
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
        clf.fit(data_train, target_train.values.ravel())    
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
df_avg = pd.read_csv( main_dir + 'data/df_diff_sent.csv', sep=',')
df_avg_embd_train, df_avg_embd_holdout, data_avg_embd = prepare_data_avg_embedding(df_avg)

features_avg_embd = df_avg_embd_train.columns.tolist()
features_avg_embd.pop()
target_avg_embd = ['rating_downgrade']


### No resampling - avg. word embedding ###
scores_lr_avg_embd, model_lr_avg_embd, cm_r_avg_embd = model_training_out_of_time(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, LogisticRegression(), downsample = False)
scores_rfc_avg_embd, model_rfc_avg_embd, cm_rfc_avg_embd = model_training_out_of_time(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, RandomForestClassifier(), downsample = False)
scores_ada_avg_embd, model_ada_avg_embd, cm_ada_avg_embd = model_training_out_of_time(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, AdaBoostClassifier(), downsample = False)
scores_svc_avg_embd, model_svc_avg_embd, cm_svc_avg_embd = model_training_out_of_time(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, SVC(), downsample = False)
scores_tpot_avg_embd, model_tpot_avg_embd, cm_tpot_avg_embd = model_training_out_of_time_tpot(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, downsample = False)


### Under-sampling - avg. word embedding ###
scores_lr_avg_embd, model_lr_avg_embd, cm_r_avg_embd = model_training_out_of_time(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, LogisticRegression(), downsample = True)
scores_rfc_avg_embd, model_rfc_avg_embd, cm_rfc_avg_embd = model_training_out_of_time(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, RandomForestClassifier(), downsample = True)
scores_ada_avg_embd, model_ada_avg_embd, cm_ada_avg_embd = model_training_out_of_time(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, AdaBoostClassifier(), downsample = True)
scores_svc_avg_embd, model_svc_avg_embd, cm_svc_avg_embd = model_training_out_of_time(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, SVC(), downsample = True)
scores_tpot_avg_embd, model_tpot_avg_embd, cm_tpot_avg_embd = model_training_out_of_time_tpot(df_avg_embd_train, df_avg_embd_holdout, target_avg_embd, features_avg_embd, downsample = True)
