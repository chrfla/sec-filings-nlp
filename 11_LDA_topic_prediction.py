#!/usr/bin/env python3
# -*- coding: utf-8 -*-


####### Import packages #########

import nltk;
nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
import pickle
import datetime
from tqdm import tqdm
from assets.TimeBasedCV import TimeBasedCV
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
import spacy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['may', 'could', 'business', 'result', 'affect', 'include'])
nlp = spacy.load('en', disable=['parser', 'ner'])
from tpot import TPOTClassifier
from assets.CfMatrix import make_confusion_matrix
from assets.utils import down_sample_majority_class
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from _settings import main_dir, lda_data_dir


dest_all_model = {'1': ('LDA_model/2007_2008/lda_2007_2008.model',
                 'LDA_model/2009/lda_2009.model'),
                '2': ('LDA_model/2007_2009/lda_2007_2009.model',
                 'LDA_model/2010/lda_2010.model'),
                '3': ('LDA_model/2007_2010/lda_2007_2010.model',
                 'LDA_model/2011/lda_2011.model'), 
                '4': ('LDA_model/2009_2011/lda_2009_2011.model',
                 'LDA_model/2012/lda_2012.model'),
                '5': ('LDA_model/2010_2012/lda_2010_2012.model',
                 'LDA_model/2012/lda_2012.model'),
                '6': ('LDA_model/2011_2013/lda_2011_2013.model',
                 'LDA_model/2014/lda_2014.model'),
                '7': ('LDA_model/2012_2014/lda_2012_2014.model',
                 'LDA_model/2015/lda_2015.model'),
                '8': ('LDA_model/2012_2015/lda_2012_2015.model',
                 'LDA_model/2016/lda_2016.model'),
                '9': ('LDA_model/2014_2016/lda_2014_2016.model',
                 'LDA_model/2017/lda_2017.model') }



####### Define functions #########

          
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def prepare_LDA_text(sentences):
    
    data_words = list(sent_to_words(sentences))
    
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    return id2word, texts, corpus



def prepare_data_LDA(df, text_feature, downsample = False):
    
    # Clean and convert data
    data = df.dropna(subset=[text_feature])
    sec_filing_date = pd.to_datetime(data['sec_filing_date'], format = '%Y%m%d')
    data['sec_filing_date'] = sec_filing_date
    data['rating_downgrade'] = data['rating_downgrade']


    # Create train & Validation set
    data = data[data['sec_filing_date'] < datetime.datetime(2019,1,1)]
    
    
    # Create holdoutset
    df_holdout = data[data['sec_filing_date'] > datetime.datetime(2017,12,31)]  
    if downsample:
        df_holdout = down_sample_majority_class(df_holdout, 'rating_downgrade')
    
    # Create train & Validation set
    df_train = data[data['sec_filing_date'] <= datetime.datetime(2017,12,31)]
    if downsample:
        df_train = down_sample_majority_class(df_train, 'rating_downgrade')
    
    return df_train, df_holdout




def get_topic_proba(data, lda_model_train):
    
    data = data.values.tolist()
    data = [val for sublist in data for val in sublist]
    output = pd.DataFrame()
    for obs in tqdm(data):
        obs = obs.split('\n')
        id2word, texts, corpus = prepare_LDA_text(obs)
        doc = lda_model_train[corpus]
        topics_prob = dict()
        for i in range(0, 30):
            topics_prob[i] = 0
        for sent in doc:
            topics, _, _ = sent
            for t in topics:
                tidx, prob = t
                topics_prob[tidx] += prob
        for key, value in topics_prob.items():
            if topics_prob[key] > 0:
                topics_prob[key] = value / len(doc)
        output = output.append(topics_prob, ignore_index=True)

    return output



def LDA_model_out_of_time(df, features, target, dest_all_model, algo, standard = True, downsample = False):

    X = df[features]
    y = df[target].astype('bool')

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
    k_folds = len(tbcv_folds)
    for k_index, (train_index, test_index) in enumerate(tbcv_folds):

        dest_train, dest_val = dest_all_model[str(k_index+1)]
 
        data_train   = X.loc[train_index].drop('sec_filing_date', axis=1)
        target_train   = y.loc[train_index]
        
        data_test   = X.loc[test_index].drop('sec_filing_date', axis=1)
        target_test   = y.loc[test_index]
        
        
        print("=========================================")
        print("==== K Fold Validation step => %d/%d ======" % (k_index+1, k_folds))
        print("=========================================")
        
        
        lda_model_train = LdaMulticore.load(lda_data_dir + dest_train)
        
        if downsample:
        
            try:
                data_train = pickle.load(open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_downsample_data_train.list', "rb" ))
            except:
                print("Prepare Train data")
                data_train = get_topic_proba(data_train, lda_model_train)
                pickle.dump(data_train, open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_downsample_data_train.list', "wb" ))
            
            try:
                data_test = pickle.load(open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_downsample_data_test.list', "rb" ))
            except:
                print("Prepare Test data")
                data_test = get_topic_proba(data_test, lda_model_train)
                pickle.dump(data_test, open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_downsample_data_test.list', "wb" ))
        
        else:
            
            try:
                data_train = pickle.load(open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_data_train.list', "rb" ))
            except:
                print("Prepare Train data")
                data_train = get_topic_proba(data_train, lda_model_train)
                pickle.dump(data_train, open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_data_train.list', "wb" ))
            
            try:
                data_test = pickle.load(open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_data_test.list', "rb" ))
            except:
                print("Prepare Test data")
                data_test = get_topic_proba(data_test, lda_model_train)
                pickle.dump(data_test, open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_data_test.list', "wb" ))
        
           
        
        clf.fit(data_train,target_train.values.ravel())
        preds = clf.predict(data_test)
    
        # accuracy for the current fold only    
        score = clf.score(data_test,target_test)

        f1 = f1_score(target_test, preds)
        
        cf_matrix_val += confusion_matrix(target_test, preds)
        scores['acc'].append(score)
        scores['f1'].append(f1)
        

    print("Cross Validation Score: " + str(sum(scores['acc']) / len(scores['acc'])))       

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


def LDA_model_out_of_time_tpot(df, features, target, dest_all_model, downsample = False):

    X = df[features]
    y = df[target].astype('bool')
    
    scores = {'acc': [], 'f1': []}
    cf_matrix_val = np.zeros((2,2), dtype=np.int)

    tbcv = TimeBasedCV(train_period=3, test_period=1, freq='years')
    tbcv_folds = tbcv.split(df,  validation_split_date = datetime.date(2008,12,31), date_column = 'sec_filing_date')
    k_folds = len(tbcv_folds)
    for k_index, (train_index, test_index) in enumerate(tbcv_folds):

        dest_train, dest_val = dest_all_model[str(k_index+1)]
 
        data_train   = X.loc[train_index].drop('sec_filing_date', axis=1)
        target_train   = y.loc[train_index]
        
        data_test   = X.loc[test_index].drop('sec_filing_date', axis=1)
        target_test   = y.loc[test_index]
        
        
        print("=========================================")
        print("==== K Fold Validation step => %d/%d ======" % (k_index+1, k_folds))
        print("=========================================")
        
        
        lda_model_train = LdaMulticore.load(lda_data_dir + dest_train)
        
        if downsample:
        
            try:
                data_train = pickle.load(open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_downsample_data_train.list', "rb" ))
            except:
                print("Prepare Train data")
                data_train = get_topic_proba(data_train, lda_model_train)
                pickle.dump(data_train, open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_downsample_data_train.list', "wb" ))
            
            try:
                data_test = pickle.load(open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_downsample_data_test.list', "rb" ))
            except:
                print("Prepare Test data")
                data_test = get_topic_proba(data_test, lda_model_train)
                pickle.dump(data_test, open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_downsample_data_test.list', "wb" ))
        
        else:
            
            try:
                data_train = pickle.load(open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_data_train.list', "rb" ))
            except:
                print("Prepare Train data")
                data_train = get_topic_proba(data_train, lda_model_train)
                pickle.dump(data_train, open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_data_train.list', "wb" ))
            
            try:
                data_test = pickle.load(open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_data_test.list', "rb" ))
            except:
                print("Prepare Test data")
                data_test = get_topic_proba(data_test, lda_model_train)
                pickle.dump(data_test, open(main_dir + 'data/topic_predictions/'  + str(k_index+1) + '_data_test.list', "wb" ))
        
        
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
df_diff_sent = pd.read_csv( main_dir + 'data/df_diff_sent.csv', sep=',')
train_LDA, holdout_LDA = prepare_data_LDA(df_diff_sent, 'diff_text')
features_lda = ['diff_text', 'sec_filing_date']
target_lda = ['rating_downgrade']

### No resampling - topic-based prediction ###
scores_lr_topic, model_lr_topic, cm_r_topic = LDA_model_out_of_time(train_LDA, features_lda, target_lda, dest_all_model, LogisticRegression(max_iter=1000))
scores_rfc_topic, model_rfc_topic, cm_rfc_topic = LDA_model_out_of_time(train_LDA, features_lda, target_lda, dest_all_model, RandomForestClassifier())
scores_ada_topic, model_ada_topic, cm_ada_topic = LDA_model_out_of_time(train_LDA, features_lda, target_lda, dest_all_model, AdaBoostClassifier())
scores_svc_topic, model_svc_topic, cm_svc_topic = LDA_model_out_of_time(train_LDA, features_lda, target_lda, dest_all_model, SVC())
scores_tpot_topic, model_tpot_topic, cm_tpot_topic = LDA_model_out_of_time_tpot(train_LDA, features_lda, target_lda, dest_all_model)


### No resampling - topic-based prediction ###
scores_lr_topic, model_lr_topic, cm_r_topic = LDA_model_out_of_time(train_LDA, features_lda, target_lda, dest_all_model, LogisticRegression(max_iter=1000), downsample = True)
scores_rfc_topic, model_rfc_topic, cm_rfc_topic = LDA_model_out_of_time(train_LDA, features_lda, target_lda, dest_all_model, RandomForestClassifier(), downsample = True)
scores_ada_topic, model_ada_topic, cm_ada_topic = LDA_model_out_of_time(train_LDA, features_lda, target_lda, dest_all_model, AdaBoostClassifier(), downsample = True)
scores_svc_topic, model_svc_topic, cm_svc_topic = LDA_model_out_of_time(train_LDA, features_lda, target_lda, dest_all_model, SVC(), downsample = True)
scores_tpot_topic, model_tpot_topic, cm_tpot_topic = LDA_model_out_of_time_tpot(train_LDA, features_lda, target_lda, dest_all_model, downsample = True)

