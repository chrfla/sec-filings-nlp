#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Import packages #########

from __future__ import print_function
import numpy as np
import pickle
import torch
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import gensim
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from sklearn.metrics import confusion_matrix
from assets.TimeBasedCV import TimeBasedCV
from assets.CfMatrix import make_confusion_matrix
from assets.utils import down_sample_majority_class
from _settings import main_dir
import multiprocessing


####### Define functions #########

def prepare_data_word_embedding(df, text_feature, downsample = False):
    
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




def model_training_out_of_time_pretrained(df, holdout, target, features, show_holdout = False, downsample = False):
    
    """
    Title: Learning Word Embeddings from 10-K Filings for Financial NLP Tasks
    Author: Saurabh Sehrawat
    Date: 2019
    Code version: 1.0
    Availability: https://github.com/ssehrawat/10K-word-embeddings
    """
    
    embed = torch.load(main_dir + 'data/10K-word-embeddings/10k_word_embeddings.tar')
    vocab_to_int = torch.load(main_dir + 'data/10K-word-embeddings/vocab_to_int.tar')
    
    X = df[features]
    y = df[target].astype('bool')
    X_holdout = holdout[features].drop('sec_filing_date', axis=1)
    y_holdout = holdout[target].astype('bool')


    alog_name = 'CNN'
    print('### ' + alog_name + ' ###')
    
    scores = dict()
    cf_matrix_val = np.zeros((2,2), dtype=np.int)
    tbcv = TimeBasedCV(train_period=3, test_period=1, freq='years')
    tbcv_folds = tbcv.split(df,  validation_split_date = datetime.date(2008,12,31), date_column = 'sec_filing_date')
    k_folds = len(tbcv_folds)
    for k_index, (train_index, test_index) in enumerate(tbcv_folds):
        
        if downsample: 
            df_kfold = down_sample_majority_class(df, 'rating_downgrade')
            df_kfold_index = df_kfold.index.tolist()
            train_index = [idx for idx in list(train_index) if idx in df_kfold_index]
            
    
        data_train   = X.loc[train_index].drop('sec_filing_date', axis=1)
        target_train = y.loc[train_index]
    
        data_test    = X.loc[test_index].drop('sec_filing_date', axis=1)
        target_test  = y.loc[test_index]
        
        print("=========================================")
        print("==== K Fold Validation step => %d/%d ======" % (k_index+1, k_folds))
        print("=========================================")
              
        x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH = create_embedding_layer(data_train, target_train, data_test, target_test, embed, vocab_to_int, trainable = False)
        history, model = train_model_keras_CNN(x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH)   
        print(history.history)
        scores[k_index] = history.history
        
        preds_y = model.predict(x_val)
        preds_y = np.rint(preds_y)
        
        preds_y = preds_y.argmax(axis=-1)
        y_val = y_val.argmax(axis=-1)
        cf_matrix_val += confusion_matrix(y_val, preds_y)
    

    if show_holdout:
        # Test model trained on last three years on holdout data
        
        frames = [test_index for train_index, test_index in tbcv_folds[-3:]]
        frames = [item for sublist in frames for item in sublist]
        data_train   = X.loc[frames].drop('sec_filing_date', axis=1)
        target_train = y.loc[frames]
        x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH = create_embedding_layer(data_train, target_train, X_holdout, y_holdout, embed, vocab_to_int, trainable = False)
        history, model = train_model_keras_CNN(x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH)   
        
        preds_y = model.predict(x_val)
        preds_y = np.rint(preds_y)
        
        preds_y = preds_y.argmax(axis=-1)
        y_val = y_val.argmax(axis=-1)
        cf_matrix = confusion_matrix(y_val, preds_y)
        
    
        
        
        scores['holdout'] = history.history
            
        #print("Holdout Score: " + str(clf.score(x_val, y_val)))     
        #print('\n')
        # Visualize confusion matrix for holdout data
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['No Downgrade', 'Downgrade']
        make_confusion_matrix(cf_matrix, 
                          group_names = labels,
                          categories = categories,
                          cbar = False,
                          title ='Confusion Matrix: ' + alog_name,
                          figsize = (10,10))
    else:
        
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['No Downgrade', 'Downgrade']
        make_confusion_matrix(cf_matrix_val, 
                          group_names = labels,
                          categories = categories,
                          cbar = False,
                          title ='Confusion Matrix: ' + alog_name,
                          figsize = (10,10))
        
    return scores, cf_matrix_val

    


def create_embedding_layer(data_train, target_train, data_test, target_test, embed, vocab_to_int, trainable = False):
    
    MAX_SEQUENCE_LENGTH = 4000
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 300

    data_train = data_train['diff_text'].values.tolist()
    target_train = target_train['rating_downgrade'].values.tolist()
    
    data_test = data_test['diff_text'].values.tolist()
    target_test = target_test['rating_downgrade'].values.tolist()
    
    num_val = len(data_test)
    
    texts = data_train + data_test
    labels = target_train + target_test
    
    
    tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
    
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    x_train = data[:-num_val]
    y_train = labels[:-num_val]
    x_val = data[-num_val:]
    y_val = labels[-num_val:]

    print('Preparing embedding matrix.')

   
    n = 0     
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        try:
            embedding_vector = embed[vocab_to_int[word]]
        except: 
            n += 1
            embedding_vector = None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    print(n)

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = MAX_SEQUENCE_LENGTH,
                                trainable = trainable)
    
    return x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH





def train_model_keras_CNN(x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH):
    ## train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(256, 10, activation='relu')(embedded_sequences)
    x = MaxPooling1D(3)(x)
    x = Conv1D(256, 10, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(256, 10, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(256, activation='relu')(x)
    preds = Dense(2, activation='sigmoid')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', f1])
    
    history = model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_val, y_val))

    return history, model
    


#taken from old keras source code
def f1(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def vis_train_epoch(scores_dict):
    result = pd.DataFrame()
    for key in scores_dict.keys():
        scores = scores_dict[key]
        key = str(key)
        train = dict(list(scores.items())[len(scores)//2:])
        test = dict(list(scores.items())[:len(scores)//2])
        train['epochs'] = list(range(1, len(train['acc']) + 1))
        test['epochs'] = list(range(1, len(test['val_acc']) + 1))
        train[''] = ['training'] * len(train['acc'])
        test[''] = ['validation'] * len(test['val_acc'])
        train['k_fold'] = [key] * len(train['acc'])
        test['k_fold'] = [key] * len(test['val_acc'])
    
    
        for k in ['val_acc', 'val_loss', 'val_f1']:
            new_key = k.replace('val_', '')
            old_key = k
            test[new_key] = test.pop(old_key)
            
        train = pd.DataFrame.from_dict(train)
        test = pd.DataFrame.from_dict(test)
    
        frames = [train, test]
        result = pd.concat(frames)
        for measure in ['f1']:
            plt.figure(figsize=(10,10))
            #title = 'Fit over epochs (' + measure + ') for ' + str(int(key+1)) + '-fold ' 
            title = str(int(key)+1) + '-fold training step' 
            sns.lineplot(x = "epochs", y = measure, hue = "" , data = result)
            plt.ylabel('f1 score\n')
            plt.xlabel('\nepochs')
            plt.title(title + '\n', fontweight='bold')
            plt.show()
    return





def model_training_custom_w2v(df, features, w2v_model_dest, w2v_words_dest, w2v_size=300, w2v_window=5, w2v_min_count=1, w2v_epochs=10):
    
    X = df[features].values
    X = [item for sublist in X.tolist() for item in sublist]
    x_train = [gensim.utils.simple_preprocess(text) for text in X]
    w2v_model = gensim.models.Word2Vec(min_count=w2v_min_count, window=w2v_window,
                                            size=w2v_size,
                                            workers=multiprocessing.cpu_count(), sg = 1)
    w2v_model.build_vocab(x_train)
    w2v_model.train(x_train, total_examples=w2v_model.corpus_count, epochs=w2v_epochs)
    w2v_words = list(w2v_model.wv.vocab)

    w2v_model.save(w2v_model_dest)
    pickle.dump(w2v_words, open(w2v_words_dest, "wb" ))

    return




def model_training_out_of_time_self_trained(df, holdout, target, features, w2v_model_dest, w2v_words_dest, show_holdout = False, downsample = False):
    
    vocab_to_int = pickle.load(open(w2v_words_dest, "rb" ))
    embed = Word2Vec.load(w2v_model_dest)
    
    X = df[features]
    y = df[target].astype('bool')
    X_holdout = holdout[features].drop('sec_filing_date', axis=1)
    y_holdout = holdout[target].astype('bool')


    alog_name = 'CNN'
    print('### ' + alog_name + ' ###')
    
    scores = dict()
    cf_matrix_val = np.zeros((2,2), dtype=np.int)
    tbcv = TimeBasedCV(train_period=3, test_period=1, freq='years')
    tbcv_folds = tbcv.split(df,  validation_split_date = datetime.date(2008,12,31), date_column = 'sec_filing_date')
    k_folds = len(tbcv_folds)
    for k_index, (train_index, test_index) in enumerate(tbcv_folds):
        
        if downsample: 
            df_kfold = down_sample_majority_class(df, 'rating_downgrade')
            df_kfold_index = df_kfold.index.tolist()
            train_index = [idx for idx in list(train_index) if idx in df_kfold_index]    
    
        data_train   = X.loc[train_index].drop('sec_filing_date', axis=1)
        target_train = y.loc[train_index]
    
        data_test    = X.loc[test_index].drop('sec_filing_date', axis=1)
        target_test  = y.loc[test_index]
        
        print("=========================================")
        print("==== K Fold Validation step => %d/%d ======" % (k_index+1, k_folds))
        print("=========================================")
              
        x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH = create_embedding_layer_self_trained(data_train, target_train, data_test, target_test, embed, vocab_to_int, trainable = False)
        history, model = train_model_keras_CNN(x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH)   
        print(history.history)
        scores[k_index] = history.history
        
        preds_y = model.predict(x_val)
        preds_y = np.rint(preds_y)
        
        preds_y = preds_y.argmax(axis=-1)
        y_val = y_val.argmax(axis=-1)
        cf_matrix_val += confusion_matrix(y_val, preds_y)
    

    if show_holdout:
        # Test model trained on last three years on holdout data
        
        frames = [test_index for train_index, test_index in tbcv_folds[-3:]]
        frames = [item for sublist in frames for item in sublist]
        data_train   = X.loc[frames].drop('sec_filing_date', axis=1)
        target_train = y.loc[frames]
        x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH = create_embedding_layer(data_train, target_train, X_holdout, y_holdout, embed, vocab_to_int, trainable = False)
        history, model = train_model_keras_CNN(x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH)   
        
        preds_y = model.predict(x_val)
        preds_y = np.rint(preds_y)
        
        preds_y = preds_y.argmax(axis=-1)
        y_val = y_val.argmax(axis=-1)
        cf_matrix = confusion_matrix(y_val, preds_y)
        
    
        
        
        scores['holdout'] = history.history
            
        #print("Holdout Score: " + str(clf.score(x_val, y_val)))     
        #print('\n')
        # Visualize confusion matrix for holdout data
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['No Downgrade', 'Downgrade']
        make_confusion_matrix(cf_matrix, 
                          group_names = labels,
                          categories = categories,
                          cbar = False,
                          title ='Confusion Matrix: ' + alog_name,
                          figsize = (10,10))
    else:
        
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['No Downgrade', 'Downgrade']
        make_confusion_matrix(cf_matrix_val, 
                          group_names = labels,
                          categories = categories,
                          cbar = False,
                          title ='Confusion Matrix: ' + alog_name,
                          figsize = (10,10))
        
    return scores, cf_matrix_val



def create_embedding_layer_self_trained(data_train, target_train, data_test, target_test, embed, vocab_to_int, trainable = False):
    
    MAX_SEQUENCE_LENGTH = 4000
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 300

    
    data_train = data_train['diff_text'].values.tolist()
    target_train = target_train['rating_downgrade'].values.tolist()
    
    data_test = data_test['diff_text'].values.tolist()
    target_test = target_test['rating_downgrade'].values.tolist()
    
    
    num_val = len(data_test)
    
    texts = data_train + data_test
    labels = target_train + target_test
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    labels = to_categorical(labels, num_classes)
    


    tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    data = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    x_train = data[:-num_val]
    y_train = labels[:-num_val]
    x_val = data[-num_val:]
    y_val = labels[-num_val:]

    print('Preparing embedding matrix.')

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, idx in word_index.items():
        if word in vocab_to_int:
            embedding_vector = embed.wv.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embed.wv[word]
        

    embedding_layer = Embedding(vocab_size,
                                EMBEDDING_DIM,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = MAX_SEQUENCE_LENGTH,
                                trainable = trainable)
    
    print('Embedding matrix prepared.')
    
    return x_train, y_train, x_val, y_val, embedding_layer, MAX_SEQUENCE_LENGTH





####### Execute code below #########


# Import dataset
df_word_embd = pd.read_csv( main_dir + 'data/df_diff_sent.csv', sep=',')
df_word_embd_train, df_word_embd_holdout = prepare_data_word_embedding(df_word_embd, 'diff_text')
features_word_embd = ['diff_text', 'sec_filing_date']
target_word_embd = ['rating_downgrade']

### No resampling - word embedding (pre-trained) ###
scores_word_CNN, cm_word_CNN = model_training_out_of_time_pretrained(df_word_embd_train, df_word_embd_holdout, target_word_embd, features_word_embd, downsample = False)
vis_train_epoch(scores_word_CNN)

### Under-resampling - word embedding (pre-trained) ###
scores_word_CNN, cm_word_CNN = model_training_out_of_time_pretrained(df_word_embd_train, df_word_embd_holdout, target_word_embd, features_word_embd, downsample = True)
vis_train_epoch(scores_word_CNN)






# Import dataset
df_word_embd_w2v = pd.read_csv( main_dir + 'data/df_diff_sent.csv', sep=',')
df_word_embd_train_w2v, df_word_embd_holdout_w2c, = prepare_data_word_embedding(df_word_embd_w2v, 'clean_text')
features_w2v = ['clean_text']   
w2v_model_dest = main_dir + 'data/w2v_model/word2vec.model'
w2v_words_dest = main_dir + 'data/w2v_model/w2v_words.list'

### Word embedding train custom model ###
model_training_custom_w2v(df_word_embd_train_w2v, features_w2v, w2v_model_dest, w2v_words_dest)





# Import dataset
df_word_embd = pd.read_csv( main_dir + 'data/df_diff_sent.csv', sep=',')
df_word_embd.columns
df_word_embd_train, df_word_embd_holdout = prepare_data_word_embedding(df_word_embd, 'diff_text')
features_word_embd = ['diff_text', 'sec_filing_date']
target_word_embd = ['rating_downgrade']

### No resampling - word embedding (custom-trained) ###
scores_word_custom_CNN, cm_word_custom_CNN = model_training_out_of_time_self_trained(df_word_embd_train, df_word_embd_holdout, target_word_embd, features_word_embd, w2v_model_dest, w2v_words_dest, downsample = False)
vis_train_epoch(scores_word_custom_CNN)

### Under-resampling - word embedding (custom-trained) ###
scores_word_custom_CNN, cm_word_custom_CNN = model_training_out_of_time_self_trained(df_word_embd_train, df_word_embd_holdout, target_word_embd, features_word_embd, w2v_model_dest, w2v_words_dest, downsample = True)
vis_train_epoch(scores_word_custom_CNN)

