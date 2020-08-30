#!/usr/bin/env python
# coding: utf-8

####### Import packages #########

import numpy as np
import pandas as pd
import pickle
import datetime
import time
import zipfile
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
import os
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import random
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['may', 'could', 'business', 'result', 'affect', 'include'])
nlp = spacy.load('en', disable=['parser', 'ner'])
from _settings import main_dir, lda_data_dir
from assets.TimeBasedCV import TimeBasedCV
from assets.utils import down_sample_majority_class



####### Define functions #########

def Convert(tup, di): 
    for a, b in tup:
        di[a] = float(b)
    return di 

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))
            
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




def prepare_LDA_text(data, subset = True):
    
    sentences = [nltk.tokenize.sent_tokenize(doc) for doc in data]
    sentences = [val for sublist in sentences for val in sublist]
    data_words = list(sent_to_words(sentences))
    print('Number of sentences: ' + str(len(data_words)))
    if subset:
        data_words = random.sample(data_words, 2000)

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




def create_dist_matrix (model_a_dest, model_b_dest, distance='jaccard', num_words=300, normed=True):
    
    a = LdaMulticore.load(model_a_dest)
    b = LdaMulticore.load(model_b_dest)

    mdiff_a_b, annotation_a_b = a.diff(b, distance=distance, num_words=num_words, normed=normed)
    mdiff_b_a, annotation_b_a = b.diff(a, distance=distance, num_words=num_words, normed=normed)
    #topic_diff_a_b = np.ones(mdiff_a_b.shape) - mdiff_a_b
    #topic_diff_b_a = np.ones(mdiff_a_b.shape) - mdiff_b_a
    topic_diff_a_b = mdiff_a_b
    topic_diff_b_a = mdiff_b_a

    a_ones = np.ones(topic_diff_a_b.shape)
    b_ones = np.ones(topic_diff_a_b.shape)

    first_half = np.concatenate((a_ones, topic_diff_b_a), axis=0)
    second_half = np.concatenate((topic_diff_a_b, b_ones), axis=0)
    total = np.concatenate((first_half, second_half), axis=1)
    
    return total



def array_to_pd (array, num_topics = 30):
    
    index = ['A' + str(n) for n in list(range(1,num_topics +1))] + ['B' + str(n) for n in list(range(1,num_topics +1))]
    columns = ['A' + str(n) for n in list(range(1,num_topics +1))] + ['B' + str(n) for n in list(range(1,num_topics +1))]

    df_diff_martrix = pd.DataFrame(array, index=index)
    df_diff_martrix.columns = columns
    
    return df_diff_martrix



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


   
def create_h_clustering (dist_matrix, n_topics = 30, title = 'Matching Topics\n', location = ''):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters = None, affinity='precomputed', linkage= "complete")

    model = model.fit(dist_matrix)
    fig = plt.figure(figsize=(8, 15))
    plt.title(title, fontweight='bold')
    # plot the top three levels of the dendrogram

    labels = ['A' + str(n) for n in list(range(1, n_topics + 1))] + ['B' + str(n) for n in list(range(1, n_topics + 1))]
    plot_dendrogram(model, truncate_mode=None, p= n_topics, leaf_font_size=10, labels = labels, 
                    orientation = 'left', color_threshold=0, above_threshold_color='k')
    plt.xlabel("Distance between topics", fontweight='bold')

    label_colors = {}
    for l in ['A' + str(n) for n in list(range(1,n_topics + 1))]:
        label_colors[l] = 'C2'

    for l in ['B' + str(n) for n in list(range(1,n_topics +1))]:
        label_colors[l] = 'C0'

    ax = plt.gca()
    ylbls = ax.get_ymajorticklabels()
    for lbl in ylbls:
        lbl.set_color(label_colors[lbl.get_text()])
    plt.savefig(location)
    plt.show()
    plt.close() 



def LDA_model_train_out_of_time(df, features, num_topics = 30, subset = True, CPUs = 6):
    
    dest_all_model = dict()
    
    X = df[features]

    tbcv = TimeBasedCV(train_period=3, test_period=1, freq='years')
    tbcv_folds = tbcv.split(df,  validation_split_date = datetime.date(2008,12,31), date_column = 'sec_filing_date')
    k_folds = len(tbcv_folds)
    for k_index, (train_index, test_index) in enumerate(tbcv_folds):
        
        train_years_start = min(X.loc[train_index]['sec_filing_date']).year
        train_years_end = max(X.loc[train_index]['sec_filing_date']).year
        val_year = min(X.loc[test_index]['sec_filing_date']).year
        
        data_train   = X.loc[train_index].drop('sec_filing_date', axis=1)
        data_val   = X.loc[test_index].drop('sec_filing_date', axis=1)
        
        print("=========================================")
        print("==== K Fold Validation step => %d/%d ======" % (k_index+1, k_folds))
        print("=========================================")
        

        start = time.time()
        data_train = data_train.values.tolist() 
        data_train = [val for sublist in data_train for val in sublist]
        id2word_train, texts_train, corpus_train = prepare_LDA_text(data_train, subset = subset)
        end = time.time()
        print("Preparing training text took: " + str(end - start))
        
        start = time.time()
        data_val = data_val.values.tolist() 
        data_val = [val for sublist in data_val for val in sublist]
        id2word_val, texts_val, corpus_val = prepare_LDA_text(data_val, subset = subset)
        end = time.time()
        print("Preparing validation text took: " + str(end - start))
        


        #Train LDA on Training data
        start = time.time()
        lda_model_train = LdaMulticore(corpus = corpus_train,
                                id2word = id2word_train,
                                num_topics = num_topics, 
                                random_state = 50,
                                chunksize = 100,
                                passes = 10,
                                per_word_topics = True,
                                workers = CPUs)
        
        doc_lda_train = lda_model_train[corpus_train]
        
        folder_train = str(train_years_start) + '_' + str(train_years_end)
        if not os.path.exists(lda_data_dir + 'LDA_model/' + folder_train + '/'):
            os.makedirs(lda_data_dir + 'LDA_model/' + folder_train + '/')
        dest_train = lda_data_dir + 'LDA_model/' + folder_train + '/' + 'lda_' + folder_train + '.model'

        lda_model_train.save(main_dir + dest_train)
        end = time.time()
        print("Train LDA on training data took: " + str(end - start))

        with open(lda_data_dir + 'LDA_model/' + folder_train + '/' + 'id2word.pkl', "wb") as fp:
            pickle.dump(id2word_train, fp)
        with open(lda_data_dir + 'LDA_model/' + folder_train + '/' + "texts.txt", "wb") as fp:
            pickle.dump(texts_train, fp)
        with open(lda_data_dir + 'LDA_model/' + folder_train + '/' + "corpus.txt", "wb") as fp:
            pickle.dump(corpus_train, fp)
        
        #Train LDA on Validation data
        start = time.time()
        lda_model_val = LdaMulticore(corpus = corpus_val,
                                id2word = id2word_val,
                                num_topics = num_topics, 
                                random_state = 50,
                                chunksize = 100,
                                passes = 10,
                                per_word_topics = True,
                                workers = CPUs)
        
        doc_lda_val = lda_model_val[corpus_val]
        
        folder_val = str(val_year)
        if not os.path.exists(lda_data_dir + 'LDA_model/' + folder_val + '/'):
            os.makedirs(lda_data_dir + 'LDA_model/' + folder_val + '/')
        dest_val = lda_data_dir + 'LDA_model/' + folder_val + '/' + 'lda_' + folder_val + '.model'
        lda_model_val.save(main_dir + dest_val)        
        end = time.time()
        print("Train LDA on validation data took: " + str(end - start))
        
        with open(lda_data_dir + 'LDA_model/' + folder_val + '/' + 'id2word.pkl', "wb") as fp:
            pickle.dump(id2word_val, fp)
        with open(lda_data_dir + 'LDA_model/' + folder_val + '/' + "texts.txt", "wb") as fp:
            pickle.dump(texts_val, fp)
        with open(lda_data_dir + 'LDA_model/' + folder_val + '/' + "corpus.txt", "wb") as fp:
            pickle.dump(corpus_val, fp)
        
        dest_all_model[str(k_index+1)] = (dest_train, dest_val)

        #Create Visualization
        start = time.time()
        pyLDAvis.enable_notebook()
        vis_train = pyLDAvis.gensim.prepare(lda_model_train, corpus_train, id2word_train, sort_topics=False)
        dest_train_vs = lda_data_dir + 'LDA_model/' + folder_train + '/' + 'vis_' + folder_train + '.html'
        pyLDAvis.save_html(vis_train, dest_train_vs)
        end = time.time()
        print("Train LDA visualization took: " + str(end - start))

        
        start = time.time()
        pyLDAvis.enable_notebook()
        vis_val = pyLDAvis.gensim.prepare(lda_model_val, corpus_val, id2word_val, sort_topics=False)
        dest_train_val = lda_data_dir + 'LDA_model/' + folder_val + '/' + 'vis_' + folder_val + '.html'
        pyLDAvis.save_html(vis_val, dest_train_val)
        end = time.time()
        print("Validation LDA visualization took: " + str(end - start))

        # Create Word Clouds
        # Train
        for t in range(lda_model_train.num_topics):
            plt.figure()
            dictionary = {} 
            plt.imshow(WordCloud().fit_words(Convert(lda_model_train.show_topic(t, 30), dictionary)))
            plt.axis("off")
            plt.title("Topic_" + str(t + 1))
            plt.savefig("wordclouds/Topic #" + str(t + 1)+'.png') # set location on server
            plt.close()
        
        dest_train_zip = lda_data_dir + 'LDA_model/' + folder_train + '/' + 'wordclouds_' + folder_train + '.zip'
        zipf = zipfile.ZipFile(dest_train_zip, 'w', zipfile.ZIP_DEFLATED)
        zipdir('wordclouds/', zipf)
        zipf.close()

        # Val
        for t in range(lda_model_val.num_topics):
            plt.figure()
            dictionary = {} 
            plt.imshow(WordCloud().fit_words(Convert(lda_model_val.show_topic(t, 30), dictionary)))
            plt.axis("off")
            plt.title("Topic_" + str(t + 1))
            plt.savefig("wordclouds/Topic #" + str(t + 1) +'.png') # set location on server
            plt.close()
        
        dest_val_zip = lda_data_dir + 'LDA_model/' + folder_val + '/' + 'wordclouds_' + folder_val + '.zip'
        zipf = zipfile.ZipFile(dest_val_zip, 'w', zipfile.ZIP_DEFLATED)
        zipdir('wordclouds/', zipf)
        zipf.close()
        
        # Matching topics
        start = time.time()
        # Using the Cosine distance requires a customization of the 'LdaMulticore.diff' method in the gensim package. To avoid erros, the code uses the Jaccard distance, but this can be changed to Cosine if needed.
        array_distance = create_dist_matrix (dest_train, dest_val, distance='jaccard', num_words=300, normed=True)
        title = 'Matching Topics (' + str(train_years_start) + '-' + str(train_years_end) + ' vs. ' + str(val_year) + ')' + '\n'
        location = lda_data_dir + 'LDA_model/matching_' + str(val_year) + '.png'
        create_h_clustering (array_distance, n_topics = 30, title = title, location = location)
        end = time.time()
        print("Matching topics took: " + str(end - start))

    return dest_all_model



####### Execute code below #########

df_lda = pd.read_csv(main_dir + 'data/df_diff_sent.csv', sep=',')
df_lda_train, df_lda_holdout = prepare_data_LDA(df_lda, 'clean_text')
features_lda = ['clean_text', 'sec_filing_date']
dest_all_model = LDA_model_train_out_of_time(df_lda_train, features_lda, num_topics = 30, subset = False, CPUs = 6)

