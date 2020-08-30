#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Import packages #########

import pandas as pd
import re
import pickle
import os
import numpy as np
import unicodedata
import cdifflib as dif
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from _settings import main_dir
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
    

        
####### Define functions #########

def parallelize_dataframe_grouped(df, func):
    ciks_split = np.array_split(df['cik'].unique(), num_partitions)
    df_split = []
    for i in ciks_split:
        df_split.append(df.loc[df['cik'].isin(i)])  
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def inner(b, list1):
    
    if not list1:
        return False, 0, True

    for idx_a, a in enumerate(list1):
        s = dif.CSequenceMatcher(None, a, b, autojunk=False).ratio()
        if s > threshold_diff_lib:
            return False, idx_a, False
    
    return True, idx_a, False


def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    new_text = new_text.replace('\n', ' ')
    return new_text


def remove_special_characters_numbers(text):
    p = re.compile('[\W_]+')
    return p.sub(' ', text)


def ComputeJaccardSimilarity(words_A, words_B):
    """
    Title: Scraping 10-Ks and 10-Qs for Alpha
    Author: Lucy Wu
    Date: 2018
    Code version: 1.0
    Availability: https://www.quantopian.com/posts/scraping-10-ks-and-10-qs-for-alpha
    """
    words_intersect = len(words_A.intersection(words_B))
    words_union = len(words_A.union(words_B))
    jaccard_score = words_intersect / words_union
    return jaccard_score


def get_cosine_sim(*strs): 
    """
    Title: Overview of Text Similarity Metrics in Python
    Author: Sanket Gupta
    Date: 2018
    Code version: 1.0
    Availability: https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
    """
    vectors = [t for t in get_vectors(*strs)]
    return round(cosine_similarity(vectors)[0][1], 4)
    

def get_vectors(*strs):
    """
    Title: Overview of Text Similarity Metrics in Python
    Author: Sanket Gupta
    Date: 2018
    Code version: 1.0
    Availability: https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
    """
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def difference_cosine_jaccard(df):
    # Loop through every companyid
    for cid in tqdm(df['cik'].unique()):
        key = cid
        filtered = df[df['cik']==key]
        #only use report, if more than one observation
        if len(filtered) < 2:
            continue
        else:
            # Order all reports by reporting year
            filtered = filtered.sort_values(by='sec_period_of_report', ascending=True)
            n = 0
            # Get previous and current text; skip first year
            for index, row in filtered.iterrows():
                if n > 0:
                                                    
                    text_current = row['text']
                    year_current = row['sec_year']
                    
                    if int(year_current) - int(year_prev) <= 2:
                       
                        text_current = remove_accented_chars(text_current)
                        text_prev = remove_accented_chars(text_prev)
                        
                        text_current_clean = re.findall(r"[a-zA-Z_]+", text_current)
                        text_prev_clean = re.findall(r"[a-zA-Z_]+", text_prev)
                        
                        words_current = set(text_current_clean)
                        words_prev = set(text_prev_clean)
                        
                        temp1= " ".join(text_current_clean)
                        temp2 = " ".join(text_prev_clean)
                    
                        df.loc[index, 'Sim_Cosine'] = get_cosine_sim(temp1, temp2)
                        df.loc[index, 'Sim_Jaccard'] = ComputeJaccardSimilarity(words_current, words_prev)
                        
                text_prev = row['text']
                year_prev = row['sec_year']
     
                n += 1
    return df




def difference_sentences(df):

    # Loop through every companyid
    for cid in tqdm(df['cik'].unique()):
        key = cid
        filtered = df[df['cik']==key]
        #only use report, if more than one observation
        if len(filtered) < 2:
            continue
        else:
            # Order all reports by reporting year
            filtered = filtered.sort_values(by='sec_period_of_report', ascending=True)
            n = 0
            # Get previous and current text; skip first year
            for index, row in filtered.iterrows():
                if n > 0:
                    
                    text_current = []
                    year_current = row['sec_year']
                    
                    if int(year_current) - int(year_prev) <= 2:
    
                        for sent in nltk.tokenize.sent_tokenize(row['text']):
                            words = nltk.word_tokenize(sent)
                            if len(words) > 4:
                                sent_clean = remove_accented_chars(sent)
                                sent_clean = remove_special_characters_numbers(sent_clean)
                                text_current.append(' '.join([word for word in nltk.word_tokenize(sent_clean) if word not in stopwords]))
    
                        difference = []
                        
                        # Execute sentence-by-sentence comparison
                        for idx_b, b in enumerate(text_current):
                            include, idx_a, empty = inner(b, text_prev)
                            if empty:
                                continue
                            if include == True:
                                difference.append(b)
                            if include == False:
                                text_prev.pop(idx_a)
      
                        if difference == []:
                            df.loc[index, 'diff_text'] = '\n'.join('')
                            df.loc[index, 'change'] = 0
                        else:
                            df.loc[index, 'diff_text'] = '\n'.join(difference)
                            df.loc[index, 'change'] =  len(nltk.word_tokenize(' '.join(difference))) / text_prev_len
                            
                        
                        df.loc[index, 'clean_text'] = '. \n'.join(text_current)
                    
                        
                text_prev = []
                year_prev = row['sec_year']
                

                for sent in nltk.tokenize.sent_tokenize(row['text']):
                    words = nltk.word_tokenize(sent)
                    if len(words) > 4:
                        sent_clean = remove_accented_chars(sent)
                        sent_clean = remove_special_characters_numbers(sent_clean)
                        text_prev.append(' '.join([word for word in nltk.word_tokenize(sent_clean) if word not in stopwords]))
                text_prev_len = len(nltk.word_tokenize(' '.join(text_prev)))
                n += 1

    return df


def save_data_csv (data, filename, path, index = False):
    if not os.path.exists(path):
        os.makedirs(path)
    filePath = path + filename + '.csv'
    return data.to_csv(filePath, sep = ",", index = index, header=True)



####### Execute code below #########

## Multi Processing Settings##
num_partitions = 7 #number of partitions to split dataframe
num_cores = 7 #number of cores   

## Similarity between consecutive reports (can be executed on local machine) ##
df = pd.read_csv( main_dir + "data/df_clean_reports.csv", sep=',')
df_cosine_jaccard = parallelize_dataframe_grouped(df, difference_cosine_jaccard)
save_data_csv (df_cosine_jaccard, "df_sim_cosine", main_dir + "data/", index = True)

## Difference by sentence comparison (recommended to run on server with 24 cores and 100GB RAM) ##
threshold_diff_lib = 0.7
df_diff_sent = parallelize_dataframe_grouped(df, difference_sentences)
save_data_csv (df_diff_sent.drop(columns=['text']), "df_diff_sent", main_dir + "data/", index = True)

## Prepare text for LDA model
data = df_diff_sent.dropna(subset=['clean_text'])
data = list(data['clean_text'])
pickle.dump(data, open(main_dir + 'data/clean_text.list', "wb" ))
