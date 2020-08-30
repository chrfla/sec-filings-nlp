#!/usr/bin/env python
# coding: utf-8

####### Import packages #########

from pprint import pprint
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
import spacy
import pyLDAvis
import pyLDAvis.gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import random
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import nltk;
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['may', 'could', 'business', 'result', 'affect', 'include'])
nlp = spacy.load('en', disable=['parser', 'ner'])
from _settings import main_dir, lda_data_dir




####### Define functions #########


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def Convert(tup, di): 
    for a, b in tup:
        di[a] = float(b)
    return di 



def train_LDA_model (data, num_topics, CPUs):

    # Pre-processing
    sentences = [nltk.tokenize.sent_tokenize(doc) for doc in data]
    sentences = [val for sublist in sentences for val in sublist]
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


    # ## Train LDA Model
    
    # Build LDA model
    lda_model = LdaMulticore(corpus=corpus,
                            id2word=id2word,
                            num_topics = num_topics, 
                            random_state=50,
                            chunksize=100,
                            passes=10,
                            per_word_topics=True,
                            workers = CPUs)
    
    model_dest = lda_data_dir + 'LDA_model/all_years_2007_2017/lda_model_all_years.model'
    lda_model.save(model_dest)

    
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Visualize the topics
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    storage_dest_lda_html = lda_data_dir + 'LDA_model/all_years_2007_2017/all_years_2007_2017_local_lda.html'
    pyLDAvis.save_html(vis, storage_dest_lda_html)

    
    wordcloud_dest = lda_data_dir + 'LDA_model/all_years_2007_2017/wordclouds/'
    
    for t in range(lda_model.num_topics):
        plt.figure()
        dictionary = {} 
        plt.imshow(WordCloud().fit_words(Convert(lda_model.show_topic(t, 30), dictionary)))
        plt.axis("off")
        plt.title("Topic_" + str(t))
        plt.show()
        plt.savefig(wordcloud_dest + "Topic #" + str(t)+'.png') # set location on server


    return lda_model



####### Execute code below #########

data = pickle.load(open(main_dir + 'data/clean_text.list', "rb" ))
lda_model = train_LDA_model (data, num_topics = 30, CPUs = 6)
