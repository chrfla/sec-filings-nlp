# About the Project

Master Thesis at BI Norwegian Business School (2020)

**Title:** HIDDEN VALUE IN CORPORATE DISCLOSURE?
Predicting downgrade in credit ratings based on risk factor sections in 10-K filings\
**Authors:** Christian Flasshoff and Jonathan S. Rau

#### Abstract
With the emergence of Textual Analysis research and latest technological innovations in machine learning techniques, various studies have confirmed the predictive power of text-based corporate disclosure. This master thesis follows a recent call in academia to investigates the extent to which qualitative risk disclosure in 10-K filings can inform a machine learning based prediction model for a downgrade in credit ratings. Conceptualized as a potential mechanism to reduce information asymmetry in the principal-agent relationship of managers and shareholders, the focal thesis contributes to two academic fields, i.e., Textual Analysis and Management research. We utilize a variety of traditional and state-of-the-art Natural Language Processing and statistical learning algorithms. Our results suggest that the predictive power of qualitative risk disclosure in 10-K filings from 2006-2018 is limited with regard to credit rating downgrades. None of our models achieve a better performance than the simplest baseline models.

Full-text is available [here](https://github.com/chrfla/sec-filings-nlp/Thesis.pdf).
___

# Documentation

## Setup
In order to reproduce the results or to use the code provided in this repository, follow this setup guide.
#### Clone repository

```bash
$ git clone https://github.com/chrfla/sec-filings-nlp
```

#### Download files
Make sure to edit the [_settings](https://github.com/chrfla/sec-filings-nlp/_settings.py) file and download all large files from GDrive (see separated README files in subfolders).

The credit rating data is retrieved from Capital IQ (via WRDS). Accessing this data requires a valid a WRDS subscription. You can insert your WRDS username in [_settings](https://github.com/chrfla/sec-filings-nlp/_settings.py), so the data can be downloaded automatically, when executing the Python scripts.

#### Install packages


```bash
$ cd sec-filings-nlp
$ pip install -r requirements.txt
```


#### Python environment
We recommend using a cloud computing instance (e.g., 24 CPUs and 100GB RAM) for some of the more computational expensive operations. The [Littlest JupyterHub](https://tljh.jupyter.org/en/latest/) implementation is a suitable solution to run a Python environment on a virtual machine.



## Structure of repository

#### Python scripts

`01_create_dataset.py`\
This script retrieves the credit rating from Capital IQ, inserts the risk sections from 10-fillings and merges both components.

`02_clean_reports.py`\
The script cleans the filings from undesired text, deletes too short risk sections and identifies incomplete sections.

`03_descriptive_statistics.py`\
The script visualizes the data and generates plots used in the thesis.

`04_calculate_sim_diff.py`\
The script calculates the Cosine similarity between two consecutive filings and extracts changed sentences. The later part is recommended to run on a cloud computing instance, to make use of the multiprocessing (approx. 3h runtime with 24 cores).

`05_train_sim.py`\
This script trains various machine learning models on the Cosine similarity data and uses out-of-time validation to validate the results. It also generates confusion matrices, which are stored under [results/SIM](https://github.com/chrfla/sec-filings-nlp/tree/master/results/SIM).

`06_train_tf-idf.py`\
This script trains various machine learning models on the TF-IDF data and uses out-of-time validation to validate the results. It also generates confusion matrices, which are stored under [results/DIF/TF-IDF](https://github.com/chrfla/sec-filings-nlp/tree/master/results/DIF/TF-IDF).

`07_train_avg_word_embedding.py`\
This script trains various machine learning models on the avg. word embedding data and uses out-of-time validation to validate the results. It also generates confusion matrices, which are stored under [results/DIF/AVG-W2V_(pre-trained)](https://github.com/chrfla/sec-filings-nlp/tree/master/results/DIF/AVG-W2V_(pre-trained)). Please note that the avg. word embedding takes time (approx. 3.5h).

`08_train_word_embedding.py`\
This script trains a CNN for word embedding with pre-trained and self-trained Word2Vec models. It also generates confusion matrices, which are stored under [results/DIF/W2V_(pre-trained)](https://github.com/chrfla/sec-filings-nlp/tree/master/results/DIF/W2V_(pre-trained)) and [results/DIF/W2V_(self-trained)](https://github.com/chrfla/sec-filings-nlp/tree/master/results/DIF/W2V_(self-trained)).

In addition, the self-trained Word2Vec model is documented in here. The weights for the self-trained Word2Vec model are located in [data/w2v_model](https://github.com/chrfla/sec-filings-nlp/tree/master/data/w2v_model).



`09_LDA_all_years.py`\
This script trains a LDA model with data from all years (2007 to 2018). The results are located in [results/LDA_model/all_years_2007_2017](https://github.com/chrfla/sec-filings-nlp/tree/master/results/LDA_model/all_years_2007_2017).


`10_LDA_out-of-time_validation.py`\
This script trains 18 different LDA models and applies out-of-time validation with hierarchical clustering to the resulting models. The resulting models and distance dendrograms are documented under [results/LDA_model](https://github.com/chrfla/sec-filings-nlp/tree/master/results/LDA_model).



`11_LDA_topic_prediction.py`\
This script uses the models trained in the previous script and calculates the topic distributions. It then trains various machine learning models on this topic-based dataset and uses out-of-time validation to validate the results. It also generates confusion matrices, which are stored under [results/DIF/TOPIC](https://github.com/chrfla/sec-filings-nlp/tree/master/results/DIF/TOPIC).


#### Assets
Some useful utility functions and input files are stored under `assets/...`.


#### Data
Datasets generated from the Python scripts are store under `data/...`. Please refer to the [README file](https://github.com/chrfla/sec-filings-nlp/blob/master/data/README.md) for more information about downloading the finished datasets.

`data/sec-text/...` stores all SEC 10-filings used in this project. Please refer to the [README file](https://github.com/chrfla/sec-filings-nlp/blob/master/data/sec-text/README.md) for more information about downloading these files in a bulk.

`data/10K-word-embeddings/...` refers to a repository with a pre-trained Word2Vec model on 10K-filings, which is used in this project.

`data/w2v_model/...` stores the weights for the self-trained Word2Vec model.
###### Usage of Word2Vec model:
```python
import pickle
from gensim.models import Word2Vec
w2v_words_dest = 'ENTER LOCATION OF VOCAB FILE'
w2v_model_dest = 'ENTER LOCATION OF WEIGHTS FILE'
vocab_to_int = pickle.load(open(w2v_words_dest, "rb" ))
embed = Word2Vec.load(w2v_model_dest)

```



#### Results
The results are store in three main folders, with content described as follows:

`results/SIM/...`
- Cosine similarity



`results/DIF/...`
- TF-IDF
- Avg. word embedding (pre-trained)
- Word embedding (pre-trained)
- Word embedding (self-trained)
- Topic-based prediction

`results/LDA_model/...`
- Weights for all trained LDA models
- Wordclouds  for all trained LDA models
- Intertopic distance maps (i.e. pyLDAvis HTML files)
- Hierarchical clustering distance dendrograms (comparing LDA models)
