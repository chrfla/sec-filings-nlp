# sec-filings-nlp
___

# About the Project

Master Thesis at BI Norwegian Business School (2020)

**Title:** HIDDEN VALUE IN CORPORATE DISCLOSURE?
Predicting downgrade in credit ratings based on risk factor sections in 10-K filings\
**Authors:** Christian Flasshoff and Jonathan S. Rau

#### Abstract
With the emergence of Textual Analysis research and latest technological innovations in machine learning techniques, various studies have confirmed the predictive power of text-based corporate disclosure. This master thesis follows a recent call in academia to investigates the extent to which qualitative risk disclosure in 10-K filings can inform a machine learning based prediction model for a downgrade in credit ratings. Conceptualized as a potential mechanism to reduce information asymmetry in the principal-agent relationship of managers and shareholders, the focal thesis contributes to two academic fields, i.e., Textual Analysis and Management research. We utilize a variety of traditional and state-of-the-art Natural Language Processing and statistical learning algorithms. Our results suggest that the predictive power of qualitative risk disclosure in 10-K filings from 2006-2018 is limited with regard to credit rating downgrades. None of our models achieve a better performance than the simplest baseline models.

Full-text is available [here](link.thesis).
___

# Documentation

## Setup
In order to reproduce the results or to use the code provided in this repository, follow this setup guide.
#### Clone repository

```bash
$ git clone https://github.com/chrfla/sec-filings-nlp
```

#### Download files
Make sure to edit the [_settings](file.link) file and download all large files from GDrive (see separated README files in subfolders).

The credit rating data is retrieved from Capital IQ (via WRDS). Accessing this data requires a valid a WRDS subscription. You can insert your WRDS username in [_settings](file.link), so the data can be downloaded automatically, when executing the Python scripts.

#### Install packages


```bash
$ cd sec-filings-nlp
$ pip install -r requirements.txt
```


#### Python environment
We recommend using a cloud computing instance (e.g., 24 CPUs and 100GB RAM) for some of the more computational expensive operations. The [Littlest JupyterHub](https://tljh.jupyter.org/en/latest/) implementation is suitable solution to run a Python environment on a virtual machine.



## Structure of repository

#### Python scripts

`01_create_dataset.py`

`01_create_dataset.py`

`01_create_dataset.py`

`01_create_dataset.py`

`01_create_dataset.py`

`01_create_dataset.py`

`01_create_dataset.py`


#### Assets
`assets/...`



#### Data
`data/...`


`data/sec-text/...`

`data/10K-word-embeddings/...`

`data/w2v_model/...`



#### Results

`results/SIM/...`

`results/DIF/...`

`results/LDA_model/...`
