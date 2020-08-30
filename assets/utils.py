#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:07:12 2020

@author: cfl
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import sqlite3


def get_cutoffs (df, measure, num_quantiles):
    cutoffs = {}
    measure = measure
    q = num_quantiles
    for year in df['sec_year'].unique().tolist():
        year_df = df[df['sec_year'] == year]
        quantiles = year_df.quantile(list(np.round(np.arange(1/q, 1.001, 1/q), 2)), interpolation='midpoint')
        cutoffs[year] = quantiles[measure].values.tolist()
    return cutoffs


def calculate_previous_quantile(df, num_quantiles, measures):
    for m in measures:
        quantiles = get_cutoffs (df, m, 5)
        years = sorted(list(quantiles.keys()))
        
        for index, row in df.iterrows():
            year = row['sec_year']
            if year == years[0]:
                continue
            cutoff = quantiles[year - 1]
            value = row[m]
            for idx, v in enumerate(cutoff):
                if float(value) >= float(v):
                    if idx == len(cutoff) -1:
                        df.loc[index,'quant_' + m] = idx
                    else:
                        continue
                else:
                    df.loc[index,'quant_' + m] = idx
                    break
    return df




def down_sample_majority_class(df, label):
    df_majority = df[df[label]== False]
    df_minority = df[df[label] == True]
    df_majority_down = resample(df_majority, replace = False, n_samples = len(df_minority))
    df_downsampled = pd.concat([df_majority_down, df_minority])
    return df_downsampled





def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except :
        print("Connection to database has faield.")
 
    return conn



def import_sec_metadata(path_sec_db):

    # import SQLite database with SEC data
    connection = create_connection(path_sec_db)
    cursor = connection.cursor()
    df = pd.read_sql_query("SELECT * FROM metadata WHERE section_name = 'Item1A' AND output_file is not null and sec_form_header = 'Form 10-K' and batch_number = 3", connection)
    
    # Convert date into date format
    tmp = pd.to_datetime(df['sec_filing_date'], format = '%Y%m%d')
    df.insert(len(df.columns), "sec_date", tmp) 
    df['sec_date'] = df['sec_date'].astype("datetime64[ns]")
    # Rename column to match other datastructures
    df = df.rename(columns = {'sec_cik':'cik'})
    
    return df