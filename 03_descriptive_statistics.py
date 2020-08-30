#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Import packages #########

import pandas as pd
import numpy as np
from assets.utils import import_sec_metadata
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from _settings import main_dir, sec_data_dir

    
####### Define functions #########

def plot_length_sections_10K(df):
    
    df['sec_year'] = df['sec_filing_date'].astype(str).str[0:4]
    df['sec_year'] = df['sec_year'].astype(int)
    df = df[df.sec_year < 2019]
    
    year_median_length = df.groupby(['sec_year']).median()['section_n_characters'].to_frame().reset_index()
    plt.figure(figsize=(7,7))
    ax = sns.barplot(x="sec_year", y="section_n_characters", data=year_median_length, color = color_single)
    plt.ylabel('Number of characters (median)\n',  fontweight='bold', fontsize = fontsize_labels)
    plt.xlabel('\nReporting year',  fontweight='bold', fontsize = fontsize_labels)
    plt.xticks(rotation=45)
    plt.title('Median length of risk section (Item 1A) in 10-K filings' + '\n', fontweight='bold', fontsize = fontsize_title)
    ax.tick_params(labelsize = fontsize_ticks)
    
    return
    


def plot_publication_month_10K(df):
    df = df[df.sec_year < 2019]
    sec_filing_date = pd.to_datetime(df['sec_filing_date'], format = '%Y%m%d')
    df['sec_filing_date'] = sec_filing_date
    df['sec_year'] = pd.DatetimeIndex(df['sec_filing_date']).year
    df['sec_month'] = pd.DatetimeIndex(df['sec_filing_date']).month

    df['sec_month'] = df['sec_month'].apply(lambda x: calendar.month_abbr[x])
    
    year_median_length = df.groupby(['sec_month']).count()['id'].to_frame().reset_index()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(7,7))
    ax = sns.barplot(x="sec_month", y="id", data=year_median_length, color = color_single, order = months)
    plt.ylabel('Total number of 10-K filings\n',  fontweight='bold', fontsize = fontsize_labels)
    plt.xlabel('\nMonth',  fontweight='bold', fontsize = fontsize_labels)
    plt.title('Distribution of 10-K filings (2006 - 2019)' + '\n', fontweight='bold', fontsize = fontsize_title)
    ax.tick_params(labelsize = fontsize_ticks)
    
    return


def plot_time_gap_credit_rating_change (df):
    df = df[df.sec_year < 2019]
    df['sec_date'] = pd.to_datetime(df['sec_date'])
    df['rating_date'] = pd.to_datetime(df['rating_date'])
    df['days_until_change'] = df['rating_date'] - df['sec_date']
    df['days_until_change'] = pd.to_numeric(df['days_until_change'].dt.days, downcast='integer')
    #df['days_until_change'] = df['days_until_change'].astype(int)
    #df_change = df[(df["rating_downgrade"] == True) | (df["rating_upgrade"] == True)]
    df_change = df[(df["rating_downgrade"] == True)]
    #df_change = df[(df["rating_upgrade"] == True)]
    df_change_grouped = df_change.groupby(['days_until_change']).count()['id'].to_frame().reset_index()
    plt.figure(figsize=(7,7))
    ax = df_change['days_until_change'].plot.hist(bins=50, alpha=1, color = color_single)
    plt.ylabel('Number of filings\n',  fontweight='bold', fontsize = fontsize_labels)
    plt.xlabel('\nDays after filing date',  fontweight='bold', fontsize = fontsize_labels)
    plt.title('Time Lag Between SEC Filing Date and Credit Ratinng Change' + '\n', fontweight='bold', fontsize = fontsize_title)
    ax.tick_params(labelsize = fontsize_ticks)

    plt.figure(figsize=(50,5))
    ax = sns.barplot(x="days_until_change", y="id", data=df_change_grouped, color = color_single)
    plt.ylabel('Number of filings\n',  fontweight='bold', fontsize = fontsize_labels)
    plt.xlabel('\nDays after filing date',  fontweight='bold', fontsize = fontsize_labels)
    plt.title('Time Lag Between SEC Filing Date and Credit Ratinng Change' + '\n', fontweight='bold', fontsize = fontsize_title)
    ax.tick_params(labelsize = fontsize_ticks)

    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 50 != 0:
            label.set_visible(False)
    
    return


def plot_class_balance (df):
    df_class_counts = df.groupby(['sec_year', 'rating_downgrade']).count()['id'].to_frame().reset_index()
    df_class_counts = df_class_counts[df_class_counts.sec_year < 2019]
    downgrade_data = df_class_counts[df_class_counts['rating_downgrade'] == True].id.tolist()
    no_downgrade_data = df_class_counts[df_class_counts['rating_downgrade'] == False].id.tolist()
    years = df_class_counts[df_class_counts['rating_downgrade'] == False].sec_year.tolist()
    sum_data = [a + b for a, b in zip(downgrade_data, no_downgrade_data)]
    balance_downgrade = [a / b for a, b in zip(downgrade_data, sum_data)]
    balance_no_downgrade = [a / b for a, b in zip(no_downgrade_data, sum_data)]
    years = list(map(int, years))
        
    N = len(years)
    ind = np.arange(N) # the x locations for the groups
    width = 0.5
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(ind, np.array(balance_downgrade), width, color=color_single)
    ax.bar(ind, np.array(balance_no_downgrade), width, bottom=np.array(balance_downgrade), color='grey')
    plt.axhline(y=sum(balance_downgrade) /(sum(balance_downgrade) + sum(balance_no_downgrade)), color='forestgreen' , lw = 3)
    print(sum(balance_downgrade) /(sum(balance_downgrade) + sum(balance_no_downgrade)))
    ax.set_ylabel('Class balance\n', fontweight='bold', fontsize = fontsize_labels)
    #ax.set_xlabel('\nYear', fontweight='bold', fontsize = fontsize_labels)
    ax.set_xticks(ind)
    ax.set_xticklabels(years)
    plt.xticks(rotation=45)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(labelsize = fontsize_ticks)
    
    ax.legend(labels=['avg. class balance', 'downgrade', 'no downgrade'], fontsize = fontsize_labels)
    plt.show()


    return 


def plot_changes_sentences (df):
    df = df[df.sec_year < 2019]
    ax = df[df['change']<= 1].change.hist(bins=50, figsize=(7,7), color = color_single)
    ax.axvline(x=df_sent['change'].median(), color='orangered' , lw = 3)
    print(df['change'].median())
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(labelsize = fontsize_ticks)
    ax.legend(labels=['median', 'number of filings'], fontsize = fontsize_labels)
    ax.set_ylabel('Number of filings\n', fontweight='bold', fontsize = fontsize_labels)
    ax.set_xlabel('\nShare of changed sentences', fontweight='bold', fontsize = fontsize_labels)

    return 



####### Execute code below #########
    
# Settings
fontsize_ticks = 15
fontsize_labels = 17
fontsize_title = 19
color_single = sns.color_palette("Blues")[-1]
color_palette = sns.color_palette("Blues")

sec_meta = import_sec_metadata(sec_data_dir + 'metadata.sqlite3')
plot_length_sections_10K(sec_meta)
plot_publication_month_10K(sec_meta)

df = pd.read_csv(main_dir + "data/df_clean_reports.csv", sep=',')
plot_time_gap_credit_rating_change (df)
plot_class_balance (df)

df_sent = pd.read_csv(main_dir + "data/df_diff_sent.csv", sep=',')
plot_changes_sentences (df_sent)