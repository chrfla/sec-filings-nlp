#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Import packages #########

import numpy as np
import wrds
import sqlite3
import pandas as pd
import datetime
import os
from tqdm import tqdm
from _settings import main_dir, sec_data_dir, wrds_user_name

####### Define functions #########

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except :
        print("Connection to database has faield.")

    return conn


def get_risk_data ():

    #Get all risk data
    ratings = db.raw_sql('SELECT * FROM ciq_ratings.wrds_erating as t')

    #Filter for unique company ids
    company_ids = ratings.company_id.unique()
    company_ids = pd.DataFrame(data = company_ids, columns=["company_id"])
    company_ids = company_ids.rename(columns = {'company_id':'companyid'})

    # Get mapping table
    companyid_gvkey_map = db.raw_sql('SELECT * FROM ciq_common.wrds_gvkey as t')

    #Left join (include all comany_ids)
    match_ids = pd.merge(company_ids,
                         companyid_gvkey_map,
                         on='companyid',
                         how='left')

    # Identify all gvkeys used for comany_ids
    match_ids = match_ids.dropna(subset=['gvkey'])
    match_ids['double'] = False
    for index, row in match_ids.iterrows():
        key = row["companyid"]
        temp = match_ids[match_ids['companyid']==key]
        if len(temp) > 1:
            match_ids.at[index, 'double'] = True
            if row["enddate"]:
                if row["enddate"] < datetime.date(2006, 1, 1):
                    match_ids.drop(index, inplace = True)

    #Filter for double_gvkey
    two_gvkey = match_ids[match_ids['double'] == True]

    return ratings, match_ids, two_gvkey


def prepare_company_ticker (companies, location_txt):
    output = pd.DataFrame(columns = ['gvkey', 'tic', 'cik'])

    #Store ticker and cik in txt file
    with open(location_txt, 'w') as f:
        for index, row in companies.iterrows():
            tic = row['tic']
            cik = row['cik']
            gvkey = row['gvkey']
            companyid = row['companyid']

            # Remove ending numbers of tickers
            try:
                tic = tic.split(".")[0]
            except:
                tic = tic
            # Write every cik/ticker in one row
            f.write("%s %s\n" % (int(cik), tic))

            output = output.append({'gvkey' : gvkey,
                                    'companyid' : companyid,
                                    'tic' : tic,
                                    'cik' : cik } , ignore_index=True)
    return output


def get_cik_tic (company_ids, location_txt):
    #Merge company_ids with cik and ticker on gvkey
    names = db.raw_sql('SELECT * FROM comp.names as t')
    companies = pd.merge(company_ids,
                         names,
                         on='gvkey',
                         how='left')
    companies = companies[companies.cik.notna()]

    #Prepare sec txt file
    sec_companies = prepare_company_ticker (companies, location_txt)

    return companies , sec_companies



def extract_rating_changes(data, companies, startyear):

    # Rating classes decreasingly
    ratings_classes = ['AAA', 'AAA-',
                       'AA+', 'AA','AA-',
                       'A+','A', 'A-',
                       'BBB+','BBB', 'BBB-',
                       'BB+', 'BB', 'BB-',
                       'B+', 'B', 'B-',
                       'CCC+','CCC', 'CCC-',
                       'CC+', 'CC', 'CC-',
                       'C+', 'C', 'C-',
                       'SD', 'D']

    ratings = data

    # Filter rating data
    ratings = ratings[ratings['rating'].isin(ratings_classes)]
    ratings = ratings[ratings['rtype'].isin(['Local Currency LT'])]
    ratings = ratings[ratings['orgdtype'].isin(['Issuer Credit Rating'])]
    ratings = ratings[ratings['unsol'].isin(['N'])]
    ratings = ratings.drop_duplicates(subset=ratings.columns.difference(['rdid', 'entity_pname', 'startdate', 'enddate']), keep='last')
    ratings['reced'] = ratings['reced'].astype("datetime64[ns]")
    ratings['recud'] = ratings['recud'].astype("datetime64[ns]")

    print(str(len(ratings[(ratings['reced'] > np.datetime64('2006-01-01')) & (ratings['reced'] < np.datetime64('2020-01-01')) ])) + ' events in Capital IQ the database')
    print(str(len(ratings[(ratings['reced'] > np.datetime64('2006-01-01')) & (ratings['reced'] < np.datetime64('2020-01-01')) ].company_id.unique())) + ' companies in Capital IQ the database')

    output = pd.DataFrame(columns = ['companyid', 'rating_date', 'rating_prev', 'rating_new'])

    n_no = 0

    # Loop through every companyid
    for cid in tqdm(companies.companyid.unique()):
        key = cid


        # filter rating for one company id
        filtered = ratings[ratings['company_id']==key]
        # Sort by reced ascending
        filtered = filtered.sort_values(by='reced', ascending=True)
        # Reset index
        filtered = filtered.reset_index()


        # No entry (no change possible)
        if len(filtered.company_id) < 1:
            n_no += 1
            continue

        else:

            for index, row in filtered.iterrows():

                # Deatils on ratings https://wrds-web.wharton.upenn.edu/wrds/ds/comp/ciq/ratingsEntity/index.cfm?navId=66

                #If last row of filtered
                if index == len(filtered.company_id) - 1:

                    # if oldest rating as no ending date, set old to None and new to selected rating
                    if row['reced'] is None and index == 0:
                        old_rating = None
                        new_rating = filtered.iloc[index].rating
                        rating_date = filtered.iloc[index].recud

                    # else if rating has no ending date but is not the oldest date, ignore
                    elif row['reced'] is None and index > 0:
                        continue

                    # else if rating has an ending date, set new to None and old to current rating
                    elif row['reced'] is not None:
                        new_rating = None
                        old_rating = filtered.iloc[index].rating
                        rating_date = filtered.iloc[index].reced

                    # else print an error message
                    else:
                        print("Check {} (last item)".format(key))
                        continue

                #If not last row of filtered
                else:

                    # if rating has an ending date, set old to current rating and new to the rating afterwads (index + 1, filtered was sorted ascending)
                    if row['reced'] is not None:
                        old_rating = filtered.iloc[index].rating
                        rating_date = filtered.iloc[index].reced
                        new_rating = filtered.iloc[index + 1].rating

                    # else print error (should not be called, since last row in filtered is handeled in if class in a level above)
                    else:
                        print("Check {}".format(key))
                        continue

                # append resulting rating to dataframe
                output = output.append({
                                        'companyid' : key,
                                        'rating_date' : rating_date,
                                        'rating_prev' : old_rating,
                                        'rating_new' : new_rating} , ignore_index=True)


    #output = output[output['rating_prev'] != output['rating_new']]

    #exclude all ratings before period of interest
    output = output[output['rating_date'] >= pd.Timestamp(startyear - 0, 1, 1)]
    output = output[output['rating_date'] < pd.Timestamp(2020, 1, 1)]
    #output = output[pd.notnull(output['rating_new'])]
    #output = output[pd.notnull(output['rating_prev'])]

    print("{} companies with no rating".format(n_no))

    return output



def add_rating_labels (rating_changes):

    rating_changes_data = rating_changes
    ratings_classes = ['AAA', 'AAA-',
                       'AA+', 'AA', 'AA-',
                       'A+','A', 'A-',
                       'BBB+', 'BBB', 'BBB-',
                       'BB+', 'BB', 'BB-',
                       'B+', 'B', 'B-',
                       'CCC+', 'CCC', 'CCC-',
                       'CC+', 'CC', 'CC-',
                       'C+', 'C', 'C-',
                       'SD', 'D']
    # Add columns
    rating_changes_data['rating_change'] = 0
    rating_changes_data['rating_downgrade'] = False
    rating_changes_data['rating_upgrade'] = False

    for index, row in rating_changes_data.iterrows():
        old = row["rating_prev"]
        new = row["rating_new"]
        # If no new rating, set change to False and change to 99
        if new is None:
            rating_changes_data.at[index, 'rating_change'] = 99
            rating_changes_data.at[index, 'rating_downgrade'] = False

        # Convert change in integer
        else:
            change = ratings_classes.index(old) - ratings_classes.index(new)
            rating_changes_data.at[index, 'rating_change'] = change
            # if negative change, downgrade
            if change < 0:
                rating_changes_data.at[index, 'rating_downgrade'] = True
                rating_changes_data.at[index, 'rating_upgrade'] = False
             # otherwise, no downgrade
            elif change > 0:
                rating_changes_data.at[index, 'rating_downgrade'] = False
                rating_changes_data.at[index, 'rating_upgrade'] = True
             # otherwise, no downgrade
            else:
                rating_changes_data.at[index, 'rating_downgrade'] = False
                rating_changes_data.at[index, 'rating_upgrade'] = False

    return rating_changes_data



def import_sec_metadata(path_sec_db):

    # import SQLite database with SEC data
    connection = create_connection(path_sec_db)
    cursor = connection.cursor()
    df = pd.read_sql_query("SELECT * FROM metadata WHERE section_name = 'Item1A' AND output_file is not null and sec_form_header = 'Form 10-K' and batch_number = 3", connection)

    # Convert date into date format
    tmp = pd.to_datetime(df['sec_filing_date'], format = '%Y%m%d')
    df.insert(len(df.columns), "sec_date", tmp)
    df['sec_date'] = df['sec_date'].astype("datetime64[ns]")
    df = df[df['sec_date'] < np.datetime64('2019-01-01')]
    # Rename column to match other datastructures
    df = df.rename(columns = {'sec_cik':'cik'})

    return df




#merging sec and risk
#check no rating assumption (example, cik: 896985)

def merge_sec_risk(sec_companies, rating_changes, sec_reports, location_output_sec_report, delay_days = 365):
    n = 0
    n_no = 0
    # add columns to sec_reports df
    sec_reports['companyid'] = 99999999
    sec_reports['rating_date'] = datetime.date(1900, 1, 1)
    sec_reports['rating_prev'] = "XX"
    sec_reports['rating_new'] = "XX"
    sec_reports['rating_change'] = 99
    sec_reports['rating_upgrade'] = False
    sec_reports['rating_downgrade'] = False
    sec_reports['default'] = False
    sec_reports['text'] = "abc"


    unique_cik = sec_reports.cik.unique()
    deleted_report_cik = []
    filtered_ciks = sec_companies[sec_companies["cik"].isin(unique_cik)]
    print(str(len(unique_cik)) + ' companies in Capital IQ and SEC')
    print(str(len(sec_reports[(sec_reports["cik"].isin(filtered_ciks.cik.unique())) & (sec_reports["sec_date"] < np.datetime64('2019-01-01'))])) + ' observation in Capital IQ and SEC')

    # Loop through every company (cik)
    for key in tqdm(unique_cik):
        cik = key
        filtered = sec_companies[sec_companies["cik"] == cik]
        # get all rating changes that use the cik
        rating_c_filtered = rating_changes[rating_changes["companyid"].isin(filtered["companyid"].tolist())]
        # if no ratings can be found, delete all reports with corresponding cik
        if len(rating_c_filtered) == 0:
            delete = sec_reports[sec_reports["cik"] == cik]
            sec_reports = sec_reports.drop(delete.index.values.tolist())
            deleted_report_cik.append(cik)
            n_no += 1
        else:

            # Filter sec reports with current cik
            sec_reports_filtered = sec_reports[sec_reports["cik"] == cik]

            # Loop through every report of cik
            for index, row in sec_reports_filtered.iterrows():

                # Get correspdonig risk section text
                file_name = row['output_file']
                file_name_adj = location_output_sec_report + file_name
                file = open(file_name_adj,"r")
                text = file.read()

                # Get sec filing date and calculate delta latest credit rating
                sec_date = row["sec_date"]
                sec_date_end = sec_date + datetime.timedelta(days = delay_days)
                # Filter credit rating for rating after sec filing and before latest date with a downgrade
                downgrade_data = rating_c_filtered[(rating_c_filtered["rating_date"] >= sec_date) & (rating_c_filtered["rating_date"] < sec_date_end) & (rating_c_filtered["rating_downgrade"] == True)]
                upgrade_data = rating_c_filtered[(rating_c_filtered["rating_date"] >= sec_date) & (rating_c_filtered["rating_date"] < sec_date_end) & (rating_c_filtered["rating_upgrade"] == True)]

                # If no downgrade occured, at info to df (firm had other credit rating before)
                if len(downgrade_data) == 0:
                    if len(upgrade_data) == 0:

                        sec_reports.at[index, 'rating_change'] = 0
                        sec_reports.at[index, 'rating_downgrade'] = False
                        sec_reports.at[index, 'rating_upgrade'] = False
                        sec_reports.at[index, 'text'] = text
                    else:
                        # get indx for smallest rating date (closes downgrade to the sec publication)
                        minidx = upgrade_data.rating_date.idxmin()
                        # get corresponding data based on the index of smallest date
                        sec_reports.at[index, 'rating_date'] = upgrade_data.rating_date[minidx]
                        sec_reports.at[index, 'rating_prev'] = upgrade_data.rating_prev[minidx]
                        sec_reports.at[index, 'rating_new'] = upgrade_data.rating_new[minidx]
                        sec_reports.at[index, 'rating_change'] = upgrade_data.rating_change[minidx]
                        sec_reports.at[index, 'rating_upgrade'] = True
                        sec_reports.at[index, 'text'] = text


                else:
                    # get indx for smallest rating date (closes downgrade to the sec publication)
                    minidx = downgrade_data.rating_date.idxmin()
                    # get corresponding data based on the index of smallest date
                    sec_reports.at[index, 'rating_date'] = downgrade_data.rating_date[minidx]
                    sec_reports.at[index, 'rating_prev'] = downgrade_data.rating_prev[minidx]
                    sec_reports.at[index, 'rating_new'] = downgrade_data.rating_new[minidx]
                    sec_reports.at[index, 'rating_change'] = downgrade_data.rating_change[minidx]
                    sec_reports.at[index, 'rating_downgrade'] = True
                    sec_reports.at[index, 'text'] = text

                    # add information of default (assuming that a firm has only one period of default)
                    default_check = downgrade_data[downgrade_data["rating_new"].isin(["D", "SD"])]
                    if len(default_check) > 0:
                        sec_reports.at[index, 'default'] = True

    print("Did not find any ratings for {} companies".format(n_no))
    print()
    print("{} observations with no change in credit rating".format(len(sec_reports[sec_reports["rating_downgrade"] == False])))
    print("{} observations with downgrade in credit rating".format(len(sec_reports[sec_reports["rating_downgrade"] == True])))
    print("{} observations with defaults".format(len(sec_reports[sec_reports["default"] == True])))
    print("{} different firms".format(len(sec_reports["cik"].unique())))

    return sec_reports, deleted_report_cik


def save_data_csv (data, filename, path, index = False):
    if not os.path.exists(path):
        os.makedirs(path)
    filePath = path + filename + '.csv'
    return data.to_csv(filePath, index = index, header=True)



####### Execute code below #########

db = wrds.Connection(wrds_username = wrds_user_name)

ratings_all, company_ids, two_gvkey = get_risk_data()

all_companies, sec_companies = get_cik_tic(company_ids, main_dir + "assets/cik_ticker.txt")

rating_changes = extract_rating_changes(ratings_all, all_companies, 2006)

rating_changes_labels = add_rating_labels (rating_changes)

sec_meta = import_sec_metadata(sec_data_dir + 'metadata.sqlite3')

datatset, deleted_report_cik = merge_sec_risk(sec_companies, rating_changes_labels, sec_meta, sec_data_dir)

save_data_csv (datatset, "df_create_dataset", main_dir + "data/", index = False)
