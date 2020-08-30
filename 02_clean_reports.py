#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Import packages #########

import pandas as pd
import re
import os
from tqdm import tqdm
from _settings import main_dir


####### Define functions #########

def manual_removal_reports (df, remove_reports_list_location):

    # Delete reports with no risk section (e.g. listed in MD&A)
    remove_reports = open(remove_reports_list_location, 'r')
    remove_reports_list = remove_reports.readlines()
    for i in remove_reports_list:
        idx = df[(df['id'] == int(i))].index.values
        df = df.drop(idx)
        
    # Remove duplicates   
    all_duplicates_drop = df[df[['cik', 'sec_period_of_report', 'sec_filing_date','section_n_characters']].duplicated(keep = 'last') == True].index.tolist()
    df = df.drop(all_duplicates_drop)

    # Manual check, adjustment of special cases
    #326409: 2005 reports are out of scope
    #286654, 325384, 325388: report published after following report was published (excluded from model)
    id_manual_drop = [326409, 286654, 325384, 325388]
    
    for i in id_manual_drop:
        idx = df[(df['id'] == i)].index.values
        df = df.drop(idx)   
      
    # Add columns for filing year
    df['sec_year'] = df['sec_filing_date'].astype(str).str[0:4]
    
    
    # Manual check, adjustment of special cases
    id_manual_change = {227674:'2007', 274632:'2010',295133:'2010',336442:'2008'}
    
    for i in id_manual_change:
        idx = df[(df['id'] == i)].index.values
        df.loc[idx,'sec_year'] = id_manual_change[i]
    
    
    # Remove 10-Q reports
    df = df.drop(df[df['document_group']== '10-Q'].index.tolist())
    
    return df

 
def num_there(s):
    return any(i.isdigit() for i in s)


def remove_footer (text):
    matches = re.findall(r'\n.{1,50}\n', text)
    if matches:
        newlines = []
        newlines.extend(matches)
        newlines = set(newlines)
        newlines_temp = newlines.copy()
        
        for i in newlines_temp:
            if i.find("10-K") != -1:
                continue
            if i.find("|") != -1:
                continue
            if i.find("wwww.") != -1:
                continue
            if i.find("Annual Report") != -1:
                continue
            if i.find("Index") != -1:
                continue
            if i.find("<PAGE>") != -1:
                continue
            if i.isupper() and num_there(i):
                continue
            else:
                newlines.remove(i)
        if newlines:
            for l in newlines:
                p = re.compile(l)
                text =  p.sub("", text)
    return text


def flag_and_clean_reports (df):
    
    df["manual_check"] = False
    df["head_trimmed"] = False
    df["foot_trimmed"] = False
    df["very_short"] = False
    df["table_of_content_text"] = False
    df["start_position"] = 0
    df["end_position"] = 0

    for index, row in tqdm(df.iterrows()):
        text = row["text"]
        
        # Remove start line from extracting process
        start = row["start_line"]
        try:
            start_idx = text.index(start)
            df.loc[index,'text'] = text[start_idx + len(start):]
            df.loc[index,'head_trimmed'] = True
            
        except:
            df.loc[index,'manual_check'] = True
            
        # Search for Item1B at the end and remove the text afterwards. 
        #Flag reports where this line is in the first 90% of the document
        text = row["text"]
        matches = re.findall(r'ITEM.{,10}1.{,10}B.{,10}UNRESOLVED.{,5}STAFF.{,5}COMMENT', text, re.DOTALL | re.IGNORECASE)
        if len(matches) > 0:
            for m in matches:
                pos = text.find(m)
                if pos != -1:
                    rel_pos = pos/len(text)
                    if rel_pos > 0.9:
                        df.loc[index,'text'] = text[:pos]
                        df.loc[index,'foot_trimmed'] = True
                    else:
                        df.loc[index,'manual_check'] = True
    
        # Search for Item2, 3 etc. at the end and remove the text afterwards. 
        #Flag reports where this line is in the first 90% of the document
        text = row["text"]
        matches = re.findall(r'ITEM.{,5}[2-9]', text, re.DOTALL | re.IGNORECASE)
        if len(matches) > 0:
            for m in matches:
                pos = text.find(m)
                if pos != -1:
                    rel_pos = pos/len(text)
                    if rel_pos > 0.9:
                        df.loc[index,'text'] = text[:pos]
                        df.loc[index,'foot_trimmed'] = True
                    else:
                        df.loc[index,'manual_check'] = True
        
        # Flag reports that are too short
        text = row["text"]
        if len(text) < 3000:
            df.loc[index,'very_short'] = True
            df.loc[index,'manual_check'] = True
            
            
        # Remove table of contens in footer of each page
        text = row["text"]
        matches = re.findall(r'Table.{,2}of.{,2}contents?', text, re.DOTALL | re.IGNORECASE)
        if len(matches) > 0:
            df.loc[index,'text'] = re.sub(r'Table.{,2}of.{,2}contents?', '', text, flags= re.DOTALL | re.IGNORECASE)
            df.loc[index,'table_of_content_text'] = True
            
            
        # Remove page count in footer of each page
        text = row["text"]
        matches = re.findall(r'Page.{,5}of', text, re.DOTALL | re.IGNORECASE)
        if len(matches) > 0:
            df.loc[index,'text'] = re.sub(r'Page.{,5}of', '', text, flags= re.DOTALL | re.IGNORECASE)
            df.loc[index,'table_of_content_text'] = True
        text = row["text"]
        df.loc[index,'text'] = re.sub(r'<PAGE>', '', text, flags= re.DOTALL | re.IGNORECASE)
    
        # Detect Table of contents at beginning
        text = row["text"].upper()
        pos = text.upper().find("..........")
        if pos != -1:
            df.loc[index,'manual_check'] = True
    
        # Remove Tables from text
        text = row["text"]
        df.loc[index,'text'] = re.sub(r'\[DATA_TABLE_REMOVED.{,15}\]', '', text)
    
    
        # Remove footers from text
        text = row["text"]
        company_names= {814585: "ITEM A. Risk Factors (continued)", 
                        882829: "Item A.   Risk Factors (contd.)", 
                        1323531: "I Maidenform Brands, Inc.", 
                        1164727: "NEWMONT MINING CORPORATION", 
                        1216596: "NORTEK, INC. AND SUBSIDIARIES", 
                        1133421: "NORTHROP GRUMMAN CORPORATION", 
                        1001606: "BLOUNT INTERNATIONAL, INC.", 
                        316709: "THE CHARLES SCHWAB CORPORATION"}
        df.loc[index,'text'] = remove_footer(text)
        if int(row["cik"]) in company_names.keys():
            cik = int(row["cik"])
            p = re.compile(company_names[cik])
            df.loc[index,'text'] = p.sub("", text)

    return df


def manual_trim_reports(df, idx_file_location):

    # Manual trim reports
    manual_start_end_reports = pd.read_csv(idx_file_location, sep=';')
    id_start_end = manual_start_end_reports['id'].values.tolist()
    
    k = 0
    for index,row in df[df['id'].isin(id_start_end)].iterrows():
        id_ = row['id']
        idx_start = manual_start_end_reports[manual_start_end_reports['id'] == id_]['start_position'].values[0]
        idx_end = manual_start_end_reports[manual_start_end_reports['id'] == id_]['end_position'].values[0]
        text = row['text']
        df.loc[index,'text'] = text[int(idx_start) : int(idx_end)]
        df.loc[index,'manual_check'] = False
        k +=1 
        
    return df



def save_data_csv (data, filename, path, index = False):
    if not os.path.exists(path):
        os.makedirs(path)
    filePath = path + filename + '.csv'
    return data.to_csv(filePath, sep = ",", index = index, header=True)



####### Execute code below #########

df = pd.read_csv(main_dir + "data/df_create_dataset.csv", sep=',')
df_clean = manual_removal_reports (df, main_dir + 'assets/remove_reports.txt')
df_clean = flag_and_clean_reports (df_clean)
df_clean = manual_trim_reports(df_clean, main_dir + 'assets/adjust_reports.csv')
save_data_csv (df_clean, "df_clean_reports", main_dir + "data/", index = True)

