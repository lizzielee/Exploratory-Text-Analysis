import sqlite3
import pandas as pd
import re

def collect(bpf, bnf, nrc, syuzhet, gi):
    
    bing_pos_file = bpf
    bing_neg_file = bnf
    nrc_file = nrc
    syuzhet_file = syuzhet
    gi_file = gi
    #my_pwd = '/Users/rca2t/COURSES/DSI/DS5559/UVA_DSI_REPO/labs/2019-04-11_Lab11'

# Import Bing lexicon
    bing_list = [(word.strip(), 'bing_positive', 1) 
                for word in open(bing_pos_file, 'r').readlines() 
                if re.match(r'^\w+$', word)]
    bing_list += [(word.strip(), 'bing_negative', 1) 
                for word in open(bing_neg_file, 'r').readlines() 
                if re.match(r'^\w+$', word)]

    bing = pd.DataFrame(bing_list, columns=['term_str', 'polarity', 'val'])
    bing = bing.set_index(['term_str','polarity'])
    bing = bing.unstack().fillna(0).astype('int')
    bing.columns = bing.columns.droplevel(0)

# Create single sentiment column
    bing['bing_sentiment'] = bing['bing_positive'] - bing['bing_negative']

# Import NRC lexicon
    nrc = pd.read_csv(nrc_file, sep='\t', header=None)
    nrc.columns = ['term_str','nrc_emotion','val']
    nrc = nrc.set_index(['term_str','nrc_emotion'])
    nrc = nrc.unstack()
    nrc.columns = nrc.columns.droplevel(0)
    nrc = nrc[nrc.sum(1) > 1]
    nrc.columns = ['nrc_'+col for col in nrc.columns]

# Import Syuzhet lexicon
    syu = pd.read_csv(syuzhet_file)
    syu.columns = ['id','term_str','syu_sentiment']
    syu = syu.drop('id', 1)
    syu = syu.set_index('term_str')

# Import General Iquirer lexicon
    gi = pd.read_csv(gi_file, index_col=['term_str'])
    gi.columns = ['gi_sentiment']

# Combine all
    combo = nrc.join(bing, how='outer')\
        .join(syu, how='outer')\
        .join(gi, how='outer')\
        .sort_index()

# Save
    with sqlite3.connect('lexicons.db') as db:
        nrc.to_sql('nrc', db, index=True, if_exists='replace')
        bing.to_sql('bing', db, index=True, if_exists='replace')
        syu.to_sql('syuzhet', db, index=True, if_exists='replace')
        gi.to_sql('gi', db, index=True, if_exists='replace')
        combo.to_sql('combo', db, index=True, if_exists='replace')  

    return combo