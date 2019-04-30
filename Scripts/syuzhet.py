import pandas as pd
import numpy as np
import scipy as sp
import sys
import scipy.fftpack as fftpack
from sklearn.neighbors import KernelDensity as KDE
from sklearn.preprocessing import scale

import sqlite3

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML

def sent_by_OHCO(tdb):

    lex_db = '../Lexicon/lexicons.db'
    text_db = tdb

    OHCO = ['chap_num', 'para_num', 'sent_num', 'token_num']
    CHAPS = OHCO[:1]
    PARAS = OHCO[:2]
    SENTS = OHCO[:3]

    emo = 'syu_sentiment'
    kde_kernel = 'gaussian'
    kde_samples = 1000

# Get Lexicons
    with sqlite3.connect(lex_db) as db:
        combo = pd.read_sql('SELECT * FROM combo', db, index_col='term_str')

# Get lexicon columns
    emo_cols = combo.columns

# Get text
    with sqlite3.connect(text_db) as db:
        tokens = pd.read_sql("SELECT * FROM token WHERE punc = 0", db, index_col=OHCO)
        vocab = pd.read_sql("SELECT * FROM vocab", db, index_col='term_id')

# Merge sentiment lexicon with vocab
    tokens = tokens.join(combo, on='term_str', how='left')

# Sentiment by OHCO
    #FIG = dict(figsize=(25, 5), legend=True, fontsize=14, rot=45)

# By chapter
    by_chapter = tokens.groupby(CHAPS)[emo].sum()
# By paragraph
    by_para = tokens.groupby(PARAS)[emo_cols].sum()
    max_x = by_para.shape[0]
    xticks1 = list(range(0, max_x, 100))
    xticks1.append(max_x - 1)
# By sentence
    by_sent = tokens.groupby(SENTS)[emo_cols].sum()
    max_x = by_sent.shape[0]
    xticks2 = list(range(0, max_x, 250))
    xticks2.append(max_x - 1)

    return tokens, by_chapter, by_para, xticks1, by_sent, xticks2

def add_text(tokens, by_sent, emo):

    tokens['html'] =  tokens.apply(lambda x: 
                               "<span class='sent{}'>{}</span>".format(int(np.sign(x[emo])), x.token_str), 1)
    by_sent['sent_str'] = tokens.groupby(SENTS).term_str.apply(lambda x: x.str.cat(sep=' '))
    by_sent['html_str'] = tokens.groupby(SENTS).html.apply(lambda x: x.str.cat(sep=' '))

    return tokens, by_sent

def sample_of_sent(by_sent):

    rows = []
    for idx in by_sent.sample(10).index:
    
        valence = round(by_sent.loc[idx, emo], 4)     
        t = 0
        if valence > t: color = '#ccffcc'
        elif valence < t: color = '#ffcccc'
        else: color = '#f2f2f2'
        z=0
        rows.append("""<tr style="background-color:{0};padding:.5rem 1rem;font-size:110%;">
        <td>{1}</td><td>{3}</td><td width="400" style="text-align:left;">{2}</td>
        </tr>""".format(color, valence, by_sent.loc[idx, 'html_str'], idx))
    
    display(HTML('<style>#sample1 td{font-size:120%;vertical-align:top;} .sent-1{color:red;font-weight:bold;} .sent1{color:green;font-weight:bold;}</style>'))
    display(HTML('<table id="sample1"><tr><th>Sentiment</th><th>ID</th><th width="600">Sentence</th></tr>'+''.join(rows)+'</table>'))

def get_transformed_values(raw_values, low_pass_size = 2, x_reverse_len = 100,  padding_factor = 2, scale_values = False, scale_range = False):

    if low_pass_size > len(raw_values):
        sys.exit("low_pass_size must be less than or equal to the length of raw_values input vector")

    raw_values_len = len(raw_values)
    padding_len = raw_values_len * padding_factor

    # Add padding, then fft
    values_fft = fftpack.fft(raw_values, padding_len)
    low_pass_size = low_pass_size * (1 + padding_factor)
    keepers = values_fft[:low_pass_size]

    # Preserve frequency domain structure
    modified_spectrum = list(keepers) \
        + list(np.zeros((x_reverse_len * (1+padding_factor)) - (2*low_pass_size) + 1)) \
        + list(reversed(np.conj(keepers[1:(len(keepers))])))
    
    
    # Strip padding
    inverse_values = fftpack.ifft(modified_spectrum)
    inverse_values = inverse_values[:x_reverse_len]

#     transformed_values = np.real(tuple(inverse_values))
    transformed_values = np.real(inverse_values)
    return transformed_values

def get_dct_transform(raw_values, low_pass_size = 5, x_reverse_len = 100):
    
    if low_pass_size > len(raw_values):
        raise ValueError("low_pass_size must be less than or equal to the length of raw_values input vector")
    values_dct = fftpack.dct(raw_values, type = 2)
    keepers = values_dct[:low_pass_size]
    padded_keepers = list(keepers) + list(np.zeros(x_reverse_len - low_pass_size))
    dct_out = fftpack.idct(padded_keepers)
    return(dct_out)

def  plot_sentiment(emo_col, type='sent'):
    if type == 'sent':
        by_sent[emo_col].fillna(0).rolling(**CFG1).mean().plot(**FIG)
    elif type == 'tokens':
        tokens[emo_col].fillna(0).rolling(**CFG2).mean().plot(**FIG)
    elif type == 'kde':
        PLOTS[emo_col].plot(**FIG)
    else:
        pass
