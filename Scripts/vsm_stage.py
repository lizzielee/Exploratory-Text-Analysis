import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import re
#% matplotlib inline
import nltk

def get_bow_from_token(K, V, CHAPS):
    # Create word mask
    WORDS = (K.punc == 0) & (K.num == 0) & K.term_id.isin(V[V.stop==0].index)

    CHAPS = CHAPS

	# Extract BOW 
    BOW = K[WORDS].groupby(CHAPS+['term_id'])['term_id'].count()

    return BOW

def bow_to_dtm(BOW):
    DTM = BOW.unstack().fillna(0)

    return DTM

def compute_tf(DTM, V, alpha=.000001):
    alpha_sum = alpha * V.shape[0]
    TF = DTM.apply(lambda x: (x + alpha) / (x.sum() + alpha_sum), axis=1)

    return TF

def compute_tfidf(TF, DTM, V):
    N_docs = DTM.shape[0]
    V['df'] = DTM[DTM > 0].count()
    TFIDF = TF * np.log2(N_docs / V[V.stop==0]['df'])

    return TFIDF

def compute_tfth(TF):
    THM = -(TF * np.log2(TF))
    TFTH = TF.apply(lambda x: x * THM.sum(), 1)

    return TFTH

def add_stats_to_V(V, TF, TFIDF, TFTH):
    THM = -(TF * np.log2(TF))
    V['tf_sum'] = TF.sum()
    V['tf_mean'] = TF.mean()
    V['tf_max'] = TF.max()
    V['tfidf_sum'] = TFIDF.sum()
    V['tfidf_mean'] = TFIDF.mean()
    V['tfidf_max'] = TFIDF.max()
    V['tfth_sum'] = TFTH.sum()
    V['tfth_mean'] = TFTH.mean()
    V['tfth_max'] = TFTH.max()
    V['th_sum'] = THM.sum()
    V['th_mean'] = THM.mean()
    V['th_max'] = THM.max()

    return V

def find_sig_words(V):
    TOPS = pd.DataFrame(index=range(10))
    for m in ['tf','tfidf','tfth', 'th']:
        for n in ['mean','max']:
            key = '{}_{}'.format(m,n)
            TOPS[key] = V.sort_values(key, ascending=False).term_str.head(10).tolist()

    return TOPS
