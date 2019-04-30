import sqlite3
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

%matplotlib inline

def get_pca(db, n_components=10):

    db_file = db

    with sqlite3.connect(db_file) as db:
        bags = pd.read_sql("SELECT * FROM bag", db, index_col='bag_id')
        tfidf = pd.read_sql("SELECT * FROM tfidf_small", db, index_col=['bag_id','term_id'])    
        vocab = pd.read_sql("select * from vocab", db, index_col='term_id')

    TFIDF = tfidf.unstack()
    TFIDF.columns = TFIDF.columns.droplevel(0)
    vocab_idx = TFIDF.columns

    pca = PCA(n_components)
    projected = pca.fit_transform(normalize(TFIDF.values, norm='l2'))

    genres = bags.genre
    letters = genres.unique().tolist()
    genre_ids = genres.apply(lambda x: letters.index(x)).values

    return projected, genre_ids

def plot_pca(subspace, labels, pc_x=0, pc_y=1, figsize=(15,10), annotate=False):
    plt.figure(figsize=figsize)
    plt.scatter(projected[:, pc_x], 
                projected[:, pc_y],
                c=labels, 
                edgecolor='none', 
                alpha=0.5,
                cmap=plt.cm.get_cmap('terrain', 10))
    plt.xlabel('PC{}'.format(pc_x))
    plt.ylabel('PC{}'.format(pc_y))
    if annotate:
        for i, x in enumerate(subspace):
            plt.annotate(bags.loc[i].author, (x[pc_x], x[pc_y])) # SHOULD BE AN FUNCTION ARG


