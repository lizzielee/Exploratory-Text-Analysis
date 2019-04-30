import pandas as pd
import numpy as np
import sqlite3
#import sys; sys.path.append(lib)
import textman as tx

import sys
import gensim
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_we_coords(db):
    
    db_file = db
    OHCO = ['book','chapter','para_num','sent_num']
    tokens = tx.get_table('token', db_file)
    tokens = tokens[~tokens.term_str.isna()]
    corpus = tokens.groupby(OHCO).term_str.apply(lambda  x:  x.tolist())\
        .reset_index()['term_str'].tolist()

    model = word2vec.Word2Vec(corpus, size=246, window=5, min_count=200, workers=4)

    coords = pd.DataFrame(index=range(len(model.wv.vocab)))
    coords['label'] = [w for w in model.wv.vocab]
    coords['vector'] = coords['label'].apply(lambda x: model.wv.get_vector(x))

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    tsne_values = tsne_model.fit_transform(coords['vector'].tolist())

    coords['x'] = tsne_values[:,0]
    coords['y'] = tsne_values[:,1]

    return coords

def plot_coords(coords):

    plt.figure(figsize=(16, 16)) 
    for i in range(len(coords)):
        plt.scatter(coords.x[i],coords.y[i])
        plt.annotate(word_labels[i],
                    xy=(coords.x[i], coords.y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    plt.show()



