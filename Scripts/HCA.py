import pandas as pd
import sqlite3

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

%matplotlib inline

def get_hca(db):

    db_file = db

    with sqlite3.connect(db_file) as db:
        bags = pd.read_sql("SELECT * FROM bag", db, index_col='bag_id')
        tfidf = pd.read_sql("SELECT * FROM tfidf_small", db, index_col=['bag_id','term_id'])

    TFIDF = tfidf.unstack()
    TFIDF.columns = TFIDF.columns.droplevel(0)

    labels = bags.apply(lambda x: ' '.join(x.astype('str')), 1).tolist()

    # generate similarity pairs
    SIMS = pdist(TFIDF, metric='cosine')

    TREE = sch.linkage(SIMS, method='ward')

    return TREE, labels

def plot_tree(tree, labels):
    plt.figure()
    fig, axes = plt.subplots(figsize=(10, 60))
    dendrogram = sch.dendrogram(tree, labels=labels, orientation="left")
    plt.tick_params(axis='both', which='major', labelsize=10)


