import pandas as pd
import sqlite3
import textman as tx

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def pre_mallet(db):

    corpus_db = db
    max_words = 10000

    # For MALLET
    num_topics = 20
    num_iters = 1000
    show_interval = 100

    sql = """
    SELECT * FROM token 
    WHERE term_id IN (
        SELECT term_id FROM vocab 
        WHERE stop = 0 
        AND term_str NOT IN ('said')
        ORDER BY tfidf_sum DESC LIMIT {}
    )
-- AND (author = 'poe' OR author = 'austen') 
    AND (pos NOT LIKE 'NNP%')
    """.format(max_words)

    with sqlite3.connect(corpus_db) as db:
        tokens = pd.read_sql(sql, db)

    tokens = tokens.set_index(['book','chapter'])

    corpus = tx.gather_tokens(tokens, level=2, col='term_str')\
        .reset_index().rename(columns={'term_str':'doc_content'})
    corpus['doc_label'] = corpus.apply(lambda x: "doyle-{}-{}".format(x.book, x.chapter), 1)

    return corpus

def get_table(table, db_file, fields='*', index_col=None):
    
    db_file = db_file
    if type(fields) is list:
        fields = ','.join(fields)
    with sqlite3.connect(db_file) as db:
        return pd.read_sql("select {} from {}".format(fields, table), db, index_col=index_col)

def import_model_tables(model_db_file):
    
    doc = get_table('doc', model_db_file, index_col=['doc_id'])
    topic = get_table('topic', model_db_file,  index_col=['topic_id'])
    doctopic = get_table('doctopic', model_db_file, ['doc_id','topic_id','topic_weight'], ['doc_id','topic_id'])
    topicword = get_table('topicword', model_db_file, ['topic_id','word_id','word_count'], ['topic_id','word_id'])
    docword = get_table('docword', model_db_file, index_col=['doc_id','word_pos'])
    vocab = get_table('word', model_db_file)

    return doctopic, topicword, docword, vocab



