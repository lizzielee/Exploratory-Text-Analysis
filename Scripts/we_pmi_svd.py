import pandas as pd
import numpy as np
import sqlite3
import sys; sys.path.append(lib)
from textman import textman as tx

def get_skipgram(proj_dir, db_file, window):

    proj = proj_dir
    pwd = '{}/play/wordembedding'.format(proj)
    db_file = "{}/data/{}".format(proj, db_file)
    lib = "{}/lib".format(proj)

# Word Embedding    
    window = window

# Build SQL query
    in_clause = ', '.join(['x.token_num + {0}, x.token_num - {0}'.format(i) for i in range(1, window + 1)])
    pos_clause = "AND pos NOT LIKE 'NNP%' " # Remove proper nouns

    sql = """
    WITH mytoken(book, chapter, para_num, sent_num,token_num,term_str,term_id) 
    AS (
        SELECT book, chapter, para_num, sent_num,token_num,term_str,term_id
        FROM token 
        WHERE term_id IN (SELECT term_id FROM vocab WHERE stop = 0) 
            AND term_str is not NULL
            {}       
    )

    SELECT x.term_str as target, y.term_str as probe, (y.token_num - x.token_num) AS dist
    FROM mytoken x 
    JOIN mytoken y USING(book, chapter, para_num, sent_num)
    WHERE y.token_num IN ({})
    ORDER BY target, dist, probe
    """.format(pos_clause, in_clause)

# Pull from DB
    skipgrams = tx.get_sql(sql, db_file)

    vocab = tx.get_table('vocab', db_file, index_col=['term_id'])
    vocab = vocab[vocab.stop == 0]
    vocab.sort_values('p', ascending=False)

    p_x = vocab[['term_str','p']].reset_index().set_index('term_str')['p']
    p_x.sort_values(ascending=False).head()

# Create compressed skipgram table
    skipgrams2 = skipgrams.groupby(['target','probe']).probe.count()\
        .to_frame().rename(columns={'probe':'n'})\
        .reset_index().set_index(['target','probe'])

    N = skipgrams2.n.sum()
    skipgrams2['p_xy'] = skipgrams2.n / N

    skipgrams2['pmi_xy'] = skipgrams2.apply(lambda row: np.log(row.p_xy / (p_x.loc[row.name[0]] * p_x.loc[row.name[1]])), 1)
    skipgrams2.sort_values('pmi_xy', ascending=False)
    skipgrams2['npmi_xy'] = skipgrams2.pmi_xy / -( np.log(skipgrams2.p_xy) )
    skipgrams2.sort_values('npmi_xy', ascending=False)

# PMI matrix
    SGM = skipgrams2.npmi_xy.unstack().fillna(0)
    SGM.loc['intestine'].sort_values(ascending=False)
    skipgrams2.loc['intestine'].sort_values('n', ascending=False)

    return SGM

def get_svd(SGM):

    import scipy as sp 

    sparse = sp.sparse.csr_matrix(SGM.values)
    SVD = sp.sparse.linalg.svds(sparse, k=256)
    U, S, V = SVD
    word_vecs = U + V.T
    word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs, axis=1, keepdims=True))

    WE = pd.DataFrame(word_vecs_norm, index=SGM.index)
    WE.index.name = 'word_str'

    return WE

def word_sims(word, SGM, n=10):

    try:
        sims = SGM.loc[word].sort_values(ascending=False).head(n).reset_index().values
        return sims
    except KeyError as e:
        print('Word "{}" not in vocabulary.'.format(word))
        return None

def word_sim_report(word, SGM):
    
    sims = word_sims(word, SGM)
    for sim_word, score in sims:
        context = ' '.join(skipgrams2.loc[sim_word].index.values.tolist()[:5])
        print("{} ({}) {}".format(sim_word.upper(), score, context))
        print('-'*80)