import re
import os
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('tagsets')
nltk.download('wordnet')


def corpus_import(start, end, file, src_file):
    
    OHCO = ['chap_num', 'para_num', 'sent_num', 'token_num']
    CHAPS = OHCO[:1]
    PARAS = OHCO[:2]
    SENTS = OHCO[:3]

    #% matplotlib inline

    body_start = start
    body_end = end
    chap_pat = r'^\s*Chapter.*$'
    para_pat = r'\n\n+'
    sent_pat = r'([.;?!"“”]+)'
    token_pat = r'([\W_]+)'
    db_file = file
    src_file_name = src_file

    extra_stopwords = """
    us rest went least would much must long one like much say well without though yet might still upon
    done every rather particular made many previous always never thy thou go first oh thee ere ye came
    almost could may sometimes seem called among another also however nevertheless even way one two three
    ever put
    """.strip().split()

    # text to lines
    lines = open(src_file_name, 'r', encoding='utf-8').readlines()
    lines = lines[body_start - 1 : body_end + 1]
    df = pd.DataFrame({'line_str':lines})
    df.index.name = 'line_id'
    del(lines)

    df.line_str = df.line_str.str.replace('—', ' — ')
    df.line_str = df.line_str.str.replace('-', ' - ')

    # lines to chapters
    chap_mask = df.line_str.str.match(chap_pat)
    df.loc[chap_mask, 'chap_id'] = df.apply(lambda x: x.name, 1)
    #df.chap_id.replace(np.nan, 0, inplace=True)
    #df.chap_id.replace(np.inf, 0, inplace=True)
    df.chap_id = df.chap_id.fillna(0).astype('int')
    chap_ids = df.chap_id.unique().tolist()
    df['chap_num'] = df.chap_id.apply(lambda x: chap_ids.index(x))
    chaps = df.groupby('chap_num')\
        .apply(lambda x: ''.join(x.line_str))\
        .to_frame()\
        .rename(columns={0:'chap_str'})
    del(df)

    # chapters to paragraphs
    paras = chaps.chap_str.str.split(para_pat, expand=True)\
        .stack()\
        .to_frame()\
        .rename(columns={0:'para_str'})
    paras.index.names = PARAS
    paras.para_str = paras.para_str.str.strip()
    paras.para_str = paras.para_str.str.replace(r'\n', ' ')
    paras.para_str = paras.para_str.str.replace(r'\s+', ' ')
    paras = paras[~paras.para_str.str.match(r'^\s*$')]
    del(chaps)

    # paragraphs to sentences
    sents = paras.para_str\
        .apply(lambda x: pd.Series(nltk.sent_tokenize(x)))\
        .stack()\
        .to_frame()\
        .rename(columns={0:'sent_str'})
    sents.index.names = SENTS
    del(paras)

    # sentences to tokens
    tokens = sents.sent_str\
        .apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x))))\
        .stack()\
        .to_frame()\
        .rename(columns={0:'pos_tuple'})
    tokens.index.names = OHCO
    tokens['pos'] = tokens.pos_tuple.apply(lambda x: x[1])
    tokens['token_str'] = tokens.pos_tuple.apply(lambda x: x[0])
    tokens = tokens.drop('pos_tuple', 1)
    del(sents)

    tokens['punc'] = tokens.token_str.str.match(r'^[\W_]*$').astype('int')
    tokens['num'] = tokens.token_str.str.match(r'^.*\d.*$').astype('int')

    WORDS = (tokens.punc == 0) & (tokens.num == 0)
    tokens.loc[WORDS, 'term_str'] = tokens.token_str.str.lower()\
        .str.replace(r'["_*.]', '')
    vocab = tokens[tokens.punc == 0].term_str.value_counts().to_frame()\
        .reset_index()\
        .rename(columns={'index':'term_str', 'term_str':'n'})
    vocab = vocab.sort_values('term_str').reset_index(drop=True)
    vocab.index.name = 'term_id'

    vocab['p'] = vocab.n / vocab.n.sum()
    stemmer = nltk.stem.porter.PorterStemmer()
    vocab['port_stem'] = vocab.term_str.apply(lambda x: stemmer.stem(x))

    # define stop words
    stopwords = set(nltk.corpus.stopwords.words('english') + extra_stopwords)
    sw = pd.DataFrame({'x':1}, index=stopwords)
    vocab['stop'] = vocab.term_str.map(sw.x).fillna(0).astype('int')
    del(sw)

    tokens['term_id'] = tokens['term_str'].map(vocab.reset_index()\
        .set_index('term_str').term_id).fillna(-1).astype('int')

    with sqlite3.connect(db_file) as db:
        tokens.to_sql('token', db, if_exists='replace', index=True)
        vocab.to_sql('vocab', db, if_exists='replace', index=True)

def vector_space_make(db):

    db_name = db
    
    OHCO = ['book','chapter', 'para_num', 'sent_num', 'token_num']

    import sqlite3
    import pandas as pd
    import numpy as np

    with sqlite3.connect(db_name) as db:
        K = pd.read_sql('SELECT * FROM token', db, index_col=OHCO)
        V = pd.read_sql('SELECT * FROM vocab', db, index_col='term_id')

    WORDS = (K.punc == 0) & (K.num == 0) & K.term_id.isin(V[V.stop==0].index)
    BOW = K[WORDS].groupby(OHCO[:1]+['term_id'])['term_id'].count()
    DTM = BOW.unstack().fillna(0)

    # compute TF
    alpha = .000001
    alpha_sum = alpha * V.shape[0]
    TF = DTM.apply(lambda x: (x + alpha) / (x.sum() + alpha_sum), axis=1)

    # compute TFIDF
    N_docs = DTM.shape[0]
    V['df'] = DTM[DTM > 0].count()
    TFIDF = TF * np.log2(N_docs / V[V.stop==0]['df'])

    print("TFIDF:")
    print(TFIDF.head())

    # compute TFTH
    THM = -(TF * np.log2(TF))
    TFTH = TF.apply(lambda x: x * THM.sum(), 1)

    # add stats to V
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

    D = DTM.sum(1).astype('int').to_frame().rename(columns={0:'term_count'})
    D['tf'] = D.term_count / D.term_count.sum()
    
    print("D:")
    print(D.head())

    # get all goc pairs
    chap_ids = D.index.tolist()
    pairs = [(i,j) for i in chap_ids for j in chap_ids if j > i]
    P = pd.DataFrame(pairs).reset_index(drop=True).set_index([0,1])
    P.index.names = ['doc_x','doc_y']

    def euclidean(row):
        D1 = TFIDF.loc[row.name[0]]
        D2 = TFIDF.loc[row.name[1]]
        x = (D1 - D2)**2
        y = x.sum() 
        z = np.sqrt(y)
        return z

    P['euclidean'] = 0
    P['euclidean'] = P.apply(euclidean, 1)

    def cosine(row):
        D1 = TFIDF.loc[row.name[0]]
        D2 = TFIDF.loc[row.name[1]]
        x = D1 * D2
        y = x.sum()
        a = np.sqrt((D1**2).sum())
        b = np.sqrt((D2**2).sum())
        c = a * b
        z = y / c
        return z

    P['cosine'] = P.apply(cosine, 1)
    
    print("P:")
    print(P.head())

    with sqlite3.connect(db_name) as db:
        V.to_sql('vocab', db, if_exists='replace', index=True)
        K.to_sql('token', db, if_exists='replace', index=True)
        D.to_sql('doc', db, if_exists='replace', index=True)
        P.to_sql('docpair', db, if_exists='replace', index=True)
#     BOW.to_frame().rename(columns={'term_id':'n'}).to_sql('bow', db, if_exists='replace', index=True)
        TFIDF.stack().to_frame().rename(columns={0:'term_weight'})\
            .to_sql('dtm_tfidf', db, if_exists='replace', index=True)

    #return V, K, D, P, TFIDF




