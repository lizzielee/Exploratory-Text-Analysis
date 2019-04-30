import glob
import sqlite3

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('tagsets')
nltk.download('wordnet')

def import_corpus(src_dir, db_file):
    
    source_dir = src_dir
    chap_pat = r'^\s*Chapter.*$'
    para_pat = r'\n\n+'
    token_pat = r'([\W_]+)'
    db_file = db_file

    extra_stopwords = """
    us rest went least would much must long one like much say well without though yet might still upon
    done every rather particular made many previous always never thy thou go first oh thee ere ye came
    almost could may sometimes seem called among another also however nevertheless even way one two three
    ever put
    """.strip().split()

    OHCO = ['book', 'chapter', 'para_num', 'sent_num', 'token_num']
    BOOKS = OHCO[:1]
    CHAPS = OHCO[:2]
    PARAS = OHCO[:3]
    SENTS = OHCO[:4]

    files = glob.glob("{}/*.txt".format(source_dir))
    codes = [f.replace('.txt','').split('/')[-1].split('_') for f in files]
    T = pd.DataFrame(codes, columns = ['book','chapter'])
    T = T[CHAPS]
    T.chapter = T.chapter.astype('int')

    T['text'] = [open(f, 'r', encoding='utf-8').read() for f in files]

    try:
        T = T.set_index(CHAPS)
        T = T.sort_index()
    except KeyError:
        pass


    sw = nltk.corpus.stopwords.words('english') + extra_stopwords

    T.text = T.text.str.replace(r"(—|-)", ' \g<1> ')


    # chapters to paragraphs
    paras = T.text.str.split(para_pat, expand=True)\
        .stack()\
        .to_frame()\
        .rename(columns={0:'para_str'})
    paras.index.names = PARAS
    paras.para_str = paras.para_str.str.strip()
    paras.para_str = paras.para_str.str.replace(r'\n', ' ')
    paras.para_str = paras.para_str.str.replace(r'\s+', ' ')
    paras = paras[~paras.para_str.str.match(r'^\s*$')]

    # paras to sentences
    sents = paras.para_str\
        .apply(lambda x: pd.Series(nltk.sent_tokenize(x)))\
        .stack()\
        .to_frame()\
        .rename(columns={0:'sent_str'})
    sents.index.names = SENTS
    del(paras)

    # sentences to tokens with POS tagging
    tokenizer = RegexpTokenizer('\s+', gaps=True)
    tokens = sents.sent_str\
        .apply(lambda x: pd.Series(nltk.pos_tag(tokenizer.tokenize(x))))\
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
        .str.replace(token_pat, '')
#     .str.replace(r'["_*.\']', '')
    vocab = tokens[tokens.punc == 0].term_str.value_counts().to_frame()\
        .reset_index()\
        .rename(columns={'index':'term_str', 'term_str':'n'})
    vocab = vocab.sort_values('term_str').reset_index(drop=True)
    vocab.index.name = 'term_id'

    vocab['p'] = vocab.n / vocab.n.sum()

    # add stems
    stemmer = nltk.stem.porter.PorterStemmer()
    vocab['port_stem'] = vocab.term_str.apply(lambda x: stemmer.stem(x))

    # define stopwords
    stopwords = set(nltk.corpus.stopwords.words('english') + extra_stopwords)
    sw = pd.DataFrame({'x':1}, index=stopwords)
    vocab['stop'] = vocab.term_str.map(sw.x).fillna(0).astype('int')
    del(sw)

    tokens['term_id'] = tokens['term_str'].map(vocab.reset_index()\
    .set_index('term_str').term_id).fillna(-1).astype('int')

    with sqlite3.connect(db_file) as db:
        T.to_sql('doc', db, if_exists='replace', index=True)
        tokens.to_sql('token', db, if_exists='replace', index=True)
        vocab.to_sql('vocab', db, if_exists='replace', index=True)

    