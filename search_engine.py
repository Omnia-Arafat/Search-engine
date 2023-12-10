import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
import pandas as pd
import numpy as np
import math
import warnings
import nltk

warnings.filterwarnings("ignore")


# read files

def read_files(file):
    if 'txt' in file:
        with open(f'Docs/{file}', 'r') as f:
            return f.read()

documents = []
for file in os.listdir('Docs'):
    documents.append(read_files(file))


token_docs = []
for document in documents:
    token_docs.append(word_tokenize(document))

stop_words = stopwords.words('english')
stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')

# preprocessing and tokinzation
documents = []
for token in token_docs:
    each_token = []
    for term in token:
        if term not in stop_words:
            each_token.append(term)
    documents.append(each_token)

def preprocessing(doc):
    token_docs = word_tokenize(doc)
    stop_words = stopwords.words('english')
    stop_words.remove('in')
    stop_words.remove('to')
    stop_words.remove('where')

    prepared_doc = []
    for term in token_docs:
        if term not in stop_words:
            prepared_doc.append(term)
    return prepared_doc

fileno = 1
pos_index = {}

file_names = natsorted(os.listdir("Docs"))
print(file_names)

for file_name in file_names:
    with open(f'Docs/{file_name}', 'r') as f:
        doc = f.read()

    final_token_list = preprocessing(doc)

    for pos, term in enumerate(final_token_list):
        if term in pos_index:
            pos_index[term][0] += 1
            if fileno in pos_index[term][1]:
                pos_index[term][1][fileno].append(pos)
            else:
                pos_index[term][1][fileno] = [pos]
        else:
            pos_index[term] = [1, {fileno: [pos]}]

    fileno += 1

print("Positional Index:")
print(pos_index)

def put_query(q, display=1):
    lis = [[] for i in range(10)]
    q = preprocessing(q)
    for term in q:
        if term in pos_index.keys():
            for key in pos_index[term][1].keys():
                if lis[key-1] != []:
                    if lis[key-1][-1] == pos_index[term][1][key][0] - 1:
                        lis[key-1].append(pos_index[term][1][key][0])
                else:
                    lis[key-1].append(pos_index[term][1][key][0])

    positions = []
    if display == 1:
        for pos, lst in enumerate(lis, start=1):
            if len(lst) == len(q):
                positions.append('document ' + str(pos))
        return positions
    else:
        for pos, lst in enumerate(lis, start=1):
            if len(lst) == len(q):
                positions.append('doc' + str(pos))
        return positions

q = ' antony brutus'
print(f"Query: '{q}'")
print("Matching Documents:", put_query(q))

documents = []
files = os.listdir('Docs')
for file in range(1, 11):
    documents.append(" ".join(preprocessing(read_files(str(file) + '.txt'))))

all_terms = []
for doc in documents:
    for term in doc.split():
        all_terms.append(term)

all_terms = set(all_terms)

def get_tf(document):
    word_dict = dict.fromkeys(all_terms, 0)
    for word in document.split():
        word_dict[word] += 1
    return word_dict

tf = pd.DataFrame(get_tf(documents[0]).values(), index=get_tf(documents[0]).keys())

for i in range(1, len(documents)):
    tf[i] = get_tf(documents[i]).values()

tf.columns = ['doc' + str(i) for i in range(1, 11)]

print("\nTerm Frequency (TF):")
print(tf)

def weighted_tf(x):
    if x > 0:
        return math.log10(x) + 1
    return 0

w_tf = tf.copy()

for i in range(0, len(documents)):
    w_tf['doc' + str(i + 1)] = tf['doc' + str(i + 1)].apply(weighted_tf)

print("\nWeighted Term Frequency (wTF):")
print(w_tf)

tdf = pd.DataFrame(columns=['df', 'idf'])
for i in range(len(tf)):
    in_term = w_tf.iloc[i].values.sum()
    tdf.loc[i, 'df'] = in_term
    tdf.loc[i, 'idf'] = math.log10(10 / (float(in_term)))

tdf.index = w_tf.index

print("\nTerm Document Frequency (DF) and Inverse Document Frequency (IDF):")
print(tdf)

tf_idf = w_tf.multiply(tdf['idf'], axis=0)

print("\nTerm Frequency-Inverse Document Frequency (TF-IDF):")
print(tf_idf)

def get_doc_len(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x**2).sum())

doc_len = pd.DataFrame()
for col in tf_idf.columns:
    doc_len.loc[0, col + '_length'] = get_doc_len(col)

print("\nDocument Lengths:")
print(doc_len)

def get_norm_tf_idf(col, x):
    try:
        return x / doc_len[col + '_length'].values[0]
    except:
        return 0

norm_tf_idf = pd.DataFrame()
for col in tf_idf.columns:
    norm_tf_idf[col] = tf_idf[col].apply(lambda x: get_norm_tf_idf(col, x))

print("\nNormalized Term Frequency-Inverse Document Frequency (TF-IDF):")
print(norm_tf_idf)

def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except:
        return 0

def insert_query(q):
    docs_found = put_query(q, 2)
    if docs_found == []:
        print("No Matching Documents Found.")
        return

    new_q = preprocessing(q)
    query = pd.DataFrame(index=norm_tf_idf.index)
    query['tf'] = [1 if x in new_q else 0 for x in list(norm_tf_idf.index)]
    query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
    product = norm_tf_idf.multiply(query['w_tf'], axis=0)
    query['idf'] = tdf['idf'] * query['w_tf']
    query['tf_idf'] = query['w_tf'] * query['idf']
    query['normalized'] = 0
    for i in range(len(query)):
        query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))

    print("\nQuery Details:")
    print(query.loc[new_q])

    product2 = product.multiply(query['normalized'], axis=0)
    scores = {}
    for col in put_query(q, 2):
        scores[col] = product2[col].sum()

    product_result = product2[list(scores.keys())].loc[new_q]

    print("\nProduct (query*matched doc):")
    print(product_result)

    print("\nProduct Sum:")
    print(product_result.sum())

    print("\nQuery Length:")
    q_len = math.sqrt(sum([x**2 for x in query['idf'].loc[new_q]]))
    print(q_len)

    print("\nCosine Similarity:")
    print(product_result.sum())

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\nReturned Documents:")
    for tuple in sorted_scores:
        print(tuple[0], end=" ")

insert_query('antony the brutus')
