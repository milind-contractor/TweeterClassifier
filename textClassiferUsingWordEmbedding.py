#%%
%load_ext autoreload
%autoreload 2

#%%
import pandas as pd
import numpy as np
from numpy import random
import gensim
import spacy
import nltk
import os
import pickle

import matplotlib.pyplot as plt
import re
import warnings
warnings.simplefilter('ignore')
pd.set_option('max_colwidth',1000)


%matplotlib inline

#%%
dir_path = '.'

df = pd.read_csv(os.path.join(dir_path, 'consumer_complaints.csv'))
df = df[pd.notnull(df['consumer_complaint_narrative'])]


# %%
# Get familiar with dataset.
df.shape

# %%
df.info()

# %%
df.to_csv(os.path.join(dir_path, 'consumer_complaints.csv'), index=False)

# %%
# Filter on text and label.
df = df[['product','consumer_complaint_narrative']]

#%%
# Experiment on smaller subset.
df = df[:10000]


# %%
#Distribution of target variable.
print(df['product'].value_counts())

# %%
from UtilTextClassification import plot_freq

plot_freq(df, col=['product'], top_classes=30)

# %%
import spacy

nlp = spacy.load('en_core_web_md')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# %%
from UtilWordEmbedding import DocPreprocess
all_docs = DocPreprocess(nlp, stop_words, df['consumer_complaint_narrative'], df['product'])

# %%
all_docs.tagdocs[1]
all_docs.labels.iloc[1]
all_docs.doc_words[4][:52]
print(all_docs.doc_words[6])
# %%
import multiprocessing
import sys
from gensim.models.word2vec import Word2Vec

workers = multiprocessing.cpu_count()
print('number of cpu: {}'.format(workers))
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise."

# %%
word_model = Word2Vec(all_docs.doc_words,
                      min_count=2,
                      size=300,
                      window=5,
                      workers=workers,
                      iter=100)

# %%
from UtilWordEmbedding import MeanEmbeddingVectorizer


mean_vec_tr = MeanEmbeddingVectorizer(word_model)
doc_vec = mean_vec_tr.transform(all_docs.doc_words)

# %%

word_model.most_similar('submit')

# %%
# Save word averaging doc2vec.
print('Shape of word-mean doc2vec...')
print(doc_vec.shape)
print('Save word-mean doc2vec as csv file...')
#np.savetxt(os.path.join(dir_path,'doc_vec.csv'), doc_vec, delimiter=',')

# %%
