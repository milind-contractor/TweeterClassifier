# -*- coding: utf-8 -*-
#%%
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from numpy as np

st = StanfordNERTagger('/Users/mcontrac/Documents/pythonForML/TweeterClassifier/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '/Users/mcontrac/Documents/pythonForML/TweeterClassifier/stanford-ner/stanford-ner.jar',
					   encoding='utf-8')

text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)


# %%
print(classified_text)
x = np.array(classified_text)
ner = ["PERSON", "ORGANIZATION", "LOCATION"]
#x[:,1]=="PERSON"
x[x[:,1]!='O']

# for a,b in classified_text:
#     if b=="PERSON":
#         print(a)
#     if b== "LOCATION":
#         print(a)

# %%
x[:,1].isin(ner)

# %%
