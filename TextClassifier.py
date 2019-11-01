# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'TweeterClassifier'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ### Twitter Text Classifier
# #### Step 1.
# Import Libraries 
# 1. Pandas: Used for structured data operations and manipulations. 
# 2. Numpy:Stands for numerical python. It works on n-dimensional array.
# 3. nltk: nlp library for Lemmatization and Steming 
# 4. sklearn:
# 

#%%
import pandas as pd
import numpy as np
import re 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


np.random.seed(500)

def removeHttp(text):
    return re.sub('http[s]?://\S+', '', text)

def dataCleaning(tweets):
    for index, entry in enumerate(tweets['text']):
        tweets.loc[index,'input_tweet'] = entry
    
    # Step - a : Remove blank rows if any.
    tweets['text'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    tweets['text'] = [entry.lower() for entry in tweets['text']]
    tweets['text'] = [removeHttp(entry) for entry in tweets['text']]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    tweets['text']= [word_tokenize(entry) for entry in tweets['text']]

    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(tweets['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word)
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        tweets.loc[index,'text_final'] = str(Final_words)
    return tweets

def getSplitRatio (numberOfClass, testsize):
    return StratifiedShuffleSplit(n_splits=numberOfClass, test_size=testsize, random_state=0)

#%%
# readFile
#column = ["text_final","label","input_tweet"]
Corpus = pd.read_csv(r"Tweets_V3_5_5_balanced.csv",encoding='latin-1',sep=";")
# Data Lemmatizer and clean all tweets for Model
Corpus = dataCleaning(Corpus)

splitRatio = getSplitRatio(numberOfClass = 9, testsize = 0.3)
dataset = Corpus

from tweeter.dataset import dataframe
feature_vector = dataframe(dataset)

#input_X, input_Y, input_T = dataset['text_final'], dataset['label'], dataset['input_tweet']
#splitRatio.get_n_splits(input_X, input_Y)
splitRatio.get_n_splits(feature_vector.X, feature_vector.Y)
for train_idx, test_idx in splitRatio.split(feature_vector.X, feature_vector.Y):
    train_index, test_index = train_idx, test_idx

# %%
from tweeter.dataset import dataset
trainingset = dataset(feature_vector,train_index)
testset = dataset(feature_vector, test_index)


#%%
Encoder = LabelEncoder()
trainingset.encode(Encoder)
testset.encode(Encoder)

#%%
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(feature_vector.X)

trainingset.transform(Tfidf_vect)
testset.transform(Tfidf_vect)

#Train_X_Tfidf = Tfidf_vect.transform(trainingset.X)
#Test_X_Tfidf = Tfidf_vect.transform(testset.X)
#%%
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=4, gamma='auto')
SVM.fit(trainingset.x_transform,trainingset.Y_encode)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(testset.x_transform)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",
    accuracy_score(predictions_SVM, testset.Y_encode)*100)

# %%
