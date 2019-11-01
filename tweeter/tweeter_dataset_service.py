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
from tweeter.dataset import dataframe
from tweeter.dataset import tweets

def strip_url(text):
    return re.sub('http[s]?://\S+', '', text)

def normalize_tweets(tweets):
    for index, entry in enumerate(tweets['text']):
        tweets.loc[index,'input_tweet'] = entry
    
    # Step - a : Remove blank rows if any.
    tweets['text'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    tweets['text'] = [entry.lower() for entry in tweets['text']]
    tweets['text'] = [strip_url(entry) for entry in tweets['text']]
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

def split_test_train_set (numberOfClass, testsize):
    return StratifiedShuffleSplit(n_splits=numberOfClass, test_size=testsize, random_state=0)


class tweeter_dataset_service(object):

    def __init__(self, csvfilepath,numberofClass, testsize):
        self.numberofClass = numberofClass
        self.testsize = testsize
        #read file
        self.all_tweets = pd.read_csv(csvfilepath,encoding='latin-1',sep=";")
        # Data Lemmatizer and clean all tweets for Model
        self.all_tweets = normalize_tweets(self.all_tweets)
        splitRatio = split_test_train_set(numberofClass, testsize)
        
        from tweeter.dataset import dataframe
        self.feature_vector = dataframe(dataset)
        
        splitRatio.get_n_splits(self.x, self.y)

        for train_idx, test_idx in splitRatio.split(self.feature_vector.X, self.feature_vector.Y):
            self.train_index, self.test_index = train_idx, test_idx

    def create_sample_set(self):
        from tweeter.dataset import dataset
        self.trainingset = dataset(self.feature_vector,self.train_index)
        self.testset = dataset(self.feature_vector, self.test_index)

        Encoder = LabelEncoder()
        self.trainingset.encode(Encoder)
        self.testset.encode(Encoder) 
        
    def generate_tfidf(self, maxfeature):
        Tfidf_vect = TfidfVectorizer(max_features=maxfeature)
        Tfidf_vect.fit(self.feature_vector.X)

        self.trainingset.transform(Tfidf_vect)
        self.testset.transform(Tfidf_vect)

    def run_svm(self):
        SVM = svm.SVC(C=1.0, kernel='linear', degree=4, gamma='auto')
        SVM.fit(self.trainingset.x_transform,self.trainingset.Y_encode)
        # predict the labels on validation dataset
        predictions_SVM = SVM.predict(self.testset.x_transform)
        # Use accuracy_score function to get the accuracy
        print("SVM Accuracy Score -> ",
                accuracy_score(predictions_SVM, self.testset.Y_encode)*100)