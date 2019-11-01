from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class dataframe():
    def __init__(self,
                data):
        self.X = data['text_final']
        self.Y = data['label']
        self.Tweets = data['input_tweet']

    def get_x(self,index):
        return self.X[index]

    def get_y(self,index):
        return self.Y[index]

    def get_tweets(self,index):
        return self.Tweets[index]
    
class dataset():
    def __init__(self, dataframe, index):
        self.X = dataframe.get_x(index)
        self.Y = dataframe.get_y(index)
        self.Tweets = dataframe.get_tweets(index)

    def encode(self, encoder):
        self.Y_encode = encoder.fit_transform(self.Y)

    def transform(self, transformer):
        self.x_transform = transformer.transform(self.X)
    