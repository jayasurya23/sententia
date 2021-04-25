#run
from tensorflow import keras

import pickle
import time
import tweepy
import pandas as pd
from pandas import Panel
pd.options.mode.chained_assignment = None
import re
from copy import deepcopy
from random import shuffle
from string import punctuation

#Model
import gensim
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from gensim.models.word2vec import Word2Vec
#Data cleanup
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

tqdm.pandas(desc="progress-bar")

consumer_key = "qkgVE4vTTvFN4VrbgcjryaZYP"
consumer_secret = "zHYsa6j6qDSYKhIGzGOSGeZDCht2gVDnycV2pQf99OOVx97xf5"
access_token = "560112074-jiP9kWwSrUwhvC7GsvtbSOpmPuJAOz7NYRn2s5Q8"
access_token_secret = "lE4KP6lrE7y413caHj1F2cLl5QEq5WXiphjudGKu7hoGq"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)



LabeledSentence = gensim.models.doc2vec.LabeledSentence 

cbow_model = gensim.models.Word2Vec.load("word2vec.model")
loaded_model = pickle.load(open("pickle_model.pkl", 'rb'))
tfidf = pickle.load(open("tf.pkl", 'rb'))
reconstructed_model = keras.models.load_model("nmodel.h5")

#DATA CLEANUP
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
#required
def tweet_cleaner(text):
    # return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()


def text_query_to_csv(text_query,count):
    tweets = []

    try:
        tweets = tweepy.Cursor(api.search,q=text_query,lang="en",exclude='retweets').items(count)
        tweets_list = [[tweet.created_at, tweet.id,tweet.user.location, tweet.text,tweet.entities['hashtags']] for tweet in tweets]

        tweets_df = pd.DataFrame(tweets_list,columns=['Datetime', 'Tweet Id','username', 'Text','Hashtag'])
        return tweets_df

    except BaseException as e:
        print('failed on_status,',str(e))
        time.sleep(3)

def tokenize(tweet):
    tokens = tokenizer.tokenize(tweet)
    return tokens

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += cbow_model[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


def prep_test(text_query,count):
    tweets_dataframe=text_query_to_csv(text_query, count)
    
    print("Cleaning and parsing the tweets...\n")
    clean_tweet_texts = []

    for j in range(count):
        clean_tweet_texts.append(tweet_cleaner(tweets_dataframe['Text'][j]))
    print("\ncleaning complete\n")    
    
    df = pd.DataFrame(clean_tweet_texts,columns=['text'])
    df['tokens'] = df['text'].progress_map(tokenize)
    b=df['tokens']
    t_test = labelizeTweets(b, 'TEST')
    btest_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, t_test))])
    btest_vecs_w2v = scale(btest_vecs_w2v)
    return btest_vecs_w2v
    

def score(keyword):
    bt=prep_test(keyword,200)
    # y_pred = loaded_model.predict(bt)
    y_pred2=reconstructed_model.predict(bt)
    test=np.where(y_pred2 > 0.09, 1, 0)
# array([0, 0, 0, 1, 1, 1])
    unique_elements, counts_elements = np.unique(test, return_counts=True)
    print(counts_elements)
    per=(counts_elements[1]/len(test))*100
    print("Positive tweets: {} : {}%".format(counts_elements[1], per))
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))

    # print(y_pred2)
    # unique_elements, counts_elements = np.unique(y_pred, return_counts=True)
    # unique_elements2, counts_elements2 = np.unique(y_pred2, return_counts=True)
    # # print("Frequency of unique values of the said array:")
    # # print(np.asarray((unique_elements, counts_elements)))
    # per=(counts_elements[1]/len(y_pred))*100
    # per2=(counts_elements2[1]/len(y_pred2))*100
    # print("Positive tweets: {} : {}%".format(counts_elements[1], per))
    # print("Positive tweets2: {} : {}%".format(counts_elements2[1], per2))
    # print("Frequency of unique values of the said array:")
    # print(np.asarray((unique_elements2, counts_elements2)))
    

    return per


