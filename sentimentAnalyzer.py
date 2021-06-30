from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random

import nltk
import json

nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def updatePostiveTweet(data):
    f = open(r'C:\nltk_data\corpora\twitter_samples\updated_positive_tweets.json', "a")
    json.dump(getJsonData(data), f)
    f.write("\n")
    f.close()

def updateNegativeTweet(data):
    f = open(r"C:\nltk_data\corpora\twitter_samples\updated_negative_tweets.json", "a")
    json.dump(getJsonData(data), f)
    f.write("\n")
    f.close()

def clearData():
    f = open(r"C:\nltk_data\corpora\twitter_samples\updated_negative_tweets.json", "w")
    f.write("")
    f.close()
    f = open(r"C:\nltk_data\corpora\twitter_samples\updated_positive_tweets.json", "w")
    f.write("")
    f.close()

def getJsonData(text):
    data = '{"text": "testData"}'
    # parse x:
    jsonData = json.loads(data)
    jsonData["text"] = text
    return jsonData

def initializeFiles():
    positiveText = "This is a positive test"
    negativeText = "This is a negative test"

    negativeData = getJsonData(negativeText)
    positiveData = getJsonData(positiveText)

    f = open(r"C:\nltk_data\corpora\twitter_samples\updated_negative_tweets.json", "w")
    json.dump(negativeData, f)
    f.write("\n")
    f.close()
    f = open(r"C:\nltk_data\corpora\twitter_samples\updated_positive_tweets.json", "w")
    json.dump(positiveData, f)
    f.write("\n")
    f.close()

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

@run_once
def getPositiveDataset(stop_words):
    print("Getting Positive Dataset")
    global positive_cleaned_tokens_list
    positive_cleaned_tokens_list = []
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

@run_once
def getNegativeDataset(stop_words):
    print("Getting Negative Dataset")
    global negative_cleaned_tokens_list
    negative_cleaned_tokens_list = []
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

def train():
    stop_words = stopwords.words('english')
    getNegativeDataset(stop_words)
    getPositiveDataset(stop_words)

    updated_negative_tweet_tokens = twitter_samples.tokenized('updated_negative_tweets.json')
    updated_positive_tweet_tokens = twitter_samples.tokenized('updated_positive_tweets.json')

    for tokens in updated_negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in updated_positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset
    test_data = dataset

    global classifier

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

def getSentiment(custom_tweet):

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    result = classifier.classify(dict([token, True] for token in custom_tokens))
    print(custom_tweet, result)

    return result