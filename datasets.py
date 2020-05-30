import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import tensorflow.keras.preprocessing as preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('wordnet')
from tensorflow.keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import random


def preprocess_text(text):
    wordLemm = WordNetLemmatizer()
    snowStem = SnowballStemmer("english")
    stopwordlist = set(stopwords.words('english'))
    # TODO: Spell check
    # Remove URLs
    # Defining dictionary containing all emojis with their meanings.

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', '>-)':
        'evilgrin', ':(': 'sad', ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry',
              ':-O': 'surprised', ':-*': 'kissing', ':-@': 'shocked', ':-$': 'confused',
              ':-\\': 'annoyed', ':-#': 'mute', '(((H)))': 'hugs', ':-X': 'kissing',
              '`:-)': 'smile', ':^)': 'smile', ':-&': 'confused', '<:-)': 'smile',
              ':->': 'smile', '(-}{-)': 'kissing', ':-Q': 'smoking', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':*)': 'smile',
              ':@': 'shocked', ':-0': 'yell', ':-----)': 'liar', '%-(': 'confused',
              '(:I': 'egghead', '|-O': 'yawning', ':@)': 'smile', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', '~:0': 'baby', '-@--@-': 'eyeglass',
              ":'-)": 'sadsmile', '{:-)': 'smile', ';)': 'wink', ';-)': 'wink',
              'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}
    text = text.strip()

    for emoji in emojis.keys():
        text = text.replace(emoji, "EMOJI" + emojis[emoji])

    text = re.sub(
        r'(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+['
        r'a-zA-Z0-9]\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
        ' ', text)

    # Remove punctuation and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Remove all @usernames
    text = re.sub(r'@[^\s]+', ' ', text)

    # Convert "#topic" to just "topic"
    text = re.sub(r'#([^\s]+)', r'\1', text)

    # Defining set containing all stopwords in english.
    # stopwordlist = set(stopwords.words('english'))

    text = re.sub(r"(.)\1\1+", r"\1\1", text)
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    for word in text.split():
        # Checking if the word is a stopword.
        if word not in stopwordlist:
            if len(word) > 1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                # Stemming the word.
                # word = snowStem.stem(word)
                text += (word + ' ')

    return text


def wordCloudSentiment(x, y, pos):
    data = pd.DataFrame()
    data['Sentiment'] = y
    data['Text'] = x
    if pos:
        data_array = data[data['Sentiment'] == 1]
    else:
        data_array = data[data['Sentiment'] == 0]
    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(np.asarray(data_array["Text"])))
    if pos:
        plt.title("Positive")
    else:
        plt.title("Negative")
    plt.imshow(wc)
    plt.show()


def create_vectorizer(corpus, max_features=20000, simple_classifier=False):
    # Use keras to tokenize words
    if simple_classifier:
        # Create Tfidf Vectorizer
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

        # Fit to corpus
        vectorizer.fit(corpus)
    else:
        # Create Keras Tokenizer
        vectorizer = preprocessing.text.Tokenizer(num_words=max_features)

        # Tokenize text data
        vectorizer.fit_on_texts(corpus)

    return vectorizer


def preprocess(vectorizer, text_arr, maxlen, simple_classifier=False):
    """
    Takes an array of text strings and outputs an array where each text is a vector of integers assigned based on word
    frequency
    """
    # Apply specialised preprocessing to each text item
    text_arr = np.array([preprocess_text(x) for x in text_arr])

    # Use keras to tokenize words
    if simple_classifier:
        # Vectorize text to sequences
        text_arr = vectorizer.transform(text_arr)
    else:
        # Vectorize text to sequences
        text_arr = vectorizer.texts_to_sequences(text_arr)

        # Pad sequences to equal length
        text_arr = sequence.pad_sequences(text_arr, maxlen)

    return text_arr


def load_sentiment_140(vectorizer,
                       data_dir="data",
                       num_rows=None,
                       maxlen=None,
                       test_split=0.2,
                       seed=100,
                       simple_classifier=False):
    """Loads the Sentiment 140 dataset, with preprocessing

    # Arguments
        num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        num_rows: number of rows of data to return
        test_split: percentage in float form of data to set aside for testing
        seed: random seed for sample shuffling.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    if not maxlen:
        maxlen = 100

    # Load dataset from file
    file_path = data_dir + "/sentiment-140/training.1600000.processed.noemoticon.csv"
    sentiment_data = pd.read_csv(file_path,
                                 encoding='ISO-8859-1',
                                 names=["Sentiment", "ID", "Date", "Query", "User", "Text"])

    # Shuffle order of rows
    sentiment_data = sentiment_data.sample(frac=1, random_state=seed)

    # Only grab num_rows rows from data
    # How many rows of data to return. Default all
    if not num_rows:
        num_rows = len(sentiment_data["Sentiment"])
    sentiment_data = sentiment_data.iloc[:num_rows]

    # Split training data
    x_train, x_test, y_train, y_test = train_test_split(sentiment_data["Text"].to_numpy(),
                                                        sentiment_data["Sentiment"].to_numpy(),
                                                        test_size=test_split,
                                                        random_state=seed)
    # Convert labels of 4 to 1
    y_train[y_train == 4] = 1
    y_test[y_test == 4] = 1

    # Create word clouds
    wordCloudSentiment(x_train, y_train, True)
    wordCloudSentiment(x_train, y_train, False)

    # Apply text preprocessing to training and testing text
    x_train = preprocess(vectorizer, x_train, maxlen,
                         simple_classifier=simple_classifier)

    x_test = preprocess(vectorizer, x_test, maxlen,
                        simple_classifier=simple_classifier)

    return (x_train, y_train), (x_test, y_test)


def load_covid(data_dir="data", num_rows=None, seed=100):
    """

    Args:
        data_dir:
        num_rows:
        seed: 

    Returns:
    """
    # Load dataset from file
    file_dir = data_dir + "/covid19-tweets/2020-*.csv"
    files_to_load = glob.glob(file_dir)

    rows_from_each = round(num_rows / len(files_to_load))

    print("Loading {} rows from {} files".format(rows_from_each, len(files_to_load)))

    # Load first csv as initial dataframe
    n = sum(1 for line in open(files_to_load[1], encoding="utf8")) - 1  # number of records in file (excludes header)
    skip = sorted(
        random.sample(range(1, n + 1),
                      n - rows_from_each))  # the 0-indexed header will not be included in the skip list
    covid_data = pd.read_csv(files_to_load[1], skiprows=skip)
    covid_data = covid_data[covid_data.lang == 'en']

    for file in files_to_load[2:]:
        print("Sampling {}".format(file))

        n = sum(1 for line in open(file, encoding="utf8")) - 1  # number of records in file (excludes header)
        skip = sorted(
            random.sample(range(1, n + 1),
                          n - rows_from_each))  # the 0-indexed header will not be included in the skip list

        df = pd.read_csv(file, skiprows=skip)
        df = df[df.lang == 'en']  # Drop rows that aren't english

        covid_data = pd.concat([covid_data, df])

    # Get rid of time in datetime column
    covid_data['created_at'] = covid_data['created_at'].apply(lambda x: x.split("T")[0])

    return covid_data
