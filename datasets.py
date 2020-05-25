import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
import tensorflow.keras.preprocessing as preprocessing
from tensorflow.keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text):
    # TODO: Spell check
    # Remove URLs
    text = re.sub(
        r'(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+['
        r'a-zA-Z0-9]\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
        ' ', text)

    # Remove punctuation and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Remove single characters
    # text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Remove all @usernames
    text = re.sub(r'@[^\s]+', ' ', text)

    # Convert "#topic" to just "topic"
    text = re.sub(r'#([^\s]+)', r'\1', text)

    text = text.strip()

    return text


def preprocess(text_array, num_words, vectorizer="tfid", maxlen=100):
    """
    Takes an array of text strings and outputs an array where each text is a vector of integers assigned based on word
    frequency
    """
    # Apply specialised preprocessing to each text item
    text_array = np.array([preprocess_text(x) for x in text_array])

    # Use keras to tokenize words
    if vectorizer == "tfid":
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

        text_array = vectorizer.fit_transform(text_array)
    elif vectorizer == "keras":
        tokenizer = preprocessing.text.Tokenizer(
            num_words=num_words
        )

        # Tokenize text data
        tokenizer.fit_on_texts(text_array)

        text_array = tokenizer.texts_to_sequences(text_array)

        text_array = sequence.pad_sequences(text_array, maxlen=maxlen)
    else:
        raise ValueError("Incorrect argument for vectorizer, must be tfid or keras")

    return text_array


def load_sentiment_140(data_dir="data", num_words=None, num_rows=None, maxlen=None, test_split=0.2, seed=100):
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
    file_dir = data_dir + "/sentiment-140/training.1600000.processed.noemoticon.csv"
    sentiment_data = pd.read_csv(file_dir,
                                 encoding='ISO-8859-1',
                                 names=["Sentiment", "ID", "Date", "Query", "User", "Text"])

    # Shuffle order of rows
    sentiment_data = sentiment_data.sample(frac=1, random_state=seed)

    # Only grab num_rows rows from data
    # How many rows of data to return. Default all
    if not num_rows:
        num_rows = len(sentiment_data["Sentiment"])
    sentiment_data = sentiment_data.iloc[:num_rows]

    # Apply text preprocessing to training text
    vectorised_sentiment_text = preprocess(sentiment_data["Text"].to_numpy(),
                                           num_words,
                                           vectorizer="keras",
                                           maxlen=maxlen)
    sentiment_values = sentiment_data["Sentiment"].to_numpy()

    # Convert 4 to 1
    sentiment_values[sentiment_values == 4] = 1

    x_train, x_test, y_train, y_test = train_test_split(vectorised_sentiment_text,
                                                        sentiment_values,
                                                        test_size=test_split,
                                                        random_state=seed)

    return (x_train, y_train), (x_test, y_test)


def load_covid_twitter():
    pass
