import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import tensorflow.keras.preprocessing as preprocessing


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


def preprocess(x_train, x_test, num_words):
    """
    Takes an array of text strings and outputs an array where each text is a vector of integers assigned based on word
    frequency
    """

    # Apply specialised preprocessing to each text item
    x_train = np.array([preprocess_text(x) for x in x_train])
    x_test = np.asarray([preprocess_text(x) for x in x_test])

    # Use keras to tokenize words
    tokenizer = preprocessing.text.Tokenizer(
        num_words=num_words
    )

    # Tokenize text data
    tokenizer.fit_on_texts(x_train)

    # Encode text data into sequences
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    return (x_train, x_test), vocab_size


def load_sentiment_140(num_words=None, num_rows=None, test_split=0.2, seed=100):
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

    # Load dataset from file
    sentiment_data = pd.read_csv("data/sentiment-140/training.1600000.processed.noemoticon.csv",
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
                                                        sentiment_data["Sentiment"].to_numpy(), test_size=test_split,
                                                        random_state=seed)

    # Convert labels of 4 to 1
    y_train[y_train == 4] = 1
    y_test[y_test == 4] = 1

    # Apply text preprocessing to training text
    (x_train, x_test), vocab_size = preprocess(x_train, x_test, num_words)

    return (x_train, y_train), (x_test, y_test), vocab_size


def load_covid_twitter():
    pass
