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
          ':->': 'smile', '(-}{-)': 'kissing', ':-Q': 'smoking','$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':*)': 'smile',
          ':@': 'shocked',':-0': 'yell', ':-----)': 'liar', '%-(': 'confused',
          '(:I': 'egghead', '|-O': 'yawning', ':@)': 'smile', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', '~:0': 'baby', '-@--@-': 'eyeglass',
          ":'-)": 'sadsmile', '{:-)': 'smile', ';)': 'wink', ';-)': 'wink', 
          'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
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
    ## Defining set containing all stopwords in english.
    # stopwordlist = set(stopwords.words('english'))
    text = re.sub(r"(.)\1\1+", r"\1\1", text)
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    for word in text.split():            
            # Checking if the word is a stopword.
            if word not in stopwordlist:                
                if len(word)>1:    
                    # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    # Stemming the word.
                    #word = snowStem.stem(word)
                    text += (word+' ')
        

    return text

def wordCloudSentiment(x, y, pos):
    data = pd.DataFrame()
    data['Sentiment'] = y
    data['Text'] = x
    if pos:
        data_array = data[data['Sentiment'] == 0]
    else:
        data_array = data[data['Sentiment'] == 1]
    plt.figure(figsize = (20,20))
    wc = WordCloud(max_words = 1000 , width = 1600 , height = 800).generate(" ".join(np.asarray(data_array["Text"])))
    if pos:
        plt.title("Positive")
    else:
        plt.title("Negative")
    plt.imshow(wc)
    plt.show()

def preprocess(x_train, x_test,  y_train, num_words, simple_classifer):
    """
    Takes an array of text strings and outputs an array where each text is a vector of integers assigned based on word
    frequency
    """
    # Apply specialised preprocessing to each text item
    x_train = np.array([preprocess_text(x) for x in x_train])
    x_test = np.asarray([preprocess_text(x) for x in x_test])
    wordCloudSentiment(x_train,  y_train, False)
    wordCloudSentiment(x_train,  y_train, True)

    # Use keras to tokenize words
    if(simple_classifer):
        vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)
        vectorizer.fit(x_train)
        x_train = vectorizer.transform(x_train)
        x_test  = vectorizer.transform(x_test)
        vocab_size = None
    else:   
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
    # Split training data
    if not simple_classifer:
        x_train, x_test, y_train, y_test = train_test_split(sentiment_data["Text"].to_numpy(),
                                                            sentiment_data["Sentiment"].to_numpy(), test_size=test_split,
                                                            random_state=seed)
        # Convert labels of 4 to 1
        y_train[y_train == 4] = 1
        y_test[y_test == 4] = 1

        # Apply text preprocessing to training text
        (x_train, x_test), vocab_size = preprocess(x_train, x_test,  y_train, num_words, simple_classifer)
    else:
        
        x_train, x_test, y_train, y_test = train_test_split(sentiment_data["Text"].to_numpy(),
                                                            sentiment_data["Sentiment"].to_numpy(), test_size=test_split,
                                                            random_state=seed)
        y_train[y_train == 4] = 1
        y_test[y_test == 4] = 1        
        
        (x_train, x_test), vocab_size = preprocess(x_train, x_test, y_train, num_words, simple_classifer)


    return (x_train, y_train), (x_test, y_test)


def load_covid_twitter():
    pass
