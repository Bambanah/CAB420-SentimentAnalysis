import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

import models
import datasets
import download_data


def run_lstm_model(x_train, y_train, x_test, y_test,
                   num_features,
                   metrics=None,
                   batch_size=None,
                   epochs=None,
                   maxlen=None,
                   embedding_size=None,
                   kernel_size=None,
                   filters=None,
                   pool_size=None,
                   lstm_output_size=None):
    # Training
    if batch_size is None:
        batch_size = 128
    if epochs is None:
        epochs = 20

    # Embedding
    if maxlen is None:
        maxlen = 100
    if embedding_size is None:
        embedding_size = 64

    # Convolution
    if kernel_size is None:
        kernel_size = 5
    if filters is None:
        filters = 64
    if pool_size is None:
        pool_size = 4

    # LSTM
    if lstm_output_size is None:
        lstm_output_size = 70

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    lstm_model = models.lstm(num_features,
                             maxlen=maxlen,
                             embedding_size=embedding_size,
                             kernel_size=kernel_size,
                             filters=filters,
                             pool_size=pool_size,
                             lstm_output_size=lstm_output_size,
                             metrics=metrics)

    print('Train...')
    lstm_model.fit(x_train,
                   y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_data=(x_test, y_test),
                   use_multiprocessing=True)
    lstm_loss, lstm_acc = lstm_model.evaluate(x_test, y_test, batch_size=batch_size)

    return lstm_loss, lstm_acc


if __name__ == "__main__":

    grab_data = False

    if grab_data:
        # Download all files in shared data folder
        download_data.download_from_drive(file_names=['sentiment140.zip'])

        # Unzip each zip saved in local data folder
        download_data.unzip_data()

        print("Data organised")

    # Run
    model_LSTM = True
    train_on_all = True
    train_140 = True
    train_imdb = False

    model_GRU = False

    num_rows = 100000  # Number of rows to load from data
    max_features = 20000  # Maximum number of features (words) to process
    maxlen = 100  # Maximum length of sequences

    # Load Sentiment 140 dataset
    (x_train_140, y_train_140), \
    (x_test_140, y_test_140) = datasets.load_sentiment_140(data_dir="data",
                                                           num_words=max_features,
                                                           num_rows=num_rows,
                                                           maxlen=maxlen,
                                                           test_split=0.2,
                                                           seed=69)

    # Load IMDB dataset
    (x_train_imdb, y_train_imdb), \
    (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data(num_words=max_features)

    # Run LSTM Modelling
    if model_LSTM:
        if train_140:
            # Train and evaluate model
            loss_140, acc_140 = run_lstm_model(x_train_140, y_train_140, x_test_140, y_test_140,
                                               num_features=max_features,
                                               metrics=["acc"])

            # Show results
            print('Test loss 140:', loss_140)
            print('Test accuracy 140:', acc_140)

        if train_imdb:
            # Train and evaluate model
            loss_imdb, acc_imdb = run_lstm_model(x_train_imdb, y_train_imdb, x_test_imdb, y_test_imdb,
                                                 num_features=max_features,
                                                 metrics=["acc"])

            # Show results
            print('Test loss IMDB:', loss_imdb)
            print('Test accuracy IMDB:', acc_imdb)

    # Run GRU modelling
    if model_GRU:
        pass
