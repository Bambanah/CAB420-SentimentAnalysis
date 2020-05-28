import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pathlib

import tensorflow as tf

import models
import datasets
import download_data
import pandas as pd
from models import ensemble_classifers
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt

def run_simple_models(x_train, y_train, x_test, y_test):
    maxlen = 100
    ensemble_classifers(x_train, y_train, x_test, y_test)






def build_lstm_model(num_features,
                     embedding_size=None,
                     kernel_size=None,
                     filters=None,
                     pool_size=None,
                     lstm_output_size=None):
    """
    Builds and compiles an LSTM model with the provided hyper-parameters
    Args:
        num_features:
        embedding_size:
        kernel_size:
        filters:
        pool_size:
        lstm_output_size:

    Returns:

    """
    # Embedding
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

    print('Build model...')

    lstm_model = models.lstm(num_features,
                             embedding_size=embedding_size,
                             kernel_size=kernel_size,
                             filters=filters,
                             pool_size=pool_size,
                             lstm_output_size=lstm_output_size)

    return lstm_model


def train_model(model, x_train, y_train, x_test, y_test,
                epochs=None, batch_size=None):
    """
    Trains model on provided data.
    Args:
        model: A compiled tensorflow model
        x_train: Training X data
        y_train: Training Y data
        x_test: Testing X data
        y_test: Testing Y data
        batch_size: Batch size for training and validation
        epochs: Number of epochs to train for
    """

    # Training
    if batch_size is None:
        batch_size = 128
    if epochs is None:
        epochs = 20

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Train...')
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              use_multiprocessing=True)


def eval_model(model, x_test, y_test, batch_size=None):
    """
    Evaluated model on provided testing data
    Args:
        model: Trained tensorflow model
        x_test: X testing data
        y_test: Y testing data
        batch_size: Batch size for evaluations (Default 128)

    Returns:
        Loss and accuracy metrics
    """
    if batch_size is None:
        batch_size = 128

    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    return loss, acc


def run_lstm():
    """"""
    # Build LSTM model
    lstm_model = build_lstm_model(num_features=max_features)

    # Assign data to evaluate model (if training in sequence)
    x_eval, y_eval = x_test_140, y_test_140

    # Train on sentiment 140 dataset
    if train_140:
        # Train and evaluate model
        train_model(lstm_model, x_train_140, y_train_140, x_test_140, y_test_140, epochs, batch_size)

        if not train_in_sequence:
            lstm_loss_140, lstm_acc_140 = eval_model(lstm_model, x_test_140, y_test_140)

#<<<<<<< Callum-Simple_Combine
    model_LSTM = False
    train_140 = True
    train_imdb = False
    simple_classifers = True
    model_GRU = False
    num_rows = 100000  # Number of rows to load from data
    max_features = 20000  # Maximum number of features (words) to process
    
            
            
    # Run LSTM Modelling
    if model_LSTM:       
        if train_140:
            # Load Sentiment 140 dataset
            (x_train_140, y_train_140), \
            (x_test_140, y_test_140), vocab_size = datasets.load_sentiment_140(num_words=max_features,
                                                                               num_rows=num_rows,
                                                                               test_split=0.2,
                                                                               seed=69,
                                                                               simple_classifer = simple_classifers)
            # Train and evaluate model
            loss_140, acc_140 = run_lstm_model(x_train_140, y_train_140, x_test_140, y_test_140,
                                               num_features=vocab_size,
                                               metrics=["acc"])
#=======
            # Show results
            print('Test loss 140:', lstm_loss_140)
            print('Test accuracy 140:', lstm_acc_140)

            # Rebuild model
            lstm_model = build_lstm_model(num_features=max_features)

    # # Train on IMDB dataset
    # if train_imdb:
    #     # Train and evaluate model
    #     train_model(lstm_model, x_train_imdb, y_train_imdb, x_test_imdb, y_test_imdb)
    #
    #     if not train_in_sequence:
    #         lstm_loss_imdb, lstm_acc_imdb = eval_model(lstm_model, x_test_imdb, y_test_imdb)
    #
    #         # Show results
    #         print('Test loss IMDB:', lstm_loss_imdb)
    #         print('Test accuracy IMDB:', lstm_acc_imdb)
    #
    #         # Rebuild model
    #         lstm_model = build_lstm_model(num_features=max_features)

    if train_in_sequence:
        # Evaluate model on assigned eval set
        lstm_loss, lstm_acc = eval_model(lstm_model, x_eval, y_eval)

        # Show results
        print('Test Loss:', lstm_loss)
        print('Test Accuracy:', lstm_acc)


if __name__ == "__main__":
    # ----- SWITCHES -----

    # Models to train
    model_LSTM = True
    model_GRU = False

    train_in_sequence = True  # Train model on multiple datasets, instead of resetting and training seperately

    # Datasets to train model on
    train_140 = True  # Train selected models on sentiment 140 dataset

    # ----- SETUP -----

    grab_data = False
#>>>>>>> master

    if grab_data:
        # Download all files in shared data folder
        download_data.download_from_drive(file_names=['sentiment140.zip'])

#<<<<<<< Callum-Simple_Combine
        if train_imdb:            
            # Load IMDB dataset
            (x_train_imdb, y_train_imdb), \
            (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data(num_words=max_features)
            # Train and evaluate model
            loss_imdb, acc_imdb = run_lstm_model(x_train_imdb, y_train_imdb, x_test_imdb, y_test_imdb,
                                                 num_features=max_features,
                                                 metrics=["acc"])

            # Show results
            print('Test loss IMDB:', loss_imdb)
            print('Test accuracy IMDB:', acc_imdb)
    if simple_classifers:
        if train_140:
            (x_train_140, y_train_140), \
            (x_test_140, y_test_140), vocab_size = datasets.load_sentiment_140(num_words=max_features,
                                                                               num_rows=num_rows,
                                                                               test_split=0.2,
                                                                               seed=69,
                                                                               simple_classifer = simple_classifers)
            run_simple_models(x_train_140, y_train_140, x_test_140, y_test_140)
        if train_imdb:
            run_simple_models(x_train_imdb, y_train_imdb, x_test_imdb, y_test_imdb)
        pass
        
#=======
    # Unzip each zip saved in local data folder
    download_data.unzip_data()

    print("Data organised")

    pathlib.Path("figures/RNN").mkdir(parents=True, exist_ok=True)
    pathlib.Path("logs").mkdir(parents=True, exist_ok=True)

    # ----- LOAD DATA -----

    # Data parameters
    num_rows = 100000  # Number of rows to load from data
    max_features = 20000  # Maximum number of features (words) to process
    maxlen = 100  # Maximum length of sequences - all sequences will be cut or padded to this length

    # Sentiment 140
    print("Loading Sentiment 140...", end="")
    (x_train_140, y_train_140), \
    (x_test_140, y_test_140) = datasets.load_sentiment_140(data_dir="data",
                                                           num_words=max_features,
                                                           num_rows=num_rows,
                                                           maxlen=maxlen,
                                                           test_split=0.2,
                                                           seed=69)
    print(" Done")

    # IMDB
    # print("Loading IMDB...", end="")
    # (x_train_imdb, y_train_imdb), \
    # (x_test_imdb, y_test_imdb) = datasets.load_imdb(num_words=max_features,
    #                                                 maxlen=maxlen)
    # print(" Done")

    # ----- TRAINING -----

    # Training parameters
    epochs = 2
    batch_size = 128

    # Run LSTM Model
    if model_LSTM:
        run_lstm()

#>>>>>>> master
    # Run GRU modelling
    if model_GRU:
        pass
