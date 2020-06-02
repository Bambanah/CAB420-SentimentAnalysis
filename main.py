import os
import pathlib

from sklearn.metrics import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import models
import datasets
import download_data
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def build_lstm_model(num_features,
                     embedding_size=None,
                     input_length=None,
                     kernel_size=None,
                     filters=None,
                     pool_size=None,
                     lstm_output_size=None):
    """
    Builds and compiles an LSTM model with the provided hyper-parameters
    Args:
        input_length:
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
        embedding_size = 20

    if input_length is None:
        input_length = 100

    # Convolution
    if kernel_size is None:
        kernel_size = 5
    if filters is None:
        filters = 64
    if pool_size is None:
        pool_size = 4

    # LSTM
    if lstm_output_size is None:
        lstm_output_size = 32

    print('Build model...')

    lstm_model = models.lstm(num_features,
                             embedding_size=embedding_size,
                             input_length=input_length,
                             kernel_size=kernel_size,
                             filters=filters,
                             pool_size=pool_size,
                             lstm_output_size=lstm_output_size)

    return lstm_model


def build_gru_model(num_features,
                    embedding_size=None,
                    kernel_size=None,
                    filters=None,
                    pool_size=None,
                    gru_output_size=None):
    """
    Builds and compiles an GRU model with the provided hyper-parameters
    Args:
        gru_output_size:
        num_features:
        embedding_size:
        kernel_size:
        filters:
        pool_size:

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

    # GRU
    if gru_output_size is None:
        gru_output_size = 70

    print('Build model...')

    gru_model = models.gru(num_features,
                           embedding_size=embedding_size,
                           kernel_size=kernel_size,
                           filters=filters,
                           pool_size=pool_size,
                           gru_output_size=gru_output_size)

    return gru_model


def train_model(model, name, x_train, y_train, x_test, y_test):
    """
    Trains model on provided data.
    Args:
        name:
        model: A compiled tensorflow model
        x_train: Training X data
        y_train: Training Y data
        x_test: Testing X data
        y_test: Testing Y data
    """

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Train...')

    my_callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                          patience=1, min_lr=1e-7),
        EarlyStopping(monitor="val_loss", patience=3)
    ]

    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        use_multiprocessing=True,
                        callbacks=my_callbacks)

    plot_history(history, name)


def eval_model(model, x_test, y_test, name=None):
    """
    Evaluated model on provided testing data
    Args:
        name:
        model: Trained tensorflow model
        x_test: X testing data
        y_test: Y testing data

    Returns:
        Loss and accuracy metrics
    """

    if name is None:
        name = "Confusion Matrix"
    else:
        name += " Confusion Matrix"

    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    # positive_bias_threshold = models.confusion_matrix_model(model, y_test, x_test)

    def normalise(arr):
        """
        Normalises input array using safe divisors. Prevents dividing by zero.
        @param
        arr: Array to normalise
        @return Normalised array
        """
        arr_float = arr.astype('float')
        safe_sum = arr.sum(axis=1)[:, np.newaxis]
        safe_sum[safe_sum == 0] = 1

        norm_arr = np.around(arr_float / safe_sum, decimals=2)

        return norm_arr

    # Predict test data
    y_pred = model.predict_classes(x_test)

    # Create confusion matrix
    conf = normalise(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    conf_df = pd.DataFrame(conf, index=["negative", "positive"], columns=["negative", "positive"])

    fig = plt.figure(figsize=(5, 4))

    # Plot confusion matrix
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(conf_df, annot=True)
    ax.set(xlabel='Predicted label', ylabel='True label', title=name)

    # Save and display figure
    plt.savefig("figures/RNN/" + name + ".png", bbox_inches="tight")
    plt.show()

    return loss, acc  # , positive_bias_threshold


# --------------- Recurrent Neural Networks ---------------

def run_lstm():
    # Build model
    lstm_model = build_lstm_model(num_features=max_features, input_length=maxlen)

    # Train moel on 140 data
    train_model(lstm_model, "LSTM", x_train_140, y_train_140, x_test_140, y_test_140)

    # --- Evaluation ---

    # Evaluate model
    lstm_loss, lstm_acc = eval_model(lstm_model, x_test_140, y_test_140, name="LSTM")

    # Display results
    print('Test Loss: %.2f' % lstm_loss)
    print('Test Accuracy: %.2f' % lstm_acc)

    # --- Predictions ---

    # Predict on COVID data
    y_pred = lstm_model.predict(processed_covid)

    # Flatten predictions
    y_pred = np.hstack(y_pred)

    # Plot
    plot_predictions(y_pred, "Positivity Trend (LSTM)")


def run_gru():
    """"""
    # Build GRU model
    gru_model = build_gru_model(num_features=max_features)

    # Train model on sentiment 140 data
    train_model(gru_model, "GRU", x_train_140, y_train_140, x_test_140, y_test_140)

    # Evaluate model on assigned eval set
    gru_loss, gru_acc = eval_model(gru_model, x_test_140, y_test_140, name="GRU")

    # Show results
    print('Test Loss: %.2f' % gru_loss)
    print('Test Accuracy: %.2f' % gru_acc)
    # print('Positive bias threshold: ', positive_bias_threshold)

    y_pred = gru_model.predict(processed_covid)
    y_pred = np.hstack(y_pred)

    plot_predictions(y_pred, "Positivity Trend (GRU)")


# --------------- Simple Models ---------------

def run_simple_models(x_train, y_train, x_test, y_test):
    models.ensemble_classifers(x_train, y_train, x_test, y_test)


def run_simple_models_covid(x_train, y_train):
    log_model = LogisticRegression(solver='liblinear', multi_class='auto', C=1)
    log_model.fit(x_train, y_train)
    y_pred = log_model.predict(processed_covid)

    plot_predictions(y_pred, "Positivity Trend (Logistic Regression)")


def run_simple():
    run_simple_models(x_train_140, y_train_140, x_test_140, y_test_140)
    run_simple_models_covid(x_train_140, y_train_140)


def plot_predictions(predictions, title="Positivity Trend"):
    # Create dataframe for plotting
    data = {
        'date': dates,
        'sentiment': predictions
    }
    df = pd.DataFrame(data)

    # Group data by date, and average sentiment values
    df = df.groupby(['date']).mean()

    # Plot predictions
    plt.figure(figsize=(14, 5))
    plt.grid(False)
    ax = sns.lineplot(x="date", y="sentiment", data=df.reset_index())
    ax.set(xlabel='Date', ylabel='Positivity', title=title)
    plt.xticks(rotation=0)

    # Save and display figure
    plt.savefig("figures/" + title + ".png", bbox_inches="tight")
    plt.show()


def plot_history(history, name):
    history = history.history

    # Plot loss
    fig = plt.figure(figsize=[15, 3])

    ax3 = fig.add_subplot(1, 4, 1)
    ax3.set(xlabel="epochs", ylabel="Loss", title="Training Loss")
    ax3.plot(history['loss'])
    ax3.grid()

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set(xlabel="epochs", ylabel="Loss", title="Validation Loss")
    ax2.plot(history['val_loss'])
    ax2.grid()

    # plt.savefig('figures/RNN/Loss.png')
    # plt.show()

    # Plot accuracy
    # fig = plt.figure(figsize=[12, 8])

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.set(xlabel="epochs", ylabel="Accuracy", title="Training Accuracy")
    ax3.plot(history['acc'])
    ax3.grid()

    ax2 = fig.add_subplot(1, 4, 4)
    ax2.set(xlabel="epochs", ylabel="Accuracy", title="Validation Accuracy")
    ax2.plot(history['val_acc'])
    ax2.grid()

    plt.savefig('figures/RNN/' + name + '_Accuracy-Loss.png', bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # ----- SWITCHES -----

    # RNNs
    model_LSTM = False
    model_GRU = True

    # Simple Classifiers
    simple_classifiers = False

    # Reload COVID dataset - leave this off if you don't need it!
    force_covid_reload = False

    # ----- SETUP -----

    grab_data = False

    if grab_data:
        # Download all files in shared data folder
        download_data.download_from_drive(file_names=['sentiment140.zip'])

        # Unzip each zip saved in local data folder
        download_data.unzip_data()

        print("Data organised")

    pathlib.Path("figures/RNN").mkdir(parents=True, exist_ok=True)
    pathlib.Path("figures/Simple").mkdir(parents=True, exist_ok=True)
    pathlib.Path("logs").mkdir(parents=True, exist_ok=True)

    # ----- LOAD DATA -----

    # Data parameters
    data_dir = "data"

    num_rows = 100000  # Number of rows to load from data
    seed = 69
    test_split = 0.2

    max_features = 20000  # Maximum number of features (words) to process
    maxlen = 100  # Maximum length of sequences - all sequences will be cut or padded to this length

    corpus = []
    dates = []

    print("Loading corpus for vectorizer...", end="")

    # Check if exists
    if os.path.isfile(data_dir + "/covid19-tweets/dataframe.csv") and not force_covid_reload:
        covid_data = pd.read_csv(data_dir + "/covid19-tweets/dataframe.csv")
    else:
        covid_data = datasets.load_covid(num_rows=num_rows, seed=seed)
        covid_data.to_csv(data_dir + "/covid19-tweets/dataframe.csv")
    corpus = covid_data["text"]
    dates = pd.to_datetime(covid_data["created_at"])

    print(" Loaded %d rows" % (len(corpus)))
    # Create vectorizer
    print("Creating vectorizer...", end="")
    vectorizer = datasets.create_vectorizer(corpus, max_features=max_features, simple_classifier=simple_classifiers)
    print(" Done")

    # Sentiment 140
    print("Loading Sentiment 140...", end="")
    (x_train_140, y_train_140), \
    (x_test_140, y_test_140) = datasets.load_sentiment_140(vectorizer=vectorizer,
                                                           data_dir=data_dir,
                                                           num_rows=num_rows,
                                                           maxlen=maxlen,
                                                           simple_classifier=simple_classifiers,
                                                           test_split=test_split,
                                                           seed=seed)
    print(" Done")

    # Covid-19 Tweets
    print("Loading COVID-19 Tweets...", end="")
    processed_covid = datasets.preprocess(vectorizer, corpus, maxlen, simple_classifier=simple_classifiers)
    print(" Done")

    # ----- TRAINING -----

    # Training parameters
    epochs = 20
    batch_size = 64

    # Run LSTM Model
    if model_LSTM:
        run_lstm()

    # Run GRU modelling
    if model_GRU:
        run_gru()

    if simple_classifiers:
        run_simple()
