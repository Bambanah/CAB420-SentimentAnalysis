import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing import sequence

from datasets import load_sentiment_140
from models import lstm


def run_lstm_model(metrics=None):

    # Variables
    num_rows = 100000  # Number of rows to load from data
    max_features = 20000  # Maximum number of features (words) to process

    # Training
    batch_size = 30
    epochs = 2

    # Embedding
    maxlen = 100
    embedding_size = 128

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 70

    # Load Sentiment 140 dataset
    (x_train, y_train), (x_test, y_test), vocab_size = load_sentiment_140(num_words=max_features,
                                                                          num_rows=num_rows,
                                                                          test_split=0.2,
                                                                          seed=69)

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')

    lstm_model = lstm(vocab_size,
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
                   validation_data=(x_test, y_test))
    lstm_score, lstm_acc = lstm_model.evaluate(x_test, y_test, batch_size=batch_size)

    return lstm_score, lstm_acc


if __name__ == "__main__":

    score, acc = run_lstm_model(metrics=["acc"])
    print('Test score:', score)
    print('Test accuracy:', acc)

