# Disable tensorflow debugging information


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D


def lstm(vocab_size,
         embedding_size=128,
         maxlen=100,
         filters=64,
         pool_size=None,
         kernel_size=5,
         lstm_output_size=70,
         metrics=None):

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_size, input_length=maxlen))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(LSTM(lstm_output_size))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=metrics)

    return model
