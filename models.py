import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

def ensemble_classifers(X_train_opinion, y_train_opinion, x_test_opinion, y_test_opinion):
     X_train_opinion = X_train_opinion[:10000]
     y_train_opinion = y_train_opinion[:10000]
     x_test_opinion = x_test_opinion[:10000]
     y_test_opinion = y_test_opinion[:10000]
     model_params = {
          'decision_tree_classifier' : {
               'model': DecisionTreeClassifier(),
               'params': {
                    'max_depth': [3, None],
                    "min_samples_leaf": np.arange(1,9),
                    "criterion": ["gini", "entropy"]
                    
               }
          },
          'svm': {
               'model': svm.SVC(),
               'params' : {
                    'C': np.arange(1,7),
                    "gamma": [0.01, 1, 2, 3]
               }  
          },
          'random_forest': {
               'model': RandomForestClassifier(),
               'params' : {
                    'n_estimators': np.arange(1,20)
               }
          },
          'logistic_regression' : {
               'model': LogisticRegression(solver='liblinear',multi_class='auto'),
               'params': {
                    'C': np.arange(1,20)
               }
          },
          'k_nearest_neighbour' : {
               'model': KNeighborsClassifier(),
               'params': {
                    'n_neighbors': np.arange(1,25)
               }
          }
     } 

     
     scores = []
     for model_name, mp in model_params.items():
          print("Started ", model_name)
          clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
          clf.fit(X_train_opinion, y_train_opinion)
          scores.append({
               'model': model_name,
               'best_score_train': clf.best_score_,
               'best_params_train': clf.best_params_
          })
          print("Finished ", model_name, " best_score: ", clf.best_score_, " best_params ", clf.best_params_)

     df = pd.DataFrame(scores,columns=['model','best_score_train','best_params_train'])
     print(df)      
     opinion_classifier = svm.SVC(gamma = 2, kernel='rbf', C= 5)
     opinion_classifier.fit(X_train_opinion, y_train_opinion)
     print(opinion_classifier.score(x_test_opinion, np.array(y_test_opinion)))   


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
    #
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(LSTM(lstm_output_size, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(lstm_output_size))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=metrics)

    print(model.summary())

    return model
