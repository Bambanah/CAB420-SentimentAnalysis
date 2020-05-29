import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
from sklearn.metrics import plot_confusion_matrix
import scikitplot as skplt
from tensorflow.keras.utils import plot_model
import os
from sklearn.metrics import classification_report, confusion_matrix
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def confusion_matrix_model(opinion_classifier, y_test_opinion, x_test_opinion, simple = False):
     positive_bias_threshold = None
     if simple:
          disp = plot_confusion_matrix(opinion_classifier, x_test_opinion, y_test_opinion,
                                        display_labels=['negative', 'positive'],
                                        cmap=plt.cm.Blues,
                                        normalize="true")
          disp.ax_.set_title("Normalized confusion matrix")
          print("Normalized confusion matrix")
          print(disp.confusion_matrix)
          plt.show()
          skplt.metrics.plot_roc_curve(y_test_opinion, opinion_classifier.predict(x_test_opinion))
          plt.show()
     else:
          from sklearn.metrics import roc_curve
          y_pred_keras = opinion_classifier.predict(x_test_opinion).ravel()
          fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_opinion, y_pred_keras)
          fnr = 1 - tpr_keras
          eer_threshold = fpr_keras[np.nanargmin(np.absolute((fnr - fpr_keras)))]
          print(eer_threshold)
          from sklearn.metrics import auc
          auc_keras = auc(fpr_keras, tpr_keras)
          plt.figure(1)
          plt.plot([0, 1], [0, 1], 'k--')
          plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
          plt.xlabel('False positive rate')
          plt.ylabel('True positive rate')
          plt.title('ROC curve')
          plt.legend(loc='best')
          plt.show()
          disp = confusion_matrix(y_test_opinion, opinion_classifier.predict_classes(x_test_opinion))
          print(disp)
          
          print(len(y_pred_keras))
          positive_bias_threshold = eer_threshold - 0.02
          y_pred_keras = np.where(y_pred_keras > positive_bias_threshold, 1, y_pred_keras)
          y_pred_keras = np.where(y_pred_keras < positive_bias_threshold, 0, y_pred_keras)
          print(y_pred_keras)
          print(len(y_pred_keras))

          cm = confusion_matrix(y_test_opinion, y_pred_keras)
          fig = plt.figure()
          ax = fig.add_subplot(111)
          cax = ax.matshow(cm)
          plt.title('Confusion matrix of the classifer with the threshold changed to classifer more neutral text as positive')
          fig.colorbar(cax)
          plt.xlabel('Predicted')
          plt.ylabel('True')
          plt.show()
          print(disp)
     return positive_bias_threshold

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        time_taken = str('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        return time_taken


def ensemble_classifers(X_train_opinion, y_train_opinion, x_test_opinion, y_test_opinion):
    X_train_opinion = X_train_opinion[:30000]
    y_train_opinion = y_train_opinion[:30000]
    x_test_opinion = x_test_opinion[:30000]
    y_test_opinion = y_test_opinion[:30000]
    max_depths = np.linspace(1, 15, 15, endpoint=True)
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    model_params = {
        # 'gradient_boost_X': {
        #     'model': XGBClassifier(),
        #     'params': {
        #         "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        #         "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        #         "min_child_weight": [1, 3, 5, 7],
        #         "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        #         "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
        #     }
        # },
        # 'decision_tree_classifier': {
        #     'model': DecisionTreeClassifier(),
        #     'params': {
        #         'max_depth': [3, None],
        #         "min_samples_leaf": np.arange(1, 9),
        #         "criterion": ["gini", "entropy"]
        #     }
        # },
        # 'svm': {
        #     'model': svm.SVC(),
        #     'params': {
        #         'C': np.arange(1, 7),
        #         "gamma": [0.01, 1, 2, 3]
        #     }
        # },
        # 'random_forest': {
        #     'model': RandomForestClassifier(),
        #     'params': {
        #         'n_estimators': np.arange(1, 20)
        #     }
        # },
        'logistic_regression': {
            'model': LogisticRegression(solver='liblinear', multi_class='auto'),
            'params': {
                'C': np.arange(1, 20)
            }
        },
        # 'k_nearest_neighbour': {
        #     'model': KNeighborsClassifier(),
        #     'params': {
        #         'n_neighbors': np.arange(1, 25)
        #     }
        # }
    }

    scores = []
    for model_name, mp in model_params.items():

        print("\nStarted ", model_name)
        start_time = timer(None)
        if model_name != "gradient_boost_X":
            clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        else:
            clf = RandomizedSearchCV(mp['model'], param_distributions=mp['params'], n_iter=5, scoring='accuracy',
                                     n_jobs=-1, cv=5, verbose=3)
        clf.fit(X_train_opinion, y_train_opinion)
        time_taken = timer(start_time)

        scores.append({
            'model': model_name,
            'best_score_train': clf.best_score_,
            'best_params_train': clf.best_params_,
            'time_taken': time_taken
        })

        print("Finished ", model_name, " best_score: ", clf.best_score_, " best_params ", clf.best_params_)

    df = pd.DataFrame(scores, columns=['model', 'best_score_train', 'best_params_train'])
    opinion_classifier = LogisticRegression(C=1)
    opinion_classifier.fit(X_train_opinion, y_train_opinion)
    print(opinion_classifier.score(x_test_opinion, np.array(y_test_opinion)))
    confusion_matrix_model(opinion_classifier, y_test_opinion, x_test_opinion)


def gru(vocab_size,
         embedding_size=128,
         filters=64,
         pool_size=None,
         kernel_size=5,
         gru_output_size=70):
    model = Sequential()

    model.add(Embedding(vocab_size, embedding_size))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(GRU(gru_output_size, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(gru_output_size))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    metrics = ['acc']

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=metrics)

    plot_model(model, to_file="figures/RNN/GRU_design.png", rankdir="TB")  # TB = Vertical, LR = horizontal
    # print(model.summary())

    return model

def lstm(vocab_size,
         embedding_size=128,
         filters=64,
         pool_size=None,
         kernel_size=5,
         lstm_output_size=70):
    model = Sequential()

    model.add(Embedding(vocab_size, embedding_size))
    model.add(Dropout(0.2))

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

    metrics = ['acc']

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=metrics)

    plot_model(model, to_file="figures/RNN/LSTM_design.png", rankdir="TB")  # TB = Vertical, LR = horizontal
    # print(model.summary())

    return model
