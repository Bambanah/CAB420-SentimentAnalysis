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
from sklearn.metrics import roc_curve

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, recall_score
from sklearn.metrics import f1_score
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def confusion_matrix_model(opinion_classifier, y_test_opinion, x_test_opinion, simple = False, model_name = None):
     if simple:
        average_precision = average_precision_score(y_test_opinion, opinion_classifier.predict(x_test_opinion))
        print('Average precision score: {0:0.2f}'.format(average_precision))
        recall = recall_score(y_test_opinion, opinion_classifier.predict(x_test_opinion), average='micro')
        print("Recall score is: {0:0.2f}" .format(recall))
        fscore = f1_score(y_test_opinion, opinion_classifier.predict(x_test_opinion), average='micro')
        print("F1 score is: " + str(fscore))
        disp = plot_confusion_matrix(opinion_classifier, x_test_opinion, y_test_opinion,
                                    display_labels=['negative', 'positive'],
                                    cmap=plt.cm.Blues,
                                    normalize="true")
        title = "Normalized confusion matrix " + model_name
        disp.ax_.set_title(title)
        print("Normalized confusion matrix")
        print(disp.confusion_matrix)
        plt.show()
        try:
            plt.savefig(disp + 'confusion_matrix.png')
        except:
            pass

            # skplt.metrics.plot_roc_curve(y_test_opinion, opinion_classifier.predict(x_test_opinion))
            # plt.show()
            # plt.savefig(model_name, 'roc.png')
        y_pred = opinion_classifier.predict_proba(x_test_opinion)[::,1]
        fpr, tpr, thresholds = roc_curve(y_test_opinion, y_pred)
        from sklearn.metrics import auc
        auc = auc(fpr, tpr)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Classifer (area = ' + str(auc) + ")")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve ' + model_name)
        plt.legend(loc='best')
        plt.show()
        

     else:
          disp = confusion_matrix(y_test_opinion, opinion_classifier.predict_classes(x_test_opinion))
          print(disp)
          y_pred_keras = opinion_classifier.predict(x_test_opinion).ravel()
          fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_opinion, y_pred_keras)
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
    X_train_opinion = X_train_opinion[:10000]
    y_train_opinion = y_train_opinion[:10000]
    x_test_opinion = x_test_opinion[:10000]
    y_test_opinion = y_test_opinion[:10000]
    max_depths = np.linspace(1, 15, 15, endpoint=True)
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    model_params = {
        # 'decision_tree_classifier': {
        #     'model': DecisionTreeClassifier(),
        #     'params': {
        #         'max_depth': [3, None],
        #         "min_samples_leaf": np.arange(1, 9),
        #         "criterion": ["gini", "entropy"]
        #     }
        # },
        'svm': {
            'model': svm.SVC(probability=True),
            'params': {
                'C':[2],
                "gamma": [ 2 ]
            }
        },
        # 'random_forest': {
        #     'model': RandomForestClassifier(),
        #     'params': {
        #         'n_estimators': np.arange(1, 20)
        #     }
        # },
        # 'logistic_regression': {
        #     'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        #     'params': {
        #         'C': np.arange(1, 20)
        #     }
        # },
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
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(X_train_opinion, y_train_opinion)
        time_taken = timer(start_time)
        confusion_matrix_model(clf, y_test_opinion, x_test_opinion, True, model_name)
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
