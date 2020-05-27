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
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
from sklearn.metrics import plot_confusion_matrix

def confusion_matrix_model(opinion_classifier, y_test_opinion, x_test_opinion):
    disp = plot_confusion_matrix(opinion_classifier, x_test_opinion, y_test_opinion,
                            display_labels=['negative', 'positive'],
                            cmap=plt.cm.Blues,
                            normalize="true")
    disp.ax_.set_title("Normalized confusion matrix")
    disp.ax_.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    disp.ax_.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    print("Normalized confusion matrix")
    print(disp.confusion_matrix)
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
     X_train_opinion = X_train_opinion[:30000]
     y_train_opinion = y_train_opinion[:30000]
     x_test_opinion = x_test_opinion[:30000]
     y_test_opinion = y_test_opinion[:30000]
     max_depths = np.linspace(1, 15, 15, endpoint=True)
     min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
     model_params = {
          'gradient_boost_X' : {
               'model': XGBClassifier(),
               'params': {
                    "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
                    "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
                    "min_child_weight" : [ 1, 3, 5, 7 ],
                    "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                    "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
               }
          },
          # 'decision_tree_classifier' : {
          #      'model': DecisionTreeClassifier(),
          #      'params': {
          #           'max_depth': [3, None],
          #           "min_samples_leaf": np.arange(1,9),
          #           "criterion": ["gini", "entropy"]
                    
          #      }
          # },
          # 'svm': {
          #      'model': svm.SVC(),
          #      'params' : {
          #           'C': np.arange(1,7),
          #           "gamma": [0.01, 1, 2, 3]
          #      }  
          # },
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
               
          print("\nStarted ", model_name)
          start_time = timer(None)
          if(model_name != "gradient_boost_X"):
               clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
          else:
               clf = RandomizedSearchCV(mp['model'], param_distributions = mp['params'] , n_iter=5, scoring = 'roc_auc',n_jobs=-1,cv=5,verbose=3)
          clf.fit(X_train_opinion, y_train_opinion)
          time_taken = timer(start_time)

          scores.append({
               'model': model_name,
               'best_score_train': clf.best_score_,
               'best_params_train': clf.best_params_,
               'time_taken': time_taken
          })
          
          print("Finished ", model_name, " best_score: ", clf.best_score_, " best_params ", clf.best_params_)

     df = pd.DataFrame(scores,columns=['model','best_score_train','best_params_train'])
     print(df)      
     opinion_classifier = XGBClassifier(min_child_weight = 1, max_depth= 6, learning_rate= 0.2, gamma= 0.2, colsample_bytree = 0.5)
     opinion_classifier.fit(X_train_opinion, y_train_opinion)
     print(opinion_classifier.score(x_test_opinion, np.array(y_test_opinion)))   
     confusion_matrix_model(opinion_classifier, y_test_opinion, x_test_opinion)
    


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
