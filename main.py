# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:04:55 2019

@author: Shanelle Recheta, Junior Data Scientist
"""

# import sys
# inFile = sys.argv[1]
   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, make_scorer, confusion_matrix, f1_score, fbeta_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
import gc

# Set Options for display
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.float_format = '{:.2f}'.format

#Filter Warnings
import warnings
warnings.filterwarnings('ignore')

#make wider graphs
sns.set(rc={'figure.figsize':(12,5)})

#set color palette
sns.set_palette('bright')
# df = pd.read_csv("hscabra_knnimpute_15pops.csv")
df = pd.read_csv("10_hscabra_knnimpute_15pops.csv")
df = df.drop(df.index[528:])
df.set_index('Index',inplace=True)
df.info()
df.head()
df.describe()
df_features_only = df.drop("POP", axis=1)
col = list(df_features_only.columns) 
features = col
print(features)
# Separating out the features
X = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['POP']].values
# count the number of populations and individual per population

pd.crosstab(index=df['POP'], columns='count').sort_values(['count'], ascending=False).head(15)


#### Building the ML models ##########################


######## STANDARDIZE THE DATA #######################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42)


# lets do the standard 80/20 train-test split

"""

####### LOGISTIC REGRESSION ##########

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

model_1 = LogisticRegression(solver='lbfgs')
model_1.fit(X_train, y_train)
predictions_LR = model_1.predict(X_test)

print('Logistic regression accuracy:', accuracy_score(predictions_LR, y_test))
print('')
print('Confusion matrix:')
print(confusion_matrix(y_test,predictions_LR))
print(classification_report(y_test,predictions_LR))

########### DECISION TREES ###############

from sklearn.tree import DecisionTreeClassifier

model_2 = DecisionTreeClassifier()
model_2.fit(X_train, y_train)
predictions_DT = model_2.predict(X_test)

print('Decision tree accuracy:', accuracy_score(predictions_DT, y_test))
print('')
print('Confusion matrix:')
print(confusion_matrix(y_test,predictions_DT))
print(classification_report(y_test,predictions_DT))

############### LINEAR SVM #######################

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

model_3 = SVC(kernel='linear')
model_3.fit(X_train, y_train)
predictions_SVC = model_3.predict(X_test)
print('Support Vector Classifier (kernel: linear) accuracy:', accuracy_score(predictions_SVC, y_test))
print('')
print('Confusion matrix:')
print(confusion_matrix(y_test,predictions_SVC))
print(classification_report(y_test,predictions_SVC))


############# NAIVE BAYES ###########################

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model_4 = GaussianNB()

# Train the model using the training sets
model_4.fit(X_train, y_train)
predictions_NB = model_4.predict(X_test)
print('Naive Bayes accuracy:', accuracy_score(predictions_NB, y_test))
print('')
print('Confusion matrix:')
print(confusion_matrix(y_test,predictions_NB))
print(classification_report(y_test,predictions_NB))

############## KNN ######################################

from sklearn.neighbors import KNeighborsClassifier
model_5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model_5.fit(X_train, y_train)

predictions_KNN = model_5.predict(X_test)
print('KNN accuracy:', accuracy_score(predictions_KNN, y_test))
print('')
print('Confusion matrix:')
print(confusion_matrix(y_test,predictions_KNN))
print(classification_report(y_test,predictions_KNN))

############ MultiLayer Perceptron ################################

from sklearn.preprocessing import MinMaxScaler

#Sigmoid activation
scaler = MinMaxScaler(feature_range=(0,1))

#TANH activation
#scaler = MinMaxScaler(feature_range=(-1,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

model_7 = MLPClassifier(hidden_layer_sizes=(5),max_iter=10000,learning_rate_init=0.001,activation='logistic')
model_7.fit(X_train,y_train)
predictions_MLP = model_7.predict(X_test)
print('MultiLayer Perceptron accuracy:', accuracy_score(predictions_MLP, y_test))
print('')
print('Confusion matrix:')
print(confusion_matrix(y_test,predictions_MLP))
print(classification_report(y_test,predictions_MLP))


"""

#################  RANDOM FOREST ###########################
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# Create the model with 100 trees
model_6 = RandomForestClassifier(n_estimators=200, 
                               bootstrap = True,
                               max_features = 'sqrt')

model_6.fit(X_train, y_train)

predictions_RF = model_6.predict(X_test)
# print('Random Forest x GridSearchCV accuracy:', accuracy_score(y_test,predictions_RF))
# print('')
# print('Confusion matrix:')
# print(confusion_matrix(y_test,predictions_RF))
# print(classification_report(y_test,predictions_RF))

"""


############### Using TPOT to automate parameter tuning #########

from tpot import TPOTClassifier
tpot = TPOTClassifier(config_dict=None, crossover_rate=0.1, cv=5,
               disable_update_check=False, early_stop=None, generations=20,
               max_eval_time_mins=5, max_time_mins=None, memory=None,
               mutation_rate=0.9, n_jobs=1, offspring_size=None,
               periodic_checkpoint_folder=None, population_size=5,
               random_state=None, scoring='f1_macro', subsample=1.0,
               template=None, use_dask=False, verbosity=20, warm_start=False)


tpot.fit(X_train, y_train)
predictions_TPOT = tpot.predict(X_test)

print(tpot.score(X_test, y_test))
print('TPOT accuracy:', accuracy_score(predictions_TPOT, y_test))
print('')
print('Confusion matrix:')
print(confusion_matrix(y_test,predictions_TPOT))
print('')
print(classification_report(y_test,predictions_TPOT))
print('')



######## MEASURING EXECUTION TIMES ################################

# import timeit

"""

####### saing the model ##########################################

from sklearn.externals import joblib

# save the model to disk
filename = 'popgen_model.sav'
joblib.dump(model_6, filename)

