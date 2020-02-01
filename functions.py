import numpy as np
import pandas as pd
import unidecode

from scipy.stats import spearmanr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.regularizers import l1, l2

import re
import os

def readCSV_DATA(PathData):

	df_dict = {}

	for file in os.listdir(PathData):
	    if file.endswith(".csv"):
	    	pathFinal = PathData+os.sep+file
	    	df_dict[file] = pd.read_csv(pathFinal)

	return df_dict

def visualizeSampleText(df):
    
    #Visualizing texts samples
    for i in np.random.randint(df.shape[0], size=(5)).tolist():
        print('\n----> Question title: <----\n')
        print(df.question_title[i])
        print('\n----> Question body: <----\n')
        print(df.question_body[i])
        print('\n----> Answer: <----\n')
        print(df.answer[i])
        
def cleanText(texts):
    return [unidecode.unidecode(re.sub('[0-9]+[^ ,.]*[0-9]*', '_num_', i).lower().strip()) for i in texts]

def prepareData(df):
    # Concat three main text columns
    df['text_all'] = ['\n\n'.join([i,j,k]) for i,j,k in zip(df.question_title, df.question_body, df.answer)]
    # Clean numbers
    df['text_all_clean'] = cleanText(df.text_all)
    
    df['question_text'] = ['\n\n'.join([i,j]) for i,j in zip(df.question_title, df.question_body)]
    df['question_text_clean'] = cleanText(df.question_text)
    
    df['answer_text_clean'] = cleanText(df.answer)

    return df

def create_model(output_dim, input_dim, loss, reg=l2(0.01), activation='sigmoid', optimizer='adam'):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation=activation, activity_regularizer=reg))

    model.compile(optimizer=optimizer, loss=loss)
    return model


def train_model(model, X_train, y_train, X_test=None, y_test=None, batch_size=64, nb_epoch=100, verbose=1):
    if X_test is not None and y_test is not None:

        model.fit(X_train.toarray(), y_train.as_matrix(),
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  verbose=verbose,
                  validation_data=(X_test.toarray(), y_test.as_matrix()))
    else:
        model.fit(X_train.toarray(), y_train.as_matrix(),
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  verbose=verbose)

    return model


def spearman_corr(y_true, y_pred):

    return np.mean([spearmanr(y_pred[:, i], y_true.iloc[:, i]).correlation for i in range(y_true.shape[1])])