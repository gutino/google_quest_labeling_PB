import numpy as np
import pandas as pd
import unidecode

from scipy.stats import spearmanr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.regularizers import l1, l2

from tqdm import tqdm

import re
import os


def readCSV_DATA(PathData):

    df_dict = {}

    for file in os.listdir(PathData):
        if file.endswith(".csv"):
            pathFinal = PathData + os.sep + file
            df_dict[file] = pd.read_csv(pathFinal)

    return df_dict


def visualizeSampleText(df):

    # Visualizing texts samples
    for i in np.random.randint(df.shape[0], size=(5)).tolist():
        print("\n----> Question title: <----\n")
        print(df.question_title[i])
        print("\n----> Question body: <----\n")
        print(df.question_body[i])
        print("\n----> Answer: <----\n")
        print(df.answer[i])


def cleanText(texts):
    return [
        unidecode.unidecode(re.sub("[0-9]+[^ ,.]*[0-9]*", "_num_", i).lower().strip())
        for i in texts
    ]


def prepareData(df):
    # Concat three main text columns
    df["text_all"] = [
        "\n\n".join([i, j, k])
        for i, j, k in zip(df.question_title, df.question_body, df.answer)
    ]
    # Clean numbers
    df["text_all_clean"] = cleanText(df.text_all)

    df["question_text"] = [
        "\n\n".join([i, j]) for i, j in zip(df.question_title, df.question_body)
    ]
    df["question_text_clean"] = cleanText(df.question_text)

    df["answer_text_clean"] = cleanText(df.answer)

    return df


def create_model(
    output_dim, input_dim, loss, reg=l2(0.01), activation="sigmoid", optimizer="adam"
):
    model = Sequential()
    model.add(
        Dense(
            output_dim,
            input_dim=input_dim,
            activation=activation,
            activity_regularizer=reg,
        )
    )

    model.compile(optimizer=optimizer, loss=loss)
    return model


def train_model(
    model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    batch_size=64,
    nb_epoch=100,
    verbose=1,
):
    if type(X_train) != np.ndarray:
        X_train = X_train.toarray()

    if type(y_train) != np.ndarray:
        y_train = y_train.as_matrix()

    if X_test is not None and y_test is not None:

        if type(X_test) != np.ndarray:
            X_test = X_test.toarray()

        if type(y_test) != np.ndarray:
            y_test = y_test.as_matrix()

        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=verbose,
            validation_data=(X_test, y_test),
        )
    else:
        model.fit(
            X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=verbose
        )

    return model


def spearman_corr(y_true, y_pred):
    return np.mean(
        [
            j
            for j in [
                spearmanr(y_pred[:, i], y_true.iloc[:, i]).correlation
                for i in range(y_true.shape[1])
            ]
            if not np.isnan(j)
        ]
    )


def word2vec_vectorizer(texts, word_tokenize, vocab):
    vectorized = np.zeros((len(texts), 150))

    for i, text in tqdm(enumerate(texts), total=len(texts)):
        vec_doc = np.zeros(150)
        words = set(word_tokenize(text))

        for word in words:
            vec_doc = vec_doc + vocab[word]
        vectorized[i] = vec_doc / len(words)

    return vectorized


def make_submission_df(sub, question_X, answer_X, question_model, answer_model):
    question_y_hat = question_model.predict(question_X)
    answer_y_hat = answer_model.predict(answer_X)

    y_hat_test = np.hstack((question_y_hat, answer_y_hat))

    for col_index, col in enumerate(list(sub.columns[1:])):
        sub[col] = y_hat_test[:, col_index]

    return sub


def divide_ans_quest(df):
    question_lst = []
    answer_lst = []

    for column in df.columns[11:]:
        match = re.match("([a-z]*)_", column)
        if match.group(1) == "question":
            question_lst.append(column)
        else:
            answer_lst.append(column)

    return question_lst, answer_lst
