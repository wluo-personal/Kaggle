import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix, hstack, vstack




def tfidf_data_process(train_sentence, test_sentence, word_ngram, word_max, word_min_df=1, word_max_df=1.0,
                       char_ngram=(0,0), char_max=100000, char_min_df=1, char_max_df=1.0):
    """
    Params:
        train_sentence: pd.Series. Usually the sentence column of a dataframe.
            e.g. train['comment_text]
        test_sentence: pd.Series. Usually the sentence column of a dataframe.
            e.g. test['comment_text]
        
        word_ngram, word_max, word_min_df, word_max_df, char_ngram, char_max, char_min_df, char_max_df: tdidf params

    return :x_train: sparse matrix
            y_train: DataFrame (containing all label columns)
            x_test: sparse matrix
            data_id: str, represents params
    """ 
    data_id = 'tfidf_word_{}_{}_{}_{}'.format(word_ngram, word_max, word_min_df, word_max_df)
    
    word_vectorizer = TfidfVectorizer(ngram_range=word_ngram, #1,3
                                        strip_accents='unicode',
                                        max_features=word_max,
                                        min_df = word_min_df,
                                        max_df = word_max_df,
                                        analyzer='word',
                                        stop_words='english',
                                        sublinear_tf=True,
                                        token_pattern=r'\w{1,}')
    print('fitting word')
    word_vectorizer.fit(train_sentence.values)
    print('transforming train word')
    train_word = word_vectorizer.transform(train_sentence.values)
    print('transforming test word')
    test_word = word_vectorizer.transform(test_sentence.values)

    y_train = train[label_cols]

    if char_ngram == (0,0):
        print('Done')
        return (train_word, y_train, test_word, data_id)

    else:
        data_id = '{}_char_{}_{}_{}_{}'.format(data_id, char_ngram, char_max, char_min_df, char_max_df)

        char_vectorizer = TfidfVectorizer(ngram_range=char_ngram,  #2,5
                                          strip_accents='unicode',
                                          max_features=char_max, #200000
                                          min_df = char_min_df,
                                          max_df = char_max_df,
                                          analyzer='char',
                                          sublinear_tf=True)

        print('fitting char')
        char_vectorizer.fit(train_sentence_retain_punctuation.values)
        print('transforming train char')
        train_char = char_vectorizer.transform(train_sentence.values)
        print('transforming test char')
        test_char = char_vectorizer.transform(test_sentence.values)

        x_train = hstack((train_char, train_word), format='csr')
        x_test = hstack((test_char, test_word), format='csr')

        print('Done')
        return (x_train, y_train, x_test, data_id)
